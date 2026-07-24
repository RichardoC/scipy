"""
Fuzzy Spectral Region Decomposition (fSRD).

A data-driven Koopman / Dynamic Mode Decomposition solver that represents a
nonlinear dynamical system as a *tree of local linear operators* blended by
fuzzy membership functions, rather than by a single global operator.

The method (arXiv:2607.17990) recursively partitions the ``(row, column)``
index grid of a snapshot matrix into fuzzy, overlapping regions along oblique
or axis-aligned boundaries, fits a regularized exact-DMD operator inside each
region, and selects how many regions to keep by the Bayesian information
criterion of the whole model.  Local reconstructions are recombined with
sigmoidal membership weights that form a partition of unity.
"""
import numpy as np
from dataclasses import dataclass

from scipy._lib._array_api import xp_capabilities

# ``FSRDResult``/``FSRDRegion`` are return types, not directly-constructed API;
# following the scipy convention for result objects they are kept out of
# ``__all__`` (and the subpackage namespace) and documented via ``fsrd``'s
# ``Returns`` section instead.
__all__ = ['fsrd']

# Numerical guards and hard limits (the limits bound worst-case cost so that a
# single call with adversarial hyper-parameters cannot exhaust memory or time).
_EPS = 1e-12
_MAX_DEPTH = 24              # hard ceiling on the tree depth argument
_MAX_NODES = 4096            # ceiling on the total number of regions grown
_MAX_ELEMENTS = 50_000_000   # ceiling on ``M * (T + forecast)`` output cells
_LOG_CLIP = 700.0            # exp(700) < float64 max; overflow-only guard


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FSRDRegion:
    """A single local Koopman/DMD operator produced by `fsrd`.

    Attributes
    ----------
    eigenvalues : ndarray
        Continuous-time eigenvalues :math:`\\omega = \\ln(\\lambda)/\\Delta t`
        of the local operator.  The real part is the growth/decay rate and the
        imaginary part the angular frequency.
    modes : ndarray
        The DMD modes :math:`\\Phi` (columns), shape ``(f, r)`` where ``f`` is
        the number of rows of the local region and ``r`` its fitted rank.
    amplitudes : ndarray
        The mode amplitudes :math:`b`.
    bounding_box : tuple of int
        ``(row_start, row_stop, col_start, col_stop)`` of the region within the
        original snapshot matrix.
    level : int
        Depth of the region in the decomposition tree (0 is the root).
    split_vector : ndarray or None
        The sigmoid coefficient vector ``[v0, v1, v2]`` of the split that
        created this region (``None`` for the root).
    rank : int
        Fitted rank of the local operator.
    """
    eigenvalues: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    bounding_box: tuple
    level: int
    split_vector: "np.ndarray | None" = None
    rank: int = 0


@dataclass
class FSRDResult:
    """Result of a Fuzzy Spectral Region Decomposition.

    Attributes
    ----------
    reconstruction : ndarray
        The fuzzy-weighted sum of the local reconstructions, shape
        ``(m, t + forecast)``.  Real if the input was real.
    regions : list of FSRDRegion
        The local operators, one per leaf of the decomposition tree.
    n_regions : int
        Number of local operators (leaves).
    bic : float
        Bayesian information criterion of the final model (lower is better).
        An additive constant (``N log(2*pi) + N``) is dropped; the value is
        internally consistent but not comparable to a BIC from another tool.
    """
    reconstruction: np.ndarray
    regions: list
    n_regions: int
    bic: float


# ---------------------------------------------------------------------------
# Low level numerical helpers (NumPy only)
# ---------------------------------------------------------------------------
def _sigmoid(u1, u2, v, tau):
    """Sigmoidal split activation over the (row, col) grid -> (m, t) array.

    ``Omega(u) = 1 / (1 + exp(-tau (v0 + v1 u1 + v2 u2)))``.
    """
    z = v[0] + v[1] * u1[:, None] + v[2] * u2[None, :]
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-tau * z))


def _centroid(phi, u1, u2):
    """Activation-weighted centre of a membership map."""
    w = phi.sum()
    if w <= 0:
        return np.array([u1.mean(), u2.mean()])
    c1 = (phi.sum(axis=1) @ u1) / w
    c2 = (phi.sum(axis=0) @ u2) / w
    return np.array([c1, c2])


def _steepness(v, dc, mu):
    """Per-split sigmoid steepness ``tau = 20 / (mu ||v_dir|| ||dc||)``.

    Only the orientation components ``v[1:]`` enter the norm; the offset
    ``v[0]`` positions the boundary and is excluded deliberately.
    """
    nv = np.linalg.norm(v[1:])
    ndc = np.linalg.norm(dc)
    denom = mu * nv * ndc
    if denom <= _EPS:
        return 20.0 / max(mu, 1e-6)
    return 20.0 / denom


def _optimal_amplitudes(phi, lam, block):
    """Least-squares DMD mode amplitudes fitted over *all* snapshots.

    Solves ``min_b || block - Phi diag(b) V ||_F`` with ``V[i, k] = lam_i^k``.
    This is the standard "optimal amplitude" fit, used here as a more robust
    alternative to the paper's single-initial-condition fit ``b = Phi^+ x0``.

    The Vandermonde is evaluated in log space and normalised so each mode's
    row has unit maximum magnitude over the window, which makes the normal
    equations well conditioned and overflow-free even for growing eigenvalues;
    the normalisation is undone exactly afterwards, so the returned amplitudes
    are those of the *unmodified* problem.
    """
    n = block.shape[1]
    k = np.arange(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        loglam = np.log(np.where(lam == 0, np.finfo(float).tiny, lam))
    scale = np.maximum(loglam.real, 0.0) * (n - 1)          # >= 0, per mode
    vand = np.exp(loglam[:, None] * k[None, :] - scale[:, None])   # |.| <= 1
    p = (phi.conj().T @ phi) * np.conj(vand @ vand.conj().T)
    q = np.conj(np.diag(vand @ block.conj().T @ phi))
    try:
        b_scaled = np.linalg.solve(p, q)
    except np.linalg.LinAlgError:
        b_scaled = np.linalg.lstsq(p, q, rcond=None)[0]
    return b_scaled * np.exp(-scale)


def _regularized_svd(x1, rcond):
    """Truncated + ridge-regularized + row-pruned SVD of ``x1``.

    Returns ``(U_r, s_r, Vh_r)``.  Small singular values (relative to the
    largest) are truncated, a ridge parameter is chosen by generalized
    cross-validation, and rows of the reconstructed right singular modes with
    negligible relative mean are pruned.  The GCV grid is scaled by the leading
    squared singular value so the selection is invariant to the data's overall
    magnitude.
    """
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return None
    # 1. hard truncation for rank deficiency (relative cutoff)
    keep = s > rcond * s[0]
    r1 = int(np.count_nonzero(keep))
    if r1 < 1:
        return None
    u, s, vh = u[:, :r1], s[:r1], vh[:r1]

    if r1 == 1:
        return u, s, vh

    # 2. ridge parameter by GCV in the SVD basis (B = diag(s^2) + delta I).
    #    The grid is scaled by s[0]**2 so delta lives on the same (squared
    #    singular value) scale as the data, making the choice scale-invariant.
    c = u.conj().T @ x1                       # (r1, q) projection onto span(U)
    cn = np.sum(np.abs(c) ** 2, axis=1)       # energy per singular direction
    r0 = np.linalg.norm(x1) ** 2 - cn.sum()   # energy outside span(U)
    r0 = max(r0, 0.0)
    nelem = x1.size
    s2 = s ** 2
    best = None
    for delta in np.logspace(-8, 1, 14) * s2[0]:
        g = s2 / (s2 + delta)                 # shrinkage factors
        rss = r0 + np.sum(((1.0 - g) ** 2) * cn)
        df = g.sum()
        gcv = nelem * rss / max((nelem - df) ** 2, _EPS)
        if best is None or gcv < best[0]:
            best = (gcv, delta)
    delta = best[1]

    # 3. reconstruct the ridge-regularized right singular modes
    shrink = s / (s2 + delta)
    s_map = shrink[:, None] * c               # (r1, q)

    # 4. prune rows with negligible relative mean
    mu = np.abs(s_map).mean(axis=1)
    mu = mu / mu.max()
    rows = mu > 1e-2
    if not rows.any():
        rows = np.zeros_like(rows)
        rows[np.argmax(mu)] = True
    return u[:, rows], s[rows], vh[rows]


def _dmd_operator(block, dt, rcond):
    """Exact DMD with the regularized SVD.  Returns ``(omega, phi, b)``."""
    if block.shape[1] < 2:
        return None
    x1, x2 = block[:, :-1], block[:, 1:]
    reg = _regularized_svd(x1, rcond)
    if reg is None:
        return None
    u_r, s_r, vh_r = reg
    v_r = vh_r.conj().T / s_r                 # V_r Sigma_r^{-1}
    atil = u_r.conj().T @ x2 @ v_r
    try:
        lam, w = np.linalg.eig(atil)
    except np.linalg.LinAlgError:
        return None                           # non-convergent node -> skip
    phi = x2 @ v_r @ w                        # exact DMD modes
    safe = np.where(lam == 0, np.finfo(float).tiny, lam)
    omega = np.log(safe) / dt
    b = _optimal_amplitudes(phi, lam, block)
    return omega, phi, b


def _dmd_reconstruct(model, n, dt):
    """Evaluate a fitted DMD model over ``n`` successive columns.

    Computes each mode's contribution ``b_i lambda_i^k`` directly in log space
    with an overflow-only guard.  In-region reconstruction is therefore exact
    (bounded, since the least-squares amplitudes fit bounded data); forecasting
    a growing mode grows correctly, saturating only at the float64 ceiling.
    """
    omega, phi, b = model
    loglam = omega * dt                       # = log(lambda)
    k = np.arange(n)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        logb = np.log(b.astype(complex))
        expo = logb[:, None] + loglam[:, None] * k[None, :]
        expo = np.minimum(expo.real, _LOG_CLIP) + 1j * expo.imag
        term = np.exp(expo)                   # (r, n) == b_i * lambda_i^k
    return phi @ term


# ---------------------------------------------------------------------------
# Topological transform for oblique regions
# ---------------------------------------------------------------------------
def _row_spans(mask):
    """First/last in-region column index per row (``-1`` if the row is empty)."""
    spans = []
    for row in mask:
        idx = np.flatnonzero(row)
        if idx.size == 0:
            spans.append((-1, -1))
        else:
            spans.append((int(idx[0]), int(idx[-1])))
    return spans


def _topological_transform(block, spans, q):
    """Expand each row's in-region run to width ``q`` by interpolation."""
    f = block.shape[0]
    out = np.zeros((f, q), dtype=block.dtype)
    for j in range(f):
        a, b = spans[j]
        if a < 0:
            continue
        seg = block[j, a:b + 1]
        if seg.size == 1:
            out[j, :] = seg[0]
        elif seg.size == q:
            out[j, :] = seg
        else:
            xp = np.linspace(0.0, 1.0, seg.size)
            xq = np.linspace(0.0, 1.0, q)
            out[j, :] = np.interp(xq, xp, seg.real)
            if np.iscomplexobj(seg):
                out[j, :] = out[j, :] + 1j * np.interp(xq, xp, seg.imag)
    return out


def _inverse_topological_transform(dense, spans, f, q):
    """Compress each dense row back onto its in-region run."""
    out = np.zeros((f, q), dtype=complex)
    for j in range(f):
        a, b = spans[j]
        if a < 0:
            continue
        lj = b - a + 1
        row = dense[j]
        if lj == 1:
            out[j, a] = row.mean()
        elif lj == q:
            out[j, a:b + 1] = row
        else:
            xp = np.linspace(0.0, 1.0, q)
            xq = np.linspace(0.0, 1.0, lj)
            comp = np.interp(xq, xp, row.real) + 1j * np.interp(xq, xp, row.imag)
            out[j, a:b + 1] = comp
    return out


# ---------------------------------------------------------------------------
# Region fit + tree machinery
# ---------------------------------------------------------------------------
class _Node:
    # ``path`` is a list of (v, tau, side, split_id); the two children of one
    # split share a unique ``split_id``, which is how sibling pairs are found
    # during pruning (no object-identity tricks required).
    __slots__ = ('phi', 'level', 'path', 'model', 'bbox', 'oblique', 'spans',
                 'split_vector', 'sse')

    def __init__(self, phi, level, path):
        self.phi = phi
        self.level = level
        self.path = path
        self.model = None
        self.bbox = None
        self.oblique = False
        self.spans = None
        self.split_vector = path[-1][0] if path else None
        self.sse = None                       # cached `_region_sse` (see below)


def _fit_node(a, node, dt, rcond, eta):
    """Fit the local DMD operator for ``node`` in place."""
    phi = node.phi
    rows = np.flatnonzero(phi.max(axis=1) > eta)
    cols = np.flatnonzero(phi.max(axis=0) > eta)
    if rows.size < 1 or cols.size < 2:
        node.model = None
        return
    m0, m1 = rows[0], rows[-1] + 1
    t0, t1 = cols[0], cols[-1] + 1
    node.bbox = (int(m0), int(m1), int(t0), int(t1))
    block = a[m0:m1, t0:t1]
    mask = phi[m0:m1, t0:t1] > eta
    node.oblique = not mask.all()
    if node.oblique:
        spans = _row_spans(mask)
        node.spans = spans
        block = _topological_transform(block, spans, block.shape[1])
    node.model = _dmd_operator(block, dt, rcond)


def _node_local_recon(node, dt, shape):
    """Local reconstruction placed into a full ``shape`` grid (zeros outside)."""
    m0, m1, t0, t1 = node.bbox
    q = t1 - t0
    dense = _dmd_reconstruct(node.model, q, dt)      # (f, q)
    if node.oblique:
        dense = _inverse_topological_transform(dense, node.spans, m1 - m0, q)
    out = np.zeros(shape, dtype=complex)
    out[m0:m1, t0:t1] = dense
    return out


def _bic(sse, n, k):
    mse = sse / n + np.finfo(float).tiny
    return n * np.log(mse) + k * np.log(n)


def _region_sse(a, node, dt):
    """Membership-weighted squared error of a node's local model.

    The membership weights the *squared* error (rather than the residual), so
    that the weights enter linearly.  Because a split's two children satisfy
    ``phi_left + phi_right == phi_parent`` pointwise, the children's summed
    weighted error then equals the parent's exactly whenever the underlying
    reconstruction is unchanged: a split can only lower this term by actually
    fitting the data better.  Weighting the residual instead would apply
    ``phi**2``, and since ``w**2 + (1 - w)**2 <= 1`` any split would lower the
    term for free, biasing the model-order selection towards more regions.
    """
    recon = _node_local_recon(node, dt, a.shape)
    return float(np.sum(node.phi * np.abs(recon - a) ** 2))


def _leaf_sse(a, node, dt):
    """`_region_sse` for a node, cached (the model and data never change)."""
    if node.sse is None:
        node.sse = _region_sse(a, node, dt)
    return node.sse


def _model_bic(a, nodes, n, dt, theta):
    """Information criterion of a whole candidate model (all of its leaves).

    Split and prune decisions compare this quantity, following the criterion of
    [1]_, which is defined over the *complete* model rather than one region: the
    memberships of the current leaves form a partition of unity, so the summed
    membership-weighted errors are a convex combination of the local errors and
    hence an additive proxy for the global error.  Judging a split by the split
    node's error alone would instead reward shrinking a residual that is already
    negligible in the full model, which drives over-segmentation.
    """
    sse = sum(_leaf_sse(a, nd, dt) for nd in nodes)
    k = sum(_region_k(nd, theta) for nd in nodes)
    return _bic(sse, n, k)


def _region_k(node, theta):
    r = 0 if node.model is None else node.model[0].size
    return r * (theta ** node.level)


_MIN_ROWS_SPLIT = 4       # need >=4 rows to attempt a row split (children >=2)
_MIN_COLS_SPLIT = 8       # need >=8 cols to attempt a column split (children >=4)
_FRACS = (0.35, 0.5, 0.65)


def _candidate_vectors(node, u1, u2, oblique):
    """Position-aware split vectors for a node.

    For each orientation the boundary is placed at the fractions in `_FRACS`
    across the region's own extent -- the deterministic stand-in for the
    paper's PSO search over the sigmoid offset ``v0``.  The paper's four fixed
    orientation vectors are the *directions* used here (two axis-aligned, two
    oblique); only the offset varies.
    """
    m0, m1, t0, t1 = node.bbox
    r0, r1 = u1[m0], u1[m1 - 1]
    c0, c1 = u2[t0], u2[t1 - 1]
    out = []
    if t1 - t0 >= _MIN_COLS_SPLIT:                 # vertical (time) splits
        for f in _FRACS:
            p = c0 + f * (c1 - c0)
            out.append(np.array([-p, 0.0, 1.0]))
    if m1 - m0 >= _MIN_ROWS_SPLIT:                 # horizontal (row) splits
        for f in _FRACS:
            p = r0 + f * (r1 - r0)
            out.append(np.array([-p, 1.0, 0.0]))
    if oblique and t1 - t0 >= _MIN_COLS_SPLIT and m1 - m0 >= _MIN_ROWS_SPLIT:
        for f in _FRACS:                           # oblique splits
            c = (r0 + c0) + f * ((r1 + c1) - (r0 + c0))
            out.append(np.array([-c, 1.0, 1.0]))
            c = (r0 - c1) + f * ((r1 - c0) - (r0 - c1))
            out.append(np.array([-c, 1.0, -1.0]))
    return out


def _try_split(a, node, v, sid, u1, u2, dt, rcond, eta, mu):
    """Build the two children of ``node`` under split vector ``v``; fit them."""
    # crisp centroids (tau -> inf) to set the steepness, then soft activation
    z = v[0] + v[1] * u1[:, None] + v[2] * u2[None, :]
    hard = (z > 0).astype(float) * node.phi
    c1 = _centroid(hard, u1, u2)
    c2 = _centroid((node.phi - hard), u1, u2)
    tau = _steepness(v, c2 - c1, mu)
    omega = _sigmoid(u1, u2, v, tau)
    left = _Node(node.phi * omega, node.level + 1,
                 node.path + [(v, tau, 0, sid)])
    right = _Node(node.phi * (1.0 - omega), node.level + 1,
                  node.path + [(v, tau, 1, sid)])
    _fit_node(a, left, dt, rcond, eta)
    _fit_node(a, right, dt, rcond, eta)
    if left.model is None or right.model is None:
        return None
    return left, right


def _forward_pass(a, u1, u2, dt, rcond, eta, mu, theta, max_depth, oblique):
    """Grow the decomposition tree one complete level at a time.

    Following [1]_, every splittable leaf of a level is split -- each at the
    position that minimises the model's information criterion -- before the
    level as a whole is assessed, and growth continues while a completed level
    lowers that criterion.  The pass therefore stops one level *after* the best
    one, deliberately over-growing the tree; `_prune` then selects the regions
    that are kept.  Returning the over-grown tree is what allows the backward
    pass to reach models that a greedy split-by-split rule cannot.

    Growth stops at ``max_depth``, when a level admits no split at all, when a
    completed level fails to improve the criterion, or when the number of
    regions reaches ``_MAX_NODES`` (a hard cost cap independent of the
    hyper-parameters).
    """
    n = a.size
    root = _Node(np.ones(a.shape), 0, [])
    _fit_node(a, root, dt, rcond, eta)
    leaves = [root]
    if root.model is None:
        return leaves          # nothing to fit (e.g. an all-zero input)
    best_bic = _model_bic(a, leaves, n, dt, theta)
    sid = 0
    for _ in range(max_depth):
        new_leaves = []
        changed = False
        for idx, leaf in enumerate(leaves):
            if (leaf.model is None or leaf.bbox is None
                    or len(new_leaves) + 2 > _MAX_NODES):
                new_leaves.append(leaf)
                continue
            # the rest of the model: leaves already kept plus those still to come
            others = new_leaves + leaves[idx + 1:]
            best = None
            for v in _candidate_vectors(leaf, u1, u2, oblique):
                pair = _try_split(a, leaf, v, sid, u1, u2, dt, rcond, eta, mu)
                sid += 1
                if pair is None:
                    continue
                child_a, child_b = pair
                split_bic = _model_bic(a, others + [child_a, child_b], n, dt,
                                       theta)
                if best is None or split_bic < best[0]:
                    best = (split_bic, child_a, child_b)
            if best is None:
                new_leaves.append(leaf)
            else:
                # split the level through: whether the extra regions are worth
                # keeping is decided for the level, and then by `_prune`
                new_leaves.extend([best[1], best[2]])
                changed = True
        if not changed:
            break
        leaves = new_leaves
        level_bic = _model_bic(a, leaves, n, dt, theta)
        if level_bic >= best_bic or len(leaves) >= _MAX_NODES:
            break                     # this level overshoots; hand it to `_prune`
        best_bic = level_bic
    return leaves


def _prune(a, leaves, dt, theta, rcond, eta):
    """Backward elimination: collapse sibling pairs back into their parent while
    that does not raise the whole model's BIC.

    This pass performs the model selection, `_forward_pass` having deliberately
    over-grown the tree, as in [1]_.  It runs the tree in reverse: each
    child-parent group is reviewed individually, deepest level first, comparing
    the model with and without that pair of children.  Sweeps repeat until a
    full sweep collapses nothing, so a parent formed by one collapse can itself
    be folded away with its sibling on a later sweep and a whole branch can
    disappear.  [1]_ notes that the resulting tree is not uniquely determined by
    this procedure.
    """
    n = a.size
    changed = True
    while changed and len(leaves) > 1:
        changed = False
        # group siblings by the unique split id of the split that made them
        groups = {}
        for leaf in leaves:
            if not leaf.path:
                continue
            groups.setdefault(leaf.path[-1][3], []).append(leaf)
        pairs = sorted((g for g in groups.values() if len(g) == 2),
                       key=lambda g: -g[0].level)
        for sibs in pairs:
            if any(s not in leaves for s in sibs):
                continue              # already folded away earlier this sweep
            parent = _Node(sibs[0].phi + sibs[1].phi, sibs[0].level - 1,
                           sibs[0].path[:-1])
            _fit_node(a, parent, dt, rcond, eta)
            if parent.model is None:
                continue
            others = [ln for ln in leaves if ln not in sibs]
            if (_model_bic(a, others + [parent], n, dt, theta)
                    <= _model_bic(a, others + sibs, n, dt, theta)):
                leaves = others + [parent]
                changed = True
    return leaves


def _reconstruct(leaves, u1, u2_out, dt, out_cols, orig_cols, shape_rows, real):
    """Blend the leaves' local reconstructions over the output grid.

    Each region is reconstructed over exactly its own bounding box; a region
    whose box reaches the last input column is extended through the forecast
    horizon.  Cells are renormalised so the fuzzy memberships form a partition
    of unity even after the ``eta`` support cutoff.
    """
    accum = np.zeros((shape_rows, out_cols), dtype=complex)
    weight = np.zeros((shape_rows, out_cols))
    for leaf in leaves:
        if leaf.model is None:
            continue
        m0, m1, t0, t1 = leaf.bbox
        col_end = out_cols if t1 >= orig_cols else t1
        qn = col_end - t0
        # membership only over the region's bounding box (rows m0:m1, cols t0:)
        u1r = u1[m0:m1]
        u2r = u2_out[t0:col_end]
        phi = np.ones((m1 - m0, qn))
        for v, tau, side, _sid in leaf.path:
            om = _sigmoid(u1r, u2r, v, tau)
            phi = phi * (om if side == 0 else (1.0 - om))
        dense = _dmd_reconstruct(leaf.model, qn, dt)
        if leaf.oblique:
            spans = [(a0, min(b0, qn - 1)) for a0, b0 in leaf.spans]
            dense = _inverse_topological_transform(dense, spans, m1 - m0, qn)
        accum[m0:m1, t0:col_end] += phi * dense
        weight[m0:m1, t0:col_end] += phi
    weight = np.where(weight > _EPS, weight, 1.0)
    accum = accum / weight
    return accum.real if real else accum


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@xp_capabilities(np_only=True)
def fsrd(a, dt=1.0, *, max_depth=3, theta=1.5, rcond=1e-4, eta=1e-4,
         smoothness=0.05, oblique=True, prune=True, forecast=0,
         check_finite=True):
    r"""
    Fuzzy Spectral Region Decomposition of a dynamical snapshot matrix.

    Learn a piecewise (multi-operator) Koopman / Dynamic Mode Decomposition
    representation of a nonlinear system.  Instead of fitting one global linear
    operator, ``fsrd`` recursively partitions the ``(row, column)`` index grid
    of the snapshot matrix into fuzzy, overlapping regions -- along oblique or
    axis-aligned boundaries -- and fits a regularized exact-DMD operator inside
    each.  How many regions to keep is selected from the data by the Bayesian
    information criterion (BIC) of the whole model, evaluated as the tree is
    grown a level at a time and again as it is pruned back.  The local
    reconstructions are recombined with sigmoidal membership weights that form a
    partition of unity.

    Parameters
    ----------
    a : (M, T) array_like
        Snapshot matrix.  Columns are successive states of the system; rows are
        state coordinates (e.g. sensors, or time-delay copies for a Hankel
        embedding).  ``T`` must be at least 2.
    dt : float, optional
        Positive time step between successive columns, used to convert the
        discrete DMD eigenvalues to continuous-time ones.  Default is 1.
    max_depth : int, optional
        Maximum depth of the decomposition tree, in ``[0, 24]``.  ``0`` fits a
        single global operator.  It is an upper bound: growth also stops once a
        level no longer improves the BIC, and the tree is then pruned back, so
        the returned model is usually shallower.  Each level doubles the regions
        that are grown, so on data that keeps admitting splits both the cost and
        the peak memory roughly double per level; raise it with care.  Default
        is 3.
    theta : float, optional
        BIC complexity-scaling parameter :math:`\Theta \ge 1`; the
        model-complexity penalty of a region grows as :math:`\Theta^{L-1}` with
        its tree level ``L``.  Larger values favour fewer regions.  Default 1.5.
    rcond : float, optional
        Positive relative singular-value cutoff for the local SVD truncation
        (values below ``rcond`` times the largest singular value are dropped).
        Default ``1e-4``, the relative analogue of the absolute cutoff used in
        [1]_ (which rescales its data to ``[0, 1]`` first).
    eta : float, optional
        Membership cutoff in ``(0, 1)``.  A region only occupies grid cells
        where its fuzzy membership exceeds ``eta``.  Default ``1e-4``.
    smoothness : float, optional
        Positive smoothness factor :math:`\mu` controlling the sigmoid boundary
        width (smaller is crisper).  Default ``0.05``.  At a sharp regime
        boundary a smaller value reconstructs the transition more accurately --
        approaching exact recovery in the crisp ``\mu -> 0`` limit -- while a
        larger value blends neighbouring regions more smoothly.
    oblique : bool, optional
        If True (default) both axis-aligned and oblique split orientations are
        tried at each node; if False only axis-aligned (row/column) splits are
        considered, which is faster.
    prune : bool, optional
        Whether to run the backward-elimination pruning pass after growing the
        tree.  Default True.  This pass performs the model selection, so with
        ``prune=False`` the deliberately over-grown tree left by the forward pass
        is returned unchanged and will usually hold more regions than the data
        supports; that setting is intended for diagnostics.  See Notes.
    forecast : int, optional
        Number of additional columns to extrapolate beyond the input.  Default
        0.  ``M * (T + forecast)`` must not exceed 5e7.
    check_finite : bool, optional
        Whether to check that the input contains only finite numbers.  Default
        True.  Disabling may improve performance, but passing non-finite data
        with ``check_finite=False`` gives undefined behaviour (the underlying
        SVD may hang on infinities).

    Returns
    -------
    res : FSRDResult
        The decomposition.  It is a returned object (not meant to be
        constructed directly) with the attributes:

        - ``reconstruction`` : ndarray -- the ``(M, T + forecast)``
          fuzzy-weighted sum of the local reconstructions; real if `a` is real.
        - ``regions`` : list -- the local operators, one per leaf of the
          decomposition tree.  Each element is an ``FSRDRegion`` with attributes
          ``eigenvalues`` (continuous-time
          :math:`\omega = \ln(\lambda)/\Delta t`, real part growth/decay rate
          and imaginary part angular frequency), ``modes`` (the DMD modes
          :math:`\Phi`, one per column), ``amplitudes`` (the mode amplitudes
          :math:`b`), ``bounding_box`` (``(row_start, row_stop, col_start,
          col_stop)`` within the input), ``level`` (depth in the tree, ``0`` at
          the root), ``split_vector`` (the sigmoid coefficients ``[v0, v1, v2]``
          of the split that created the region, ``None`` at the root) and
          ``rank`` (the fitted local rank).
        - ``n_regions`` : int -- the number of local operators (leaves).
        - ``bic`` : float -- the Bayesian information criterion of the final
          model (lower is better); an additive constant is dropped, so the value
          is internally consistent but not comparable to a BIC from another tool.

    Raises
    ------
    ValueError
        If `a` is not a non-empty 2-D array with at least two columns, or if
        any of `dt`, `max_depth`, `theta`, `rcond`, `eta`, `smoothness` or
        `forecast` is outside its valid range, or if the requested output size
        exceeds the internal limit.

    See Also
    --------
    scipy.linalg.svd : Singular value decomposition used by each local fit.
    scipy.linalg.eig : Eigendecomposition of each reduced operator.

    Notes
    -----
    .. versionadded:: 1.19.0

    Each region fits an exact Dynamic Mode Decomposition [2]_.  For a snapshot
    pair ``(X1, X2)`` with ``X2 ~ A X1`` the reduced operator is
    :math:`\tilde{A} = U_r^{H} X_2 V_r \Sigma_r^{-1}`, where
    :math:`U_r \Sigma_r V_r^{H}` is a truncated, ridge-regularized SVD of
    ``X1``; the modes are :math:`\Phi = X_2 V_r \Sigma_r^{-1} W` with ``W`` the
    eigenvectors of :math:`\tilde{A}`, and each state is reconstructed as
    :math:`x(t) = \Phi \operatorname{diag}(b)\, e^{\omega t}`.  The global model
    is :math:`\tilde{X} = \sum_i \phi_i(U) \odot f_i`, a fuzzy-membership
    weighted sum of the local reconstructions ``f_i``.

    This implementation follows the structure of [1]_ -- oblique and
    axis-aligned fuzzy splits, the topological transform for non-rectangular
    regions, BIC-driven forward growth and backward pruning -- but makes a few
    deliberate, documented substitutions where [1]_ defers detail to its
    supplement or leaves a component under-specified:

    - Split orientation is chosen from the paper's four candidate vectors and
      the boundary offset from the fixed fractions ``(0.35, 0.5, 0.65)`` of the
      region, rather than refined by particle-swarm optimization, and candidates
      are ranked by the model's information criterion rather than by the paper's
      weighted-NRMSE net error.  This tends to find better-fitting partitions,
      and so can retain more regions than the paper reports for a given system.
    - The ridge parameter is chosen by generalized cross-validation rather than
      the paper's BIC fixed-point iteration.  It governs the SVD row-pruning
      selection only and does not shrink the returned local operator, so on
      well-conditioned data -- where the selected ridge tends to zero -- each
      region reduces to a plain truncated-SVD DMD fit.
    - Mode amplitudes are fitted by least squares over all snapshots rather
      than from a single initial condition.
    - A region's operator is fitted to its snapshot block directly, rather than
      to the block pre-multiplied by the region's membership; the membership
      weights the model selection and the final blend instead.

    Model order is selected as in [1]_, in two passes over the whole model's
    criterion.  The forward pass splits every splittable region of a level
    before assessing the level, and continues while a completed level lowers the
    criterion; when it is the level comparison that stops the growth, the tree is
    left deliberately over-grown by one level.  The backward pass then runs the
    tree in reverse, reviewing each pair of children deepest-level first and
    collapsing it back into its parent whenever that does not raise the
    criterion, until a full sweep changes nothing.  [1]_ notes that the resulting
    tree is not uniquely determined by this procedure.  Growth also stops at
    ``max_depth`` or at an internal cap on the number of regions, in which case
    the tree is not over-grown.

    The reconstruction and forecast use the fitted eigenvalues as returned in
    ``regions[i].eigenvalues`` (no clipping); a region's in-window
    reconstruction is bounded because its amplitudes are a least-squares fit to
    bounded data, while forecasting a mode with ``|lambda| > 1`` grows as
    expected.  Forecasting is intended for axis-aligned regions; an oblique
    region is not extrapolated along its diagonal beyond the input window, so a
    multi-region forecast that would rely on an oblique region past the data is
    bounded and finite but not accuracy-guaranteed.

    Every decision of both passes is taken on the information criterion of the
    *complete* model, never on that of one region in isolation, as in [1]_; the
    summed membership-weighted region errors are used as an additive proxy for
    the global error, which they equal in the crisp (``smoothness -> 0``) limit.
    The memberships weight the squared errors linearly, so that a split lowers
    the criterion only by fitting the data better.  The reported ``bic`` is a
    diagnostic computed differently: it uses the true global reconstruction error
    and drops an additive constant, so it is not comparable to a BIC from another
    tool, and because it is not the quantity the two passes minimise it need not
    order two models the same way they do -- an over-grown tree obtained with
    ``prune=False`` can carry the lower reported ``bic``.  Use it to compare fits
    of the same model order, not to choose a model order.

    References
    ----------
    .. [1] C. Bokor, M. Cary, D. Morrey, and F. Bonatesta, "fSRD: Fuzzy
           Spectral Region Decomposition -- Automated Multi Operator Koopman
           Representations via an Adaptive Spectral Learning Architecture",
           arXiv:2607.17990.
    .. [2] J. H. Tu et al., "On Dynamic Mode Decomposition: Theory and
           Applications", Journal of Computational Dynamics, 1(2), 2014.

    Examples
    --------
    Recover a single linear (rotational) system with one region:

    >>> import numpy as np
    >>> from scipy.linalg import fsrd
    >>> theta = 0.3
    >>> A = np.array([[np.cos(theta), -np.sin(theta)],
    ...               [np.sin(theta),  np.cos(theta)]])
    >>> x = np.empty((2, 40))
    >>> x[:, 0] = [1.0, 0.0]
    >>> for k in range(39):
    ...     x[:, k + 1] = A @ x[:, k]
    >>> res = fsrd(x, max_depth=0)
    >>> bool(np.allclose(res.reconstruction, x, atol=1e-6))
    True
    >>> res.n_regions
    1

    The imaginary part of the continuous-time eigenvalue recovers the rotation
    frequency:

    >>> bool(np.isclose(np.abs(res.regions[0].eigenvalues.imag).max(), theta,
    ...                 atol=1e-6))
    True

    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")
    if a.size == 0:
        raise ValueError("`a` must not be empty.")
    if a.shape[1] < 2:
        raise ValueError("`a` must have at least two columns (T >= 2).")
    if check_finite and not np.isfinite(a).all():
        raise ValueError("array must not contain infs or NaNs")

    max_depth = int(max_depth)
    forecast = int(forecast)
    if not 0 <= max_depth <= _MAX_DEPTH:
        raise ValueError(f"`max_depth` must be in [0, {_MAX_DEPTH}].")
    if forecast < 0:
        raise ValueError("`forecast` must be non-negative.")
    dt = float(dt)
    if not dt > 0:
        raise ValueError("`dt` must be positive.")
    if not rcond > 0:
        raise ValueError("`rcond` must be positive.")
    if not 0 < eta < 1:
        raise ValueError("`eta` must lie in (0, 1).")
    if not smoothness > 0:
        raise ValueError("`smoothness` must be positive.")
    if not theta >= 1:
        raise ValueError("`theta` must be >= 1.")

    m, t = a.shape
    if m * (t + forecast) > _MAX_ELEMENTS:
        raise ValueError(
            f"requested output size M*(T+forecast) exceeds {_MAX_ELEMENTS}; "
            "reduce `forecast`.")

    real = not np.iscomplexobj(a)
    work = a.astype(np.float64) if real else a.astype(np.complex128)

    u1 = np.linspace(0.0, 1.0, m) if m > 1 else np.zeros(1)
    u2 = np.linspace(0.0, 1.0, t) if t > 1 else np.zeros(1)

    leaves = _forward_pass(work, u1, u2, dt, rcond, eta, smoothness, theta,
                           max_depth, oblique)
    if prune and len(leaves) > 1:
        leaves = _prune(work, leaves, dt, theta, rcond, eta)

    out_cols = t + forecast
    if forecast and t > 1:
        u2_out = np.arange(out_cols) * (1.0 / (t - 1))
    else:
        u2_out = np.linspace(0.0, 1.0, out_cols) if out_cols > 1 else np.zeros(1)

    recon = _reconstruct(leaves, u1, u2_out, dt, out_cols, t, m, real)

    n = work.size
    fit_cols = min(out_cols, t)
    sse = float(np.sum(np.abs(recon[:, :fit_cols] - work[:, :fit_cols]) ** 2))
    total_k = sum(_region_k(ln, theta) for ln in leaves if ln.model is not None)
    bic = _bic(sse, n, total_k)

    regions = []
    for leaf in sorted(leaves, key=lambda ln: (ln.bbox or (0, 0, 0, 0))[2]):
        if leaf.model is None:
            continue
        omega, phi_modes, b = leaf.model
        regions.append(FSRDRegion(
            eigenvalues=omega, modes=phi_modes, amplitudes=b,
            bounding_box=leaf.bbox, level=leaf.level,
            split_vector=leaf.split_vector, rank=omega.size))

    return FSRDResult(reconstruction=recon, regions=regions,
                      n_regions=len(regions), bic=bic)
