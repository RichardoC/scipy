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
import operator
import warnings

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
                             # (peak memory is several times this; see `fsrd`)
_LOG_CLIP = 700.0            # exp(700) < float64 max; overflow-only guard
_SSE_FLOOR_REL = 1e-24       # relative floor on the criterion's error term:
                             # a Frobenius residual of 1e-12 times the data's
                             # own norm, below which nothing is model error
                             # (see `_sse_floor`)


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
    with np.errstate(over='ignore', under='ignore'):
        return 1.0 / (1.0 + np.exp(-tau * z))


def _centroid(phi, u1, u2):
    """Activation-weighted centre of a membership map."""
    w = phi.sum()
    if w <= 0:
        return np.array([u1.mean(), u2.mean()])
    c1 = (phi.sum(axis=1) @ u1) / w
    c2 = (phi.sum(axis=0) @ u2) / w
    return np.array([c1, c2])


def _median_scale(phi, u1, u2):
    """Medians of the coordinates a region actually occupies.

    The split coefficients are expressed relative to these, so that a boundary's
    orientation means the same thing wherever the region sits on the grid.
    """
    keep = phi > 1e-2
    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    m1 = np.median(u1[rows]) if rows.size else 1.0
    m2 = np.median(u2[cols]) if cols.size else 1.0
    return (m1 if abs(m1) > _EPS else 1.0), (m2 if abs(m2) > _EPS else 1.0)


def _steepness(v, dc, mu, scale):
    """Per-split sigmoid steepness ``tau = 20 / (mu ||v|| ||dc||)``.

    The whole coefficient vector enters the norm, the offset included, with the
    two orientation components taken relative to the region's coordinate medians
    (`_median_scale`) -- the normalisation the split search itself works in.
    """
    v_norm = np.array([v[0], v[1] * scale[0], v[2] * scale[1]])
    nv = np.linalg.norm(v_norm)
    ndc = np.linalg.norm(dc)
    denom = mu * nv * ndc
    if denom <= _EPS:
        return 20.0 / max(mu, 1e-6)
    return 20.0 / denom


def _initial_amplitudes(w, lam, s_r, vh_r):
    """Mode amplitudes from the first snapshot: ``b = (W Lambda)^-1 x0_tilde``.

    ``x0_tilde`` is the first column of :math:`\\tilde{\\Sigma}\\tilde{V}^{H}`,
    i.e. the initial state expressed in the reduced basis, so the amplitudes are
    those that reproduce the first snapshot exactly.
    """
    x0 = s_r * vh_r[:, 0]
    wl = w * lam[None, :]                     # W Lambda
    try:
        return np.linalg.solve(wl, x0)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(wl, x0, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None


def _regularized_svd(x1, rcond):
    """Truncated + ridge-regularized + row-pruned SVD of ``x1``.

    Returns ``(U_r, s_r, Vh_r)``.  Small singular values (relative to the
    largest) are truncated, a ridge parameter is chosen by generalized
    cross-validation, and rows of the right singular modes with negligible
    relative mean are pruned.  The GCV grid is scaled by the leading squared
    singular value so the selection is invariant to the data's overall magnitude,
    and includes zero -- which, once the small singular values have been
    truncated, is what it almost always selects, leaving a plain truncated SVD.
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
    # zero is included so that well-conditioned data, for which no shrinkage is
    # warranted, is left alone -- the ridge weights the operator itself, so a
    # floor above zero would damp an exactly recoverable fit.
    best = None
    for delta in np.concatenate(([0.0], np.logspace(-8, 1, 14) * s2[0])):
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
    # 5. return the *ridge-weighted* right factor, so the shrinkage carries into
    #    the operator rather than only into this row selection.  Here
    #    ``s_map = diag(s**2 / (s**2 + delta)) Vh``, so the caller's
    #    ``s_map.conj().T / s`` is ``V diag(s / (s**2 + delta))`` -- the ridge
    #    pseudo-inverse of Sigma.  With no ridge that is ``Vh`` exactly, so use
    #    the factor the decomposition already gives rather than rebuilding it.
    vh_r = vh[rows] if delta == 0.0 else s_map[rows]
    return u[:, rows], s[rows], vh_r


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
    b = _initial_amplitudes(w, lam, s_r, vh_r)
    if b is None:
        return None
    return omega, phi, b


def _dmd_reconstruct(model, n, dt):
    """Evaluate a fitted DMD model over ``n`` successive columns.

    Computes each mode's contribution ``b_i lambda_i^k`` directly in log space
    with an overflow-only guard.  In-region reconstruction is therefore exact
    (the amplitudes are fixed to reproduce the first snapshot); forecasting
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
                 'split_vector', 'terms', 'nrmse')

    def __init__(self, phi, level, path):
        self.phi = phi
        self.level = level
        self.path = path
        self.model = None
        self.bbox = None
        self.oblique = False
        self.spans = None
        self.split_vector = path[-1][0] if path else None
        self.terms = None                     # cached `_leaf_blend_terms`
        self.nrmse = None                     # cached `_region_nrmse`


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


def _leaf_blend_terms(leaf, u1, u2_out, dt, out_cols, orig_cols):
    """One leaf's membership-weighted contribution to the assembled model.

    Returns ``(m0, m1, t0, col_end, phi * f_i, phi)``: the block of the output
    grid the leaf occupies, its weighted local reconstruction, and the weights
    themselves (needed to renormalise the blend).

    The terms depend on the leaf alone -- its path, its fitted model and its
    bounding box, none of which change once `_fit_node` has run -- so for the
    input window they are cached on the node and reused across the many
    candidate models the criterion scores.  A forecast horizon extends some
    regions and not others, so those terms are not cached.
    """
    in_window = out_cols == orig_cols
    if in_window and leaf.terms is not None:
        return leaf.terms
    m0, m1, t0, t1 = leaf.bbox
    if leaf.oblique:
        # An oblique region is not extrapolated along its diagonal: its
        # model is only evaluated over its own in-window span, since the
        # inverse transform has to map the result back onto that span.
        col_end = min(t1, orig_cols)
    else:
        col_end = out_cols if t1 >= orig_cols else t1
    qn = col_end - t0
    # membership only over the region's bounding box (rows m0:m1, cols t0:)
    u1r = u1[m0:m1]
    u2r = u2_out[t0:col_end]
    phi = np.ones((m1 - m0, qn))
    with np.errstate(under='ignore'):
        for v, tau, side, _sid in leaf.path:
            om = _sigmoid(u1r, u2r, v, tau)
            phi = phi * (om if side == 0 else (1.0 - om))
    dense = _dmd_reconstruct(leaf.model, qn, dt)
    if leaf.oblique:
        dense = _inverse_topological_transform(dense, leaf.spans, m1 - m0, qn)
    terms = (m0, m1, t0, col_end, phi * dense, phi)
    if in_window:
        leaf.terms = terms
    return terms


def _reconstruct(leaves, u1, u2_out, dt, out_cols, orig_cols, shape_rows, real,
                 warn=True):
    """Blend the leaves' local reconstructions over the output grid.

    Each region is reconstructed over exactly its own bounding box; a region
    whose box reaches the last input column is extended through the forecast
    horizon.  Cells are renormalised so the fuzzy memberships form a partition
    of unity even after the ``eta`` support cutoff.

    This is the single assembly path: the model-selection criterion scores a
    candidate set of leaves by blending it here too (see `_model_sse`), so the
    quantity that is minimised and the array that is returned cannot drift
    apart.  ``warn=False`` suppresses the unsupported-forecast report for that
    internal use, which never asks for a forecast in the first place.
    """
    accum = np.zeros((shape_rows, out_cols), dtype=complex)
    weight = np.zeros((shape_rows, out_cols))
    for leaf in leaves:
        if leaf.model is None:
            continue
        m0, m1, t0, col_end, contrib, phi = _leaf_blend_terms(
            leaf, u1, u2_out, dt, out_cols, orig_cols)
        accum[m0:m1, t0:col_end] += contrib
        weight[m0:m1, t0:col_end] += phi
    unsupported = warn and out_cols > orig_cols and bool(
        np.any(weight[:, orig_cols:] <= _EPS))
    if unsupported:
        # Only regions that can be extended past the data contribute to the
        # forecast; where none does, the cells stay at zero, which must not be
        # mistaken for a prediction.
        warnings.warn("no region could be extrapolated over part of the "
                      "requested forecast horizon; those entries of the "
                      "reconstruction are zero rather than predicted",
                      RuntimeWarning, stacklevel=3)
    weight = np.where(weight > _EPS, weight, 1.0)
    accum = accum / weight
    return accum.real if real else accum


def _bic(sse, n, k):
    mse = sse / n + np.finfo(float).tiny
    return n * np.log(mse) + k * np.log(n)


def _leaf_nrmse(a, node, dt):
    """`_region_nrmse` for a node, cached (the model and data never change)."""
    if node.nrmse is None:
        node.nrmse = _region_nrmse(a, node, dt)
    return node.nrmse


def _region_nrmse(a, node, dt):
    """Normalised RMSE of a region's local model, over the region's own extent.

    The residual is normalised by the region's own spread about its mean, so
    regions of different magnitude contribute comparably.
    """
    m0, m1, t0, t1 = node.bbox
    block = a[m0:m1, t0:t1]
    recon = _node_local_recon(node, dt, a.shape)[m0:m1, t0:t1]
    scale = np.linalg.norm(block - block.mean())
    err = np.linalg.norm(recon - block)
    return float(err / scale) if scale > _EPS else float(err)


def _model_wnrmse(a, nodes, dt):
    """Weighted net error of a whole candidate model, following [1]_.

    Each region's normalised error is weighted by the share of the grid its
    membership accounts for; the weights of the current leaves sum to one.  This
    is the cost the split search minimises -- as distinct from the information
    criterion, which decides how many regions to keep.
    """
    total = 0.0
    denom = float(a.size)
    for nd in nodes:
        if nd.model is None or nd.bbox is None:
            continue
        total += float(nd.phi.sum()) / denom * _leaf_nrmse(a, nd, dt)
    return total


def _sse_floor(a):
    """Smallest squared error attributable to the model rather than round-off.

    `_bic` is a log-likelihood in the SSE and so is unbounded below as the SSE
    goes to zero.  On data that a single operator already reproduces to machine
    precision the residual is not model error at all: it is round-off in the
    exponential reconstruction, ``b_i lambda_i^k``, whose relative size grows
    with the number of steps ``k`` a region spans.  Splitting a region in time
    halves that span and so shrinks the residual, and without a floor every
    such split would read as an improvement of the fit -- the criterion would
    segment an exactly linear system indefinitely, chasing round-off.

    The floor is a fixed relative one, ``_SSE_FLOOR_REL`` times the data's own
    energy, so it scales with the data and leaves any genuine residual (which
    is many orders of magnitude larger) untouched.  Two models that both sit
    under it are separated by their complexity alone, which is the intended
    outcome: neither explains the data any better than the other.
    """
    return _SSE_FLOOR_REL * float(np.sum(np.abs(a) ** 2))


def _model_sse(a, nodes, u1, u2, dt):
    """``|| X - X_global ||_F^2`` of the model assembled from ``nodes``.

    The criterion of [1]_ defines its error term on the *assembled* global
    model, explicitly not as anything derived from the local per-region errors
    the split search uses.  So the candidate leaves are blended exactly as the
    returned reconstruction is -- by `_reconstruct` itself, so the two cannot
    drift apart -- and the result compared against the data.

    The blend covers the input window only: that is what the criterion compares
    against, and it also keeps this path structurally clear of `_reconstruct`'s
    unsupported-forecast warning, which only a forecast horizon can trigger.
    """
    rows, cols = a.shape
    recon = _reconstruct(nodes, u1, u2, dt, cols, cols, rows,
                         not np.iscomplexobj(a), warn=False)
    return max(float(np.sum(np.abs(recon - a) ** 2)), _sse_floor(a))


def _model_bic(a, nodes, n, u1, u2, dt, theta):
    """Information criterion of a whole candidate model (all of its leaves).

    Split and prune decisions compare this quantity, following the criterion of
    [1]_, which is defined over the *complete* model rather than one region:
    the error term is the squared error of the fully assembled reconstruction
    (`_model_sse`), and the complexity term the summed per-region complexity.
    Judging a split by the split node's error alone would instead reward
    shrinking a residual that is already negligible in the full model, which
    drives over-segmentation.
    """
    sse = _model_sse(a, nodes, u1, u2, dt)
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
    tau = _steepness(v, c2 - c1, mu, _median_scale(node.phi, u1, u2))
    omega = _sigmoid(u1, u2, v, tau)
    # a saturated membership times a saturated activation underflows to zero,
    # which is the intended value
    with np.errstate(under='ignore'):
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
    position that minimises the model's weighted net error -- before the level as
    a whole is assessed, and growth continues while a completed level lowers the
    model's information criterion.  The pass therefore stops one level *after* the best
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
    best_bic = _model_bic(a, leaves, n, u1, u2, dt, theta)
    sid = 0
    for _ in range(max_depth):
        new_leaves = []
        changed = False
        for idx, leaf in enumerate(leaves):
            pending = len(leaves) - idx - 1
            if (leaf.model is None or leaf.bbox is None
                    or len(new_leaves) + 2 + pending > _MAX_NODES):
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
                # the boundary is positioned by the weighted net error of the
                # resulting model, not by the information criterion
                cost = _model_wnrmse(a, others + [child_a, child_b], dt)
                if best is None or cost < best[0]:
                    best = (cost, child_a, child_b)
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
        level_bic = _model_bic(a, leaves, n, u1, u2, dt, theta)
        if level_bic >= best_bic or len(leaves) >= _MAX_NODES:
            break                     # this level overshoots; hand it to `_prune`
        best_bic = level_bic
    return leaves


def _prune(a, leaves, u1, u2, dt, theta, rcond, eta):
    """Backward elimination: collapse sibling pairs back into their parent while
    that strictly lowers the whole model's BIC.

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
            if (_model_bic(a, others + [parent], n, u1, u2, dt, theta)
                    < _model_bic(a, others + sibs, n, u1, u2, dt, theta)):
                leaves = others + [parent]
                changed = True
    return leaves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _as_float(value, name):
    """Coerce to a finite float, reporting the parameter by name."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a real number.") from exc
    if not np.isfinite(out):
        raise ValueError(f"`{name}` must be finite.")
    return out


def _as_index(value, name):
    """Coerce to an exact integer, reporting the parameter by name.

    Truncating silently would let ``max_depth=2.9`` mean 2.
    """
    try:
        return operator.index(value)
    except TypeError:
        pass
    out = _as_float(value, name)
    if out != int(out):
        raise ValueError(f"`{name}` must be an integer.")
    return int(out)



@xp_capabilities(np_only=True)
def fsrd(a, dt=1.0, *, max_depth=3, theta=3.0, rcond=1e-4, eta=1e-4,
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
        its tree level ``L``.  Larger values favour fewer regions.  Default 3,
        the value [1]_ used for its own reported results.
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
        0.  ``M * (T + forecast)`` must not exceed 5e7.  Note that this bounds
        the number of output cells, not bytes: the blend holds a complex
        accumulator and a real weight per cell and builds one grid per region, so
        peak memory runs to roughly eight times the size of the output array.
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

    Warns
    -----
    RuntimeWarning
        If part of a requested forecast horizon is covered only by regions that
        cannot be extrapolated, in which case those entries of `reconstruction`
        are zero rather than predicted.

    See Also
    --------
    numpy.linalg.svd : Singular value decomposition used by each local fit.
    numpy.linalg.eig : Eigendecomposition of each reduced operator.

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
      region, rather than refined by particle-swarm optimization.  Candidates are
      ranked by the paper's weighted net error, as there, so only the search over
      positions is coarser.
    - The ridge parameter is chosen by generalized cross-validation rather than
      the paper's BIC fixed-point iteration.  It shrinks the local operator as in
      [1]_, but this selection rule effectively never chooses a nonzero ridge
      once the small singular values have been truncated, so in practice each
      region reduces to a plain truncated-SVD DMD fit.
    - The explicit error model that [1]_ applies to whiten the data before
      fitting is not implemented, so noisy data is fitted as given.
    - The input is used as supplied, with the singular-value cutoff applied
      relative to the largest singular value.  [1]_ instead rescales every data
      set to ``[0, 1]`` and applies `rcond` as an absolute cutoff; rescale the
      input yourself to follow that convention.
    - A row split requires a region at least 4 rows deep, a column split one at
      least 8 columns wide, and an oblique split both; [1]_ imposes no such
      minimum, so very small regions are left intact here.

    Model order is selected as in [1]_, in two passes over the whole model's
    criterion.  The forward pass splits every splittable region of a level
    before assessing the level, and continues while a completed level lowers the
    criterion; when it is the level comparison that stops the growth, the tree is
    left deliberately over-grown by one level.  The backward pass then runs the
    tree in reverse, reviewing each pair of children deepest-level first and
    collapsing it back into its parent whenever that strictly lowers the
    criterion, until a full sweep changes nothing.  [1]_ notes that the resulting
    tree is not uniquely determined by this procedure.  Growth also stops at
    ``max_depth`` or at an internal cap on the number of regions, in which case
    the tree is not over-grown.

    The reconstruction and forecast use the fitted eigenvalues as returned in
    ``regions[i].eigenvalues`` (no clipping); a region's in-window
    reconstruction is bounded because its amplitudes are fixed by the region's
    first snapshot, while forecasting a mode with ``|lambda| > 1`` grows as
    expected.  Forecasting is intended for axis-aligned regions; an oblique
    region is not extrapolated along its diagonal beyond the input window, so a
    multi-region forecast that would rely on an oblique region past the data is
    bounded and finite but not accuracy-guaranteed.

    The two costs of [1]_ are kept distinct.  Where a boundary is placed is
    decided by the weighted net error of the resulting model -- the local cost --
    while how many regions to keep is decided by the information criterion of the
    *complete* model, in the level test and in pruning.  Neither is ever taken
    over one region in isolation.  The criterion's error term is the squared
    error of the fully assembled reconstruction,
    :math:`\| X - \tilde{X}_{global} \|_F^2` -- a candidate set of regions is
    blended exactly as the returned ``reconstruction`` is and then compared with
    the data -- rather than anything summed from the regions' own local errors.
    The reported ``bic`` is therefore precisely the quantity both passes
    minimise, so it does order models as they do; because pruning only ever
    collapses a pair of regions when that strictly lowers it, the selected model
    never carries a higher ``bic`` than the over-grown tree ``prune=False``
    returns.  An additive constant is dropped from it, so it remains
    incomparable to a BIC from another tool.

    Two numerical details of that criterion are worth knowing.  Because it is a
    log-likelihood in the squared error it is unbounded below as that error goes
    to zero, and on data a single operator already reproduces to machine
    precision the residual is not model error but round-off in the exponential
    reconstruction -- round-off that shrinks with the number of steps a region
    spans, so that splitting in time would appear to improve the fit
    indefinitely.  The error term is therefore floored at ``1e-24`` times the
    data's own squared norm (a relative Frobenius residual of ``1e-12``), below
    which two models are separated by their complexity alone.  And because the
    assembled error is not additive over regions, a level of the forward pass
    can score marginally worse than the level above it even though a deeper one
    would score much better; growth stops at the first such level, so on some
    inputs -- particularly at small `theta`, where many regions are affordable
    -- a deeper tree that the criterion would prefer is not reached.  Raising
    `max_depth` does not recover it; lowering `smoothness` or raising `theta`
    changes which levels are compared.

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
    if a.dtype.kind not in 'biufc':
        raise ValueError("`a` must have a numeric dtype.")
    if check_finite and not np.isfinite(a).all():
        raise ValueError("array must not contain infs or NaNs")

    max_depth = _as_index(max_depth, 'max_depth')
    forecast = _as_index(forecast, 'forecast')
    if not 0 <= max_depth <= _MAX_DEPTH:
        raise ValueError(f"`max_depth` must be in [0, {_MAX_DEPTH}].")
    if forecast < 0:
        raise ValueError("`forecast` must be non-negative.")
    # each of these must be finite as well as in range: a one-sided comparison
    # would let infinity through, and infinity propagates into every fit
    dt = _as_float(dt, 'dt')
    rcond = _as_float(rcond, 'rcond')
    eta = _as_float(eta, 'eta')
    smoothness = _as_float(smoothness, 'smoothness')
    theta = _as_float(theta, 'theta')
    if not dt > 0:
        raise ValueError("`dt` must be positive.")
    if not 0 < rcond < 1:
        raise ValueError("`rcond` must lie in (0, 1).")
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

    out_cols = t + forecast
    if forecast and t > 1:
        u2_out = np.arange(out_cols) * (1.0 / (t - 1))
    else:
        u2_out = np.linspace(0.0, 1.0, out_cols) if out_cols > 1 else np.zeros(1)

    # Fuzzy memberships saturate and their products underflow, and data of
    # extreme magnitude overflows intermediate sums.  Neither is actionable by
    # the caller, and reporting them would bury the one thing that is -- a
    # reconstruction that did not come out finite, which is checked below.  So
    # the floating-point environment is quietened over the numerics only, and
    # never over the caller's own code.
    with np.errstate(all='ignore'):
        leaves = _forward_pass(work, u1, u2, dt, rcond, eta, smoothness, theta,
                               max_depth, oblique)
        if prune and len(leaves) > 1:
            leaves = _prune(work, leaves, u1, u2, dt, theta, rcond, eta)
        recon = _reconstruct(leaves, u1, u2_out, dt, out_cols, t, m, real)

    if not np.isfinite(recon).all():
        warnings.warn("the reconstruction is not finite everywhere; the fitted "
                      "dynamics may be too large to represent, or the input too "
                      "poorly scaled", RuntimeWarning, stacklevel=2)

    fitted = [ln for ln in leaves if ln.model is not None]
    if not fitted:
        # Nothing could be fitted, so the reconstruction is identically zero.
        # That is not a model of the data, and reporting a criterion for it would
        # be worse than useless: with no parameters to penalise it scores better
        # than any genuine fit.
        if work.any():
            warnings.warn("no local operator could be fitted, so the "
                          "reconstruction is zero everywhere and no region is "
                          "reported; the data may be rank deficient, or `rcond` "
                          "too large", RuntimeWarning, stacklevel=2)
        bic = np.nan
    else:
        n = work.size
        with np.errstate(all='ignore'):
            # the same quantity the two passes minimised, recomputed from the
            # reconstruction already in hand rather than by blending again: a
            # forecast horizon does not touch the in-window columns
            sse = max(float(np.sum(np.abs(recon[:, :t] - work) ** 2)),
                      _sse_floor(work))
            bic = _bic(sse, n, sum(_region_k(ln, theta) for ln in fitted))
        if not np.isfinite(bic):
            bic = np.nan

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
