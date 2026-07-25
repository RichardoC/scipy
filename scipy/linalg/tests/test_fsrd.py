import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scipy.linalg import fsrd
from scipy.linalg._fsrd import (
    FSRDResult, FSRDRegion,
    _Node, _try_split, _prune, _fit_node, _bic, _region_k, _sigmoid,
    _topological_transform, _row_spans, _region_sse, _node_local_recon,
    _forward_pass,
)
from scipy._lib._array_api import make_xp_test_case, xp_assert_close


def _orbit(A, x0, n):
    x = np.empty((A.shape[0], n), dtype=A.dtype)
    x[:, 0] = x0
    for k in range(n - 1):
        x[:, k + 1] = A @ x[:, k]
    return x


def _rot(theta, scale=1.0):
    c, s = np.cos(theta), np.sin(theta)
    return scale * np.array([[c, -s], [s, c]])


def _two_regime(n_left=40, n_right=40):
    # two temporal regimes with different rotation frequencies, concatenated
    left = _orbit(_rot(0.15, 0.99), [1.0, 0.0], n_left)
    right = _orbit(_rot(0.9, 0.99), left[:, -1], n_right)
    return np.concatenate([left, right[:, 1:]], axis=1)


@make_xp_test_case(fsrd)
class TestFSRD:
    def test_single_linear_system(self, xp):
        # One linear operator -> one region, (near) exact reconstruction.
        x = _orbit(_rot(0.3), [1.0, 0.0], 40)
        res = fsrd(xp.asarray(x), max_depth=0)
        assert isinstance(res, FSRDResult)
        assert res.n_regions == 1
        assert isinstance(res.regions[0], FSRDRegion)
        xp_assert_close(res.reconstruction, xp.asarray(x), atol=1e-6)
        # imaginary part of omega = ln(lambda)/dt recovers the rotation freq
        assert_allclose(np.abs(res.regions[0].eigenvalues.imag).max(), 0.3,
                        atol=1e-6)

    def test_no_oversegmentation_of_one_operator(self, xp):
        # One operator must not be split up as the depth budget grows, even on a
        # large grid.  The criterion is evaluated over the whole model, so a
        # split that does not improve the global fit is rejected however many
        # cells the (here negligible) residual is spread over.
        x = np.repeat(_orbit(_rot(0.3, 0.99), [1.0, 0.0], 120), 30, axis=0)
        counts = [fsrd(xp.asarray(x), max_depth=d).n_regions
                  for d in (0, 2, 4)]
        assert counts == [1, 1, 1]

    @pytest.mark.parametrize('oblique', [False, True])
    def test_forecast_leaves_in_window_untouched(self, xp, oblique):
        # Asking for a forecast must not change the in-window reconstruction, on
        # any split geometry.  An oblique region has to be mapped back onto its
        # own span, so it is not extrapolated along its diagonal; that must not
        # leak the extrapolated columns back inside the window.
        rows, cols = 30, 60
        g1, g2 = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols),
                             indexing='ij')
        x = np.sin(8 * g2) + 0.1 * g1
        corner = g1 + g2 > 1
        x[corner] = 3 * np.cos(12 * g2[corner])

        ref = fsrd(xp.asarray(x), oblique=oblique).reconstruction
        for forecast in (1, 7):
            with warnings.catch_warnings():
                # an oblique region cannot be extrapolated, which this data
                # provokes; that is reported separately and is not what this
                # test is about
                warnings.simplefilter('ignore', RuntimeWarning)
                res = fsrd(xp.asarray(x), oblique=oblique, forecast=forecast)
            assert res.reconstruction.shape == (rows, cols + forecast)
            xp_assert_close(res.reconstruction[:, :cols], ref, atol=1e-10)

    def test_unsupported_forecast_warns(self, xp):
        # Oblique regions are not extrapolated, so a forecast horizon they alone
        # cover cannot be predicted.  Those entries stay zero, which must be
        # announced rather than passed off as a prediction.
        rows, cols = 30, 60
        g1, g2 = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols),
                             indexing='ij')
        x = np.sin(8 * g2) + 0.1 * g1
        corner = g1 + g2 > 1
        x[corner] = 3 * np.cos(12 * g2[corner])
        with pytest.warns(RuntimeWarning, match='forecast horizon'):
            fsrd(xp.asarray(x), forecast=10)

        # a region that can be extended must forecast silently
        t = np.linspace(0, 6, 80)
        y = np.stack([np.sin(2 * t), np.cos(2 * t)])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            res = fsrd(xp.asarray(y), dt=float(t[1] - t[0]), max_depth=0,
                       forecast=20)
        assert res.reconstruction.shape == (2, 100)

    @pytest.mark.parametrize('shape', [(4, 10), (2, 2)])
    def test_degenerate_all_zero_input(self, xp, shape):
        # Nothing can be fitted, so no region is returned -- but the call must
        # still succeed and give a finite, correctly shaped reconstruction.
        # no region exists, so the whole forecast horizon is unsupported and
        # says so
        with pytest.warns(RuntimeWarning, match='forecast horizon'):
            res = fsrd(xp.zeros(shape), forecast=3)
        assert res.n_regions == 0
        assert res.reconstruction.shape == (shape[0], shape[1] + 3)
        assert bool(np.all(np.isfinite(np.asarray(res.reconstruction))))

    def test_prune_selects_the_model_order(self, xp):
        # `prune` performs the model selection: without it the over-grown tree
        # the forward pass produced is returned, so it can only have at least as
        # many regions, and on a single operator pruning recovers exactly one.
        for x in (_orbit(_rot(0.3), [1.0, 0.0], 80), _two_regime()):
            selected = fsrd(xp.asarray(x), max_depth=3)
            grown = fsrd(xp.asarray(x), max_depth=3, prune=False)
            assert grown.n_regions >= selected.n_regions
        assert fsrd(xp.asarray(_orbit(_rot(0.3), [1.0, 0.0], 80)),
                    max_depth=3).n_regions == 1

    @pytest.mark.parametrize('data', ['two_regime', 'noisy'])
    def test_extra_regions_must_pay_for_themselves(self, xp, data):
        # Region count must converge with depth, and any region the deeper model
        # adds has to buy accuracy -- the error must never stay flat while the
        # count grows (the signature of splitting for its own sake).
        if data == 'two_regime':
            x = _two_regime()
        else:
            rng = np.random.default_rng(1234)
            x = (np.repeat(_orbit(_rot(0.3, 0.99), [1.0, 0.0], 90), 12, axis=0)
                 + 1e-3 * rng.standard_normal((24, 90)))

        def run(depth):
            res = fsrd(xp.asarray(x), max_depth=depth)
            err = float(np.linalg.norm(np.asarray(res.reconstruction) - x))
            return res.n_regions, err

        n_mid, err_mid = run(3)
        n_deep, err_deep = run(6)
        assert n_deep == n_mid, 'region count must converge with depth'
        if n_deep > 1:
            n_one, err_one = run(0)
            assert err_deep < err_one, 'extra regions must reduce the error'

    def test_eigenvalues_match_exact_dmd(self, xp):
        # On a pure rotation the DMD eigenvalues must be exp(+/- i theta).
        x = _orbit(_rot(0.5), [1.0, 0.2], 30)
        res = fsrd(xp.asarray(x), max_depth=0)
        lam = np.exp(res.regions[0].eigenvalues)      # discrete eigenvalues
        assert_allclose(np.sort(np.angle(lam)), [-0.5, 0.5], atol=1e-6)
        assert_allclose(np.abs(lam), [1.0, 1.0], atol=1e-6)

    def test_real_input_real_output(self, xp):
        x = _orbit(_rot(0.4, 0.98), [1.0, 0.5], 40)
        res = fsrd(xp.asarray(x), max_depth=0)
        assert not np.iscomplexobj(res.reconstruction)
        xp_assert_close(res.reconstruction, xp.asarray(x), atol=1e-6)

    def test_complex_input_complex_output(self, xp):
        A = _rot(0.35) + 0j
        x = _orbit(A, np.array([1.0 + 0.2j, -0.3 + 0.1j]), 40)
        res = fsrd(xp.asarray(x), max_depth=0)
        assert np.iscomplexobj(res.reconstruction)
        xp_assert_close(res.reconstruction, xp.asarray(x), atol=1e-6)

    def test_float32_input(self, xp):
        x = _orbit(_rot(0.3), [1.0, 0.0], 40).astype(np.float32)
        res = fsrd(xp.asarray(x), max_depth=0)
        assert np.all(np.isfinite(res.reconstruction))
        resid = (np.linalg.norm(res.reconstruction - x)
                 / np.linalg.norm(x))
        assert resid < 1e-4

    def test_growing_system_reconstructed_accurately(self, xp):
        # A genuinely growing system (|lambda| > 1) must be reconstructed
        # accurately, not silently flattened; the eigenvalue magnitude is 1.2.
        x = _orbit(_rot(0.2, 1.2), [1.0, 0.0], 40)
        res = fsrd(xp.asarray(x), max_depth=0)
        xp_assert_close(res.reconstruction, xp.asarray(x), rtol=1e-6, atol=1e-9)
        assert_allclose(np.exp(res.regions[0].eigenvalues.real).max(), 1.2,
                        atol=1e-6)

    def test_reconstruction_stays_finite(self, xp):
        rng = np.random.default_rng(3)
        g = rng.standard_normal((10, 80))
        for oblique in (False, True):
            res = fsrd(xp.asarray(g), max_depth=3, oblique=oblique)
            assert np.all(np.isfinite(res.reconstruction))

    def test_piecewise_dynamics_multi_region(self, xp):
        x = _two_regime()
        glob = fsrd(xp.asarray(x), max_depth=0)
        multi = fsrd(xp.asarray(x), max_depth=4)

        def resid(r):
            return np.linalg.norm(r.reconstruction - x) / np.linalg.norm(x)

        assert multi.n_regions > glob.n_regions
        assert resid(multi) < 0.5 * resid(glob)

    def test_stacked_systems_with_row_splits(self, xp):
        a = _orbit(_rot(0.2, 0.99), [1.0, 0.3], 60)
        b = _orbit(_rot(1.1, 0.99), [0.5, -0.2], 60)
        x = np.vstack([a, b])          # 4 x 60
        for oblique in (False, True):
            res = fsrd(xp.asarray(x), max_depth=3, oblique=oblique)
            assert np.all(np.isfinite(res.reconstruction))
            resid = np.linalg.norm(res.reconstruction - x) / np.linalg.norm(x)
            assert resid < 0.1

    def test_forecast_shape_and_extrapolation(self, xp):
        x = _orbit(_rot(0.25), [1.0, 0.0], 60)
        res = fsrd(xp.asarray(x), max_depth=0, forecast=10)
        assert res.reconstruction.shape == (2, 70)
        future = _orbit(_rot(0.25), x[:, 0], 70)
        xp_assert_close(res.reconstruction[:, 60:], xp.asarray(future[:, 60:]),
                        atol=1e-4)

    def test_forecast_multi_region_shape(self, xp):
        x = _two_regime()
        res = fsrd(xp.asarray(x), max_depth=4, forecast=15)
        assert res.reconstruction.shape[1] == x.shape[1] + 15
        assert np.all(np.isfinite(res.reconstruction))

    def test_theta_controls_complexity(self, xp):
        # A genuine multi-regime signal: larger theta -> fewer regions.
        x = _two_regime()
        few = fsrd(xp.asarray(x), max_depth=4, theta=5.0)
        many = fsrd(xp.asarray(x), max_depth=4, theta=1.0)
        assert few.n_regions <= many.n_regions
        assert many.n_regions >= 2

    def test_scale_invariance(self, xp):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((6, 60))
        results = [fsrd(xp.asarray(s * base), max_depth=2)
                   for s in (1e-6, 1.0, 1e6)]
        n = [r.n_regions for r in results]
        assert n[0] == n[1] == n[2]

    def test_deterministic(self, xp):
        x = _two_regime()
        r1 = fsrd(xp.asarray(x), max_depth=4)
        r2 = fsrd(xp.asarray(x), max_depth=4)
        xp_assert_close(r1.reconstruction, r2.reconstruction, rtol=0, atol=0)

    def test_check_finite(self, xp):
        x = _orbit(_rot(0.3), [1.0, 0.0], 20)
        for bad in (np.nan, np.inf):
            y = x.copy()
            y[0, 0] = bad
            with pytest.raises(ValueError, match="infs or NaNs"):
                fsrd(xp.asarray(y))

    def test_invalid_arguments(self, xp):
        x = xp.asarray(np.ones((3, 10)))
        with pytest.raises(ValueError):
            fsrd(xp.asarray(np.ones((2, 3, 4))))       # not 2-D
        with pytest.raises(ValueError):
            fsrd(xp.asarray(np.ones(5)))               # 1-D
        with pytest.raises(ValueError):
            fsrd(xp.asarray(np.ones((0, 5))))          # empty
        with pytest.raises(ValueError):
            fsrd(xp.asarray(np.ones((3, 1))))          # T < 2
        for kw in (dict(max_depth=-1), dict(max_depth=999), dict(forecast=-1),
                   dict(dt=0.0), dict(rcond=0.0), dict(eta=1.5),
                   dict(smoothness=0.0), dict(theta=0.5), dict(forecast=10**9)):
            with pytest.raises(ValueError):
                fsrd(x, **kw)


class TestFSRDInternals:
    # NumPy-only tests of the tree machinery.
    def test_partition_of_unity(self):
        # The two children of a split partition their parent's membership.
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 40))
        m, t = a.shape
        u1 = np.linspace(0, 1, m)
        u2 = np.linspace(0, 1, t)
        root = _Node(np.ones((m, t)), 0, [])
        pair = _try_split(a, root, np.array([-0.5, 0.0, 1.0]), 0,
                          u1, u2, 1.0, 1e-4, 1e-4, 0.05)
        assert pair is not None
        left, right = pair
        assert_allclose(left.phi + right.phi, np.ones((m, t)), atol=1e-12)

    def test_prune_collapses_unnecessary_split(self):
        # A single-operator system split in two should collapse back to one.
        x = _orbit(_rot(0.3), [1.0, 0.0], 60)
        m, t = x.shape
        u1 = np.linspace(0, 1, m)
        u2 = np.linspace(0, 1, t)
        root = _Node(np.ones((m, t)), 0, [])
        left, right = _try_split(x, root, np.array([-0.5, 0.0, 1.0]), 0,
                                 u1, u2, 1.0, 1e-4, 1e-4, 0.05)
        pruned = _prune(x, [left, right], 1.0, 1.5, 1e-4, 1e-4)
        assert len(pruned) == 1

    def test_forward_pass_overgrows_and_prune_selects(self):
        # The forward pass must overshoot (it splits a level through before
        # assessing it) and the backward pass must do the selection: on a single
        # linear operator the grown tree has more than one leaf and pruning
        # folds it back to exactly one.
        x = _orbit(_rot(0.3), [1.0, 0.0], 80)
        m, t = x.shape
        u1 = np.linspace(0, 1, m)
        u2 = np.linspace(0, 1, t)
        grown = _forward_pass(x, u1, u2, 1.0, 1e-4, 1e-4, 0.05, 1.5, 3, True)
        pruned = _prune(x, grown, 1.0, 1.5, 1e-4, 1e-4)
        assert len(pruned) < len(grown), 'pruning should select a smaller tree'
        assert len(pruned) == 1

    def test_region_sse_weights_membership_linearly(self):
        # The membership must weight the *squared* error, so that halving it
        # halves the reported error.  Weighting the residual instead applies
        # phi**2 (halving would quarter it), and since w**2 + (1-w)**2 <= 1 a
        # fuzzy split would then lower the criterion without fitting anything
        # better -- the mechanism behind over-segmentation.
        x = _two_regime()
        m, t = x.shape
        full = _Node(np.ones((m, t)), 0, [])
        _fit_node(x, full, 1.0, 1e-4, 1e-4)
        half = _Node(0.5 * np.ones((m, t)), 0, [])
        _fit_node(x, half, 1.0, 1e-4, 1e-4)
        # uniform membership => identical bounding box, hence an identical fit
        assert half.bbox == full.bbox
        assert_allclose(_region_sse(x, half, 1.0),
                        0.5 * _region_sse(x, full, 1.0), rtol=1e-10)

    def test_split_of_unchanged_fit_does_not_lower_sse(self):
        # Partition of unity (phi_left + phi_right == phi_parent) plus linear
        # weighting means a split cannot reduce the criterion's error term
        # unless the local fits actually improve.
        x = _two_regime()
        m, t = x.shape
        u1 = np.linspace(0, 1, m)
        u2 = np.linspace(0, 1, t)
        root = _Node(np.ones((m, t)), 0, [])
        _fit_node(x, root, 1.0, 1e-4, 1e-4)
        left, right = _try_split(x, root, np.array([-0.5, 0.0, 1.0]), 0,
                                 u1, u2, 1.0, 1e-4, 1e-4, 0.05)
        # weight the *parent's* own residual by each child's membership: the two
        # halves must add back up to the parent's error, to within the effect of
        # each child's eta-truncated bounding box.
        recon = _node_local_recon(root, 1.0, x.shape)
        sq = np.abs(recon - x) ** 2
        halves = float(np.sum(left.phi * sq) + np.sum(right.phi * sq))
        assert_allclose(halves, _region_sse(x, root, 1.0), rtol=1e-10)

    # ------------------------------------------------------------------
    # Supplementary-material fidelity checks (audit regression guards).
    # ------------------------------------------------------------------
    def test_sigmoid_matches_sm_s14(self):
        # SM eq (S14): Omega = 1 / (1 + exp(-tau (v0 + v1 u1 + v2 u2))).
        u1 = np.linspace(0, 1, 5)
        u2 = np.linspace(0, 1, 7)
        v = np.array([-0.3, 1.0, -0.5])
        tau = 3.7
        z = v[0] + v[1] * u1[:, None] + v[2] * u2[None, :]
        expected = 1.0 / (1.0 + np.exp(-tau * z))
        got = _sigmoid(u1, u2, v, tau)
        assert_allclose(got, expected, atol=1e-14)
        # SM eq (S19): a split and its complement form a partition of unity.
        assert_allclose(got + (1.0 - got), np.ones_like(got), atol=1e-14)

    def test_bic_matches_sm_s37_s39(self):
        # SM eqs (S37)-(S39) with the documented dropped constant:
        # BIC = N log(SSE/N) + k log(N).
        sse, n, k = 12.34, 200, 5.0
        expected = n * np.log(sse / n + np.finfo(float).tiny) + k * np.log(n)
        assert_allclose(_bic(sse, n, k), expected, rtol=1e-12)

    def test_k_scales_with_level_sm_s45(self):
        # SM eq (S45): k = sum_i r_i Theta^{L_i - 1}; with the root at code
        # level 0 this is r_i * Theta^level.
        x = np.random.default_rng(1).standard_normal((8, 60))
        theta = 2.0
        root = _Node(np.ones(x.shape), 0, [])
        _fit_node(x, root, 1.0, 1e-4, 1e-4)
        assert_allclose(_region_k(root, theta),
                        root.model[0].size * theta ** 0)
        deep = _Node(np.ones(x.shape), 2, [])
        _fit_node(x, deep, 1.0, 1e-4, 1e-4)
        assert_allclose(_region_k(deep, theta),
                        deep.model[0].size * theta ** 2)

    def test_topological_transform_sm11_branches(self):
        # SM-11 forward transform: single-element rows become constant,
        # full rows pass through, partial runs are interpolated to width q.
        block = np.arange(1, 16, dtype=float).reshape(3, 5)
        mask = np.array([[0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0]], dtype=bool)
        spans = _row_spans(mask)
        assert spans == [(2, 2), (0, 4), (1, 3)]
        q = 5
        y = _topological_transform(block, spans, q)
        # step 5: single in-region element -> whole row equals that value
        assert_allclose(y[0], block[0, 2])
        # step 6: full-width run -> identity
        assert_allclose(y[1], block[1])
        # step 7: partial run interpolated from its length up to q
        seg = block[2, 1:4]
        xp_ = np.linspace(0.0, 1.0, seg.size)
        xq_ = np.linspace(0.0, 1.0, q)
        assert_allclose(y[2], np.interp(xq_, xp_, seg))

    def test_default_hyperparameters_match_sm_table_s1(self):
        # SM Table S1: mu=0.05, eta=1e-4, r=1e-4, Theta=1.5 (default).
        import inspect
        d = {k: v.default for k, v in inspect.signature(fsrd).parameters.items()}
        assert d['smoothness'] == 0.05
        assert d['eta'] == 1e-4
        assert d['rcond'] == 1e-4
        assert d['theta'] == 1.5
