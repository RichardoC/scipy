""" Benchmark linalg.fsrd for various tree depths and split modes.

"""
import numpy as np

from .common import Benchmark, safe_import

with safe_import():
    import scipy.linalg


def _snapshot_matrix(rng):
    # A two-regime, multi-frequency signal, time-delay embedded into a Hankel
    # matrix -- the kind of nonlinear/chaotic-like input fSRD targets.
    t = np.linspace(0, 40, 900)
    sig = np.where(t < 20, np.sin(2.0 * t), np.sin(0.6 * t) * np.cos(0.2 * t))
    sig = sig + 0.02 * rng.standard_normal(sig.size)
    delays = 60
    cols = sig.size - delays
    return np.array([sig[d:d + cols] for d in range(delays + 1)])


class FSRD(Benchmark):
    params = [
        [0, 2, 4],
        [False, True],
    ]
    param_names = ['max_depth', 'oblique']

    def setup(self, max_depth, oblique):
        rng = np.random.default_rng(1742808411247533)
        self.a = _snapshot_matrix(rng)

    def time_fsrd(self, max_depth, oblique):
        scipy.linalg.fsrd(self.a, dt=0.05, max_depth=max_depth,
                          oblique=oblique)

    def peakmem_fsrd(self, max_depth, oblique):
        scipy.linalg.fsrd(self.a, dt=0.05, max_depth=max_depth,
                          oblique=oblique)
