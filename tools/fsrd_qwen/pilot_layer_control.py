#!/usr/bin/env python3
"""Positive control for the pilot's held-out protocol.

Plants a genuine 2-regime linear system in a signature space of the same shape as
the real experiment (D=4096, L=64 layers, latent rank 12, 1% noise), writes it in
the same .npz layout, and runs pilot_layer_fsrd.py's evaluator on it.  Confirms
the protocol can detect layer-banded linear dynamics when they exist, so a null on
the real weights is a statement about the weights and not about the protocol.

Result (recorded): fSRD depth-2 held-out rel-L2 0.011-0.027 at cuts 48/56/60 vs
0.95-1.02 for mean/global-DMD, ratio 0.395 aggregate -> passes the GO gate, and the
fitted region boundary lands at column 30-32 for a true switch at 32.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pilot_layer_control"


def build(path):
    rng = np.random.default_rng(7)
    L, D, K, r = 64, 4096, 64, 12
    U = np.linalg.qr(rng.standard_normal((D, r)))[0]

    def regime(seed, scale):
        rr = np.random.default_rng(seed)
        A = rr.standard_normal((r, r)) * 0.3
        w, V = np.linalg.eig(A)
        return (V @ np.diag(w / np.abs(w) * scale) @ np.linalg.inv(V)).real

    A1, A2 = regime(1, 0.98), regime(2, 1.01)
    z = np.zeros((r, L))
    z[:, 0] = rng.standard_normal(r)
    for l in range(1, L):
        z[:, l] = (A1 if l < 32 else A2) @ z[:, l - 1]
    X = U @ z
    X = X + 0.01 * np.linalg.norm(X) / np.sqrt(D * L) * rng.standard_normal((D, L))
    gram = np.stack([X.T.reshape(L, K, K)] * 2).astype(np.float32)
    raw = np.stack([X.T.reshape(L, 1, D)] * 2).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, layers=np.arange(L), raw=raw, gram=gram,
             rownorm=np.zeros((L, 5120), np.float32), fro=np.linalg.norm(X, axis=0),
             cos_next=np.full(L, 0.5), rel_next=np.full(L, 0.5),
             seeds=np.array([0, 1]), kind="synth", n_ff=D, n_embd=5120)


def main():
    build(os.path.join(SCRATCH, "results", "pilot_layer_sig_ffn_gate_s0.npz"))
    src = open(os.path.join(HERE, "pilot_layer_fsrd.py")).read()
    src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))',
                      f'HERE = {SCRATCH!r}')
    src = src.replace('sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))',
                      f'sys.path.insert(0, {HERE!r})')
    drv = os.path.join(SCRATCH, "_ctrl_driver.py")
    open(drv, "w").write(src)
    return subprocess.call([sys.executable, drv])


if __name__ == "__main__":
    raise SystemExit(main())
