"""PILOT: instrument a tiny char-level transformer training run.

Records, per training step: loss, grad norm, per-layer weight norms, and a
256-dim CountSketch (sparse JL) projection of the full flattened parameter
vector.  Optionally stores the full parameter vectors over a window (for the
invertible SVD-basis 'jump' test).

Usage:  python pilot_train_run.py SEED [--store-params]
"""
import argparse
import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "prompts_wt2.txt")
OUTDIR = os.path.join(HERE, "pilot_train_data")

# ---------------------------------------------------------------- config
N_LAYER, D_MODEL, N_HEAD, BLOCK = 4, 96, 4, 48
BATCH = 16
STEPS = 700
LR = 3e-3
WARMUP = 40
DROP_STEP = 350
DROP_FACTOR = 0.1
SKETCH_DIM = 256
SKETCH_SEED = 12345


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR * (DROP_FACTOR if step >= DROP_STEP else 1.0)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.fc2 = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        hd = C // N_HEAD
        q = q.view(B, T, N_HEAD, hd).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hd).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--store-params", action="store_true")
    ap.add_argument("--param-from", type=int, default=200)
    ap.add_argument("--param-to", type=int, default=700)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    text = open(CORPUS, "r", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = 20000
    train, val = data[:-n_val], data[-n_val:]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = TinyLM(len(chars))
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)

    # fixed CountSketch (sparse JL): identical across training seeds
    rs = np.random.RandomState(SKETCH_SEED)
    h_idx = rs.randint(0, SKETCH_DIM, size=n_par)
    h_sign = rs.choice([-1.0, 1.0], size=n_par).astype(np.float32)

    def sketch(vec):
        w = vec.numpy() * h_sign
        return np.bincount(h_idx, weights=w, minlength=SKETCH_DIM).astype(np.float64)

    g = torch.Generator().manual_seed(args.seed + 999)

    def batch(src):
        ix = torch.randint(len(src) - BLOCK - 1, (BATCH,), generator=g)
        x = torch.stack([src[i:i + BLOCK] for i in ix])
        y = torch.stack([src[i + 1:i + 1 + BLOCK] for i in ix])
        return x, y

    layer_names = [n for n, _ in model.named_parameters()]
    losses, gnorms, states, lnorms, lrs = [], [], [], [], []
    val_losses = []
    stored, stored_steps = [], []
    t0 = time.time()

    for step in range(STEPS):
        lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = batch(train)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters()))
        opt.step()

        fp = flat_params(model)
        losses.append(float(loss))
        gnorms.append(gn)
        lrs.append(lr)
        states.append(sketch(fp))
        lnorms.append([float(p.detach().norm()) for p in model.parameters()])
        if args.store_params and args.param_from <= step < args.param_to:
            stored.append(fp.numpy().astype(np.float32).copy())
            stored_steps.append(step)
        if step % 100 == 0:
            with torch.no_grad():
                vx, vy = batch(val)
                val_losses.append((step, float(model(vx, vy))))

    dt = time.time() - t0
    a = np.array(states).T  # (SKETCH_DIM, STEPS)
    out = dict(a=a, loss=np.array(losses), gnorm=np.array(gnorms),
               lr=np.array(lrs), lnorm=np.array(lnorms),
               val=np.array(val_losses), n_par=n_par, seed=args.seed,
               wall=dt)
    np.savez_compressed(os.path.join(OUTDIR, f"run_seed{args.seed}.npz"), **out)
    if stored:
        np.savez(os.path.join(OUTDIR, f"params_seed{args.seed}.npz"),
                 P=np.array(stored), steps=np.array(stored_steps))
    print(json.dumps(dict(seed=args.seed, n_par=n_par, wall=round(dt, 1),
                          loss_first=round(losses[0], 4),
                          loss_last=round(float(np.mean(losses[-20:])), 4),
                          val=val_losses)))


if __name__ == "__main__":
    main()
