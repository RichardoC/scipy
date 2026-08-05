"""pilot_ckpt_compress §0.1: train the tiny LM and dump every 2nd step's FULL
fp32 parameter vector to a memmap in pilot_ckpt_data/.

Model / optimiser / corpus / batch stream are IDENTICAL to pilot_train_run.py:
the classes and constants are imported from that module and the training loop
is replicated verbatim (including the every-100-step val batch draws, which
advance the same torch.Generator and therefore change the batch stream).

Usage: python pilot_ckpt_train.py SEED
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse — not rewrite — the sibling pilot's model/optimiser/corpus (spec §0.1)
from pilot_train_run import (TinyLM, lr_at, CORPUS, BATCH, STEPS, LR, BLOCK)

OUTDIR = os.path.join(HERE, "pilot_ckpt_data")
STORE_EVERY = 2          # every 2nd step -> 350 checkpoints (spec §0.1)


def main():
    seed = int(sys.argv[1])
    os.makedirs(OUTDIR, exist_ok=True)

    text = open(CORPUS, "r", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = 20000
    train, val = data[:-n_val], data[-n_val:]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyLM(len(chars))
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)

    g = torch.Generator().manual_seed(seed + 999)

    def batch(src):
        ix = torch.randint(len(src) - BLOCK - 1, (BATCH,), generator=g)
        x = torch.stack([src[i:i + BLOCK] for i in ix])
        y = torch.stack([src[i + 1:i + 1 + BLOCK] for i in ix])
        return x, y

    n_store = STEPS // STORE_EVERY                      # 350
    mm_path = os.path.join(OUTDIR, f"ckpts_seed{seed}.f32")
    mm = np.memmap(mm_path, dtype=np.float32, mode="w+",
                   shape=(n_store, n_par))
    stored_steps = []

    losses, val_losses = [], []
    t0 = time.time()
    for step in range(STEPS):
        lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = batch(train)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % STORE_EVERY == 0:
            k = step // STORE_EVERY
            with torch.no_grad():
                fp = torch.cat([p.detach().reshape(-1)
                                for p in model.parameters()])
            mm[k, :] = fp.numpy().astype(np.float32)
            stored_steps.append(step)
        if step % 100 == 0:
            # kept verbatim from pilot_train_run.py: this draw advances g
            with torch.no_grad():
                vx, vy = batch(val)
                val_losses.append((step, float(model(vx, vy))))

    mm.flush()
    del mm
    wall = time.time() - t0
    meta = dict(seed=seed, n_par=n_par, n_store=n_store,
                stored_steps=stored_steps, vocab=len(chars),
                loss_first=losses[0],
                loss_last=float(np.mean(losses[-20:])),
                val=val_losses, wall=round(wall, 1),
                bytes=os.path.getsize(mm_path))
    with open(os.path.join(OUTDIR, f"ckpts_seed{seed}_meta.json"), "w") as f:
        json.dump(meta, f)
    print(json.dumps({k: meta[k] for k in
                      ("seed", "n_par", "n_store", "loss_first", "loss_last",
                       "wall", "bytes")}))
    print("val:", val_losses)


if __name__ == "__main__":
    main()
