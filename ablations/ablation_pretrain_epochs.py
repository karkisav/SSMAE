#!/usr/bin/env python
"""Ablation 2 — does MORE SS-MAE pretraining give BETTER change detection?

Takes each self-supervised checkpoint (epochs 10..100) plus a random-init
control, trains the change detector from it under one fixed budget, and records
the best validation F1.

Why this matters: the matched-budget comparison found SS-MAE (0.8781) losing to
a random encoder (0.8859), and the label-efficiency sweep found SS-MAE behind at
4 of 6 label fractions. If downstream F1 is flat or falling in pretraining
epochs, the pretext task is not learning transferable structure -- which is a
direct, mechanistic explanation rather than a hyperparameter accident.

Two regimes per checkpoint:
  frozen     -- encoder fixed, head only. Probes representation quality directly.
  finetuned  -- encoder at 1e-5, head at 3e-4. The practical setting.

Class definitions are lifted from the notebook itself (AST extraction), so this
cannot drift from the architecture that produced the published numbers.

Writes ablations/results_pretrain_epochs.json ; plotting is separate.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo root
import ast, io, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

NB = str(ROOT / 'notebooks' / 'ssmae_pipeline.ipynb')
OUT = str(ROOT / 'ablations' / 'results' / 'results_pretrain_epochs.json')
CKPT_DIR = str(ROOT / 'checkpoints' / 'pretrain')

EPOCHS = 20           # matches cells 28/29
HEAD_LR = 3e-4
ENC_LR = 1e-5
THRESHOLD = 0.5
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------- notebook code
def load_notebook_defs(cells):
    """Pull only class/function definitions out of the given notebook cells.

    Executing the cells wholesale would re-run training; we want just the
    definitions, guaranteeing the ablation uses the notebook's exact classes.
    """
    nb = json.load(io.open(NB, encoding='utf-8'))
    ns = dict(torch=torch, nn=nn, np=np, numpy=np, os=os, Dataset=Dataset,
              DataLoader=DataLoader, device=device, f1_score=f1_score,
              roc_auc_score=roc_auc_score)
    import pandas as pd, rasterio, random
    from einops import rearrange
    ns.update(pd=pd, pandas=pd, rasterio=rasterio, random=random, rearrange=rearrange)
    # Cell 2 is pure configuration/helpers with no training, so it runs in full
    # (its Config class closes over module-level names like _EUROSAT_ROOT).
    # The rest contribute definitions only -- executing them would load datasets
    # and start training.
    FULL_EXEC = {2}
    for ci in cells:
        src = ''.join(nb['cells'][ci]['source'])
        src = '\n'.join('' if l.lstrip().startswith(('!', '%')) else l
                        for l in src.split('\n'))
        tree = ast.parse(src)
        if ci in FULL_EXEC:
            keep = tree.body
        else:
            keep = [n for n in tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Import,
                                      ast.ImportFrom))]
        exec(compile(ast.Module(body=keep, type_ignores=[]), f'<cell{ci}>', 'exec'), ns)
    return ns


NS = load_notebook_defs([2, 3, 4, 5, 26, 27])
Config = NS['Config']
EuroSATDataset = NS['EuroSATDataset']
SSMAE = NS['SSMAE']
BalancedExploitationDataset = NS['BalancedExploitationDataset']
ChangeDetector = NS['ChangeDetector']

config = Config()
print(f"device={device}  bands={config.NUM_BANDS}  embed={config.EMBED_DIM}", flush=True)


# ---------------------------------------------------------------- data (once)
def build_loaders():
    t0 = time.time()
    train_ds = EuroSATDataset(config.TRAIN_CSV, config.IMG_DIR)
    val_ds = EuroSATDataset(config.VAL_CSV, config.IMG_DIR)
    # identical construction to notebook cell 26
    train_pairs = BalancedExploitationDataset(train_ds, num_pairs=2000, seed=42)
    val_pairs = BalancedExploitationDataset(val_ds, num_pairs=600, seed=99)
    tl = DataLoader(train_pairs, batch_size=32, shuffle=True,
                    num_workers=8, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val_pairs, batch_size=32, shuffle=False,
                    num_workers=8, pin_memory=True, persistent_workers=True)
    print(f"data ready in {time.time()-t0:.0f}s  "
          f"({len(train_pairs)} train / {len(val_pairs)} val pairs)", flush=True)
    return tl, vl


# ---------------------------------------------------------------- one run
def run(ckpt_path, frozen, train_loader, val_loader, tag):
    torch.manual_seed(SEED); np.random.seed(SEED)

    encoder = SSMAE(config).to(device)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(sd)

    det = ChangeDetector(encoder=encoder, embed_dim=config.EMBED_DIM,
                         freeze_encoder=frozen).to(device)

    if frozen:
        opt = torch.optim.AdamW(
            [p for p in det.parameters() if p.requires_grad],
            lr=HEAD_LR, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {'params': det.encoder.parameters(), 'lr': ENC_LR},
            {'params': det.change_head.parameters(), 'lr': HEAD_LR},
        ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()

    best_f1, best_auc, curve = -1.0, float('nan'), []
    for ep in range(1, EPOCHS + 1):
        det.train()
        for img1, img2, y in train_loader:
            img1, img2, y = img1.to(device), img2.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(det(img1, img2), y)
            loss.backward()
            nn.utils.clip_grad_norm_(det.parameters(), 1.0)
            opt.step()
        sched.step()

        det.eval()
        ys, ps = [], []
        with torch.no_grad():
            for img1, img2, y in val_loader:
                ps.extend(torch.sigmoid(det(img1.to(device), img2.to(device))).cpu().numpy())
                ys.extend(y.numpy())
        ys = np.array(ys); pr = (np.array(ps) >= THRESHOLD).astype(int)
        f1 = f1_score(ys, pr)
        curve.append(float(f1))
        if f1 > best_f1:
            best_f1 = float(f1)
            best_auc = float(roc_auc_score(ys, np.array(ps)))
    print(f"  {tag:<28} best F1={best_f1:.4f}  AUC={best_auc:.4f}", flush=True)
    return {'best_f1': best_f1, 'best_auc': best_auc, 'curve': curve}


def main():
    train_loader, val_loader = build_loaders()

    points = [(0, None)] + [(e, os.path.join(CKPT_DIR, f'ssmae_pretrain_epoch{e}.pth'))
                            for e in range(10, 101, 10)]
    points = [(e, p) for e, p in points if p is None or os.path.exists(p)]
    print(f"\n{len(points)} checkpoints x 2 regimes = {len(points)*2} runs\n", flush=True)

    results = {'meta': {'epochs': EPOCHS, 'head_lr': HEAD_LR, 'enc_lr': ENC_LR,
                        'seed': SEED, 'train_pairs': 2000, 'val_pairs': 600},
               'frozen': {}, 'finetuned': {}}

    t0 = time.time()
    for regime, frozen in (('frozen', True), ('finetuned', False)):
        print(f"--- regime: {regime} ---", flush=True)
        for ep, path in points:
            tag = f"{regime} @ pretrain epoch {ep}" + (" (random init)" if path is None else "")
            results[regime][str(ep)] = run(path, frozen, train_loader, val_loader, tag)
            json.dump(results, open(OUT, 'w'), indent=1)   # checkpoint after each
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
