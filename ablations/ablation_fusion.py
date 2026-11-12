#!/usr/bin/env python
"""Ablation 5 — does the fusion operator explain why pretraining loses?

The change head consumes |f1 - f2|, an L1 distance. Two consequences:

  1. It is SYMMETRIC, so Forest->Urban and Urban->Forest are indistinguishable.
     For an encroachment detector, direction is the whole point.
  2. A random high-dimensional projection approximately preserves pairwise
     distances (Johnson-Lindenstrauss), so a random ViT is already a competent
     distance encoder -- which is the leading explanation for why random init
     beat every pretrained checkpoint in ablation 2.

If the distance operator is what flatters random init, then a fusion that keeps
the two embeddings separate (concat) should let pretrained features win. That
would convert ablation 2's negative result into a design finding.

Grid: 5 fusion operators x {pretrained ep100, random init} x {frozen, finetuned}.
Everything else is held at the ablation-2 settings so numbers are comparable.

Writes ablations/results_fusion.json ; plotting is separate.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo root
import ast, io, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

NB = str(ROOT / 'notebooks' / 'ssmae_pipeline.ipynb')
OUT = str(ROOT / 'ablations' / 'results' / 'results_fusion.json')
PRETRAINED = str(ROOT / 'checkpoints' / 'pretrain' / 'ssmae_pretrain_epoch100.pth')

EPOCHS, HEAD_LR, ENC_LR, THRESHOLD, SEED = 20, 3e-4, 1e-5, 0.5, 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_notebook_defs(cells):
    nb = json.load(io.open(NB, encoding='utf-8'))
    ns = dict(torch=torch, nn=nn, np=np, numpy=np, os=os, Dataset=Dataset,
              DataLoader=DataLoader, device=device, f1_score=f1_score,
              roc_auc_score=roc_auc_score)
    import pandas as pd, rasterio, random
    from einops import rearrange
    ns.update(pd=pd, pandas=pd, rasterio=rasterio, random=random, rearrange=rearrange)
    FULL_EXEC = {2}
    for ci in cells:
        src = ''.join(nb['cells'][ci]['source'])
        src = '\n'.join('' if l.lstrip().startswith(('!', '%')) else l
                        for l in src.split('\n'))
        tree = ast.parse(src)
        keep = tree.body if ci in FULL_EXEC else [
            n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=keep, type_ignores=[]), f'<cell{ci}>', 'exec'), ns)
    return ns


NS = load_notebook_defs([2, 3, 4, 5, 26, 27])
Config, EuroSATDataset = NS['Config'], NS['EuroSATDataset']
SSMAE, ChangeDetector = NS['SSMAE'], NS['ChangeDetector']
BalancedExploitationDataset = NS['BalancedExploitationDataset']
config = Config()

# width of the fused vector, in units of embed_dim
FUSIONS = {
    'abs':        (1, '|f1 - f2|            (current, symmetric)'),
    'diff':       (1, 'f1 - f2              (signed)'),
    'concat':     (2, '[f1, f2]             (order-aware)'),
    'absprod':    (2, '[|f1-f2|, f1*f2]     (distance + interaction)'),
    'concat_abs': (3, '[f1, f2, |f1-f2|]    (siamese standard)'),
}


class FusionChangeDetector(ChangeDetector):
    """ChangeDetector with a swappable fusion operator.

    Reuses the notebook's encode() verbatim; only the fusion and the head's
    input width change, so any difference is attributable to fusion alone.
    """

    def __init__(self, encoder, embed_dim=512, freeze_encoder=True, fusion='abs'):
        super().__init__(encoder, embed_dim=embed_dim, freeze_encoder=freeze_encoder)
        self.fusion = fusion
        mult = FUSIONS[fusion][0]
        self.change_head = nn.Sequential(      # same depth/width as the notebook's
            nn.Linear(embed_dim * mult, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def fuse(self, f1, f2):
        if self.fusion == 'abs':        return torch.abs(f1 - f2)
        if self.fusion == 'diff':       return f1 - f2
        if self.fusion == 'concat':     return torch.cat([f1, f2], dim=1)
        if self.fusion == 'absprod':    return torch.cat([torch.abs(f1 - f2), f1 * f2], dim=1)
        if self.fusion == 'concat_abs': return torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)
        raise ValueError(self.fusion)

    def forward(self, img1, img2):
        return self.change_head(self.fuse(self.encode(img1), self.encode(img2))).squeeze(1)


def build_loaders():
    t0 = time.time()
    tr = BalancedExploitationDataset(EuroSATDataset(config.TRAIN_CSV, config.IMG_DIR),
                                     num_pairs=2000, seed=42)
    va = BalancedExploitationDataset(EuroSATDataset(config.VAL_CSV, config.IMG_DIR),
                                     num_pairs=600, seed=99)
    tl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=8,
                    pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=32, shuffle=False, num_workers=8,
                    pin_memory=True, persistent_workers=True)
    print(f"data ready in {time.time()-t0:.0f}s", flush=True)
    return tl, vl


def run(fusion, pretrained, frozen, tl, vl, tag):
    torch.manual_seed(SEED); np.random.seed(SEED)
    enc = SSMAE(config).to(device)
    if pretrained:
        enc.load_state_dict(torch.load(PRETRAINED, map_location=device, weights_only=False))
    det = FusionChangeDetector(enc, embed_dim=config.EMBED_DIM,
                               freeze_encoder=frozen, fusion=fusion).to(device)

    if frozen:
        opt = torch.optim.AdamW([p for p in det.parameters() if p.requires_grad],
                                lr=HEAD_LR, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {'params': det.encoder.parameters(), 'lr': ENC_LR},
            {'params': det.change_head.parameters(), 'lr': HEAD_LR}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()

    best_f1, best_auc = -1.0, float('nan')
    for _ in range(EPOCHS):
        det.train()
        for i1, i2, y in tl:
            i1, i2, y = i1.to(device), i2.to(device), y.to(device)
            opt.zero_grad(); crit(det(i1, i2), y).backward()
            nn.utils.clip_grad_norm_(det.parameters(), 1.0); opt.step()
        sched.step()
        det.eval(); ys, ps = [], []
        with torch.no_grad():
            for i1, i2, y in vl:
                ps.extend(torch.sigmoid(det(i1.to(device), i2.to(device))).cpu().numpy())
                ys.extend(y.numpy())
        ys = np.array(ys); pr = (np.array(ps) >= THRESHOLD).astype(int)
        f1 = f1_score(ys, pr)
        if f1 > best_f1:
            best_f1, best_auc = float(f1), float(roc_auc_score(ys, np.array(ps)))
    head_params = sum(p.numel() for p in det.change_head.parameters())
    print(f"  {tag:<46} F1={best_f1:.4f}  AUC={best_auc:.4f}", flush=True)
    return {'best_f1': best_f1, 'best_auc': best_auc, 'head_params': head_params}


def main():
    tl, vl = build_loaders()
    results = {'meta': {'epochs': EPOCHS, 'seed': SEED, 'ckpt': 'epoch100',
                        'fusions': {k: v[1] for k, v in FUSIONS.items()}}}
    n = len(FUSIONS) * 4
    print(f"\n{n} runs (5 fusions x pretrained/random x frozen/finetuned)\n", flush=True)
    t0 = time.time()
    for regime, frozen in (('frozen', True), ('finetuned', False)):
        for init, pre in (('pretrained', True), ('random', False)):
            key = f'{regime}|{init}'
            results[key] = {}
            print(f"--- {regime} / {init} ---", flush=True)
            for fu in FUSIONS:
                results[key][fu] = run(fu, pre, frozen, tl, vl, f'{regime}/{init}/{fu}')
                json.dump(results, open(OUT, 'w'), indent=1)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
