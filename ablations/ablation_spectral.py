#!/usr/bin/env python
"""Ablation 1 — does SPECTRAL masking (the "dual" in SS-MAE) do anything?

Dual spatial-spectral masking is the method's stated novelty. Setting
SPECTRAL_MASK_RATIO = 0 reduces it to an ordinary spatial MAE, so this sweep
asks directly whether the spectral half earns its place.

  spectral = 0.00   pure spatial MAE (the ablation)
  spectral = 0.15   light
  spectral = 0.30   as published  -- reuses checkpoints/pretrain/epoch100
  spectral = 0.50   heavy

Spatial masking is held at 0.75 throughout, so total masked fraction varies as
1 - 0.25*(1-spectral): 75% / 79% / 82.5% / 87.5%.

Two downstream readouts, because ablation 2 showed they disagree:
  cls    linear probe on EuroSAT classification (where SS-MAE looked strong)
  cd     change detection, frozen + finetuned, using the `diff` fusion that
         ablation 5 found best

Writes ablations/results_spectral.json. Pretrained weights go to
ablations/ckpt/ so the main checkpoints/ tree is untouched.
"""
import ast, io, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

NB = '/home/iit-dharwad/sav/rsproject-iiitdwd (5).ipynb'
OUT = '/home/iit-dharwad/sav/ablations/results_spectral.json'
CKPT_OUT = '/home/iit-dharwad/sav/ablations/ckpt'
PUBLISHED = '/home/iit-dharwad/sav/checkpoints/pretrain/ssmae_pretrain_epoch100.pth'

PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 1.5e-4
PRETRAIN_BS = 128
CD_EPOCHS, HEAD_LR, ENC_LR = 20, 3e-4, 1e-5
CLS_EPOCHS, CLS_LR = 10, 1e-3
SEED = 42
RATIOS = [0.0, 0.15, 0.3, 0.5]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(CKPT_OUT, exist_ok=True)


def load_notebook_defs(cells):
    nb = json.load(io.open(NB, encoding='utf-8'))
    ns = dict(torch=torch, nn=nn, np=np, numpy=np, os=os, Dataset=Dataset,
              DataLoader=DataLoader, device=device, f1_score=f1_score,
              roc_auc_score=roc_auc_score, accuracy_score=accuracy_score)
    import pandas as pd, rasterio, random
    from einops import rearrange
    ns.update(pd=pd, pandas=pd, rasterio=rasterio, random=random, rearrange=rearrange)
    for ci in cells:
        src = ''.join(nb['cells'][ci]['source'])
        src = '\n'.join('' if l.lstrip().startswith(('!', '%')) else l
                        for l in src.split('\n'))
        tree = ast.parse(src)
        keep = tree.body if ci == 2 else [
            n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=keep, type_ignores=[]), f'<cell{ci}>', 'exec'), ns)
    return ns


NS = load_notebook_defs([2, 3, 4, 5, 7, 26, 27])
Config, EuroSATDataset = NS['Config'], NS['EuroSATDataset']
SSMAE, ChangeDetector = NS['SSMAE'], NS['ChangeDetector']
SSMAEClassifier = NS['SSMAEClassifier']
BalancedExploitationDataset = NS['BalancedExploitationDataset']


def cfg_with(spectral):
    """Config clone with a different spectral masking ratio.

    SSMAE builds its DualMasking from config at __init__, so overriding the
    attribute on an instance is enough to change the pretext task.
    """
    c = Config()
    c.SPECTRAL_MASK_RATIO = spectral
    return c


class DiffChangeDetector(ChangeDetector):
    """Signed fusion (f1 - f2), the best operator from ablation 5."""
    def forward(self, img1, img2):
        return self.change_head(self.encode(img1) - self.encode(img2)).squeeze(1)


# ---------------------------------------------------------------- pretraining
def pretrain(spectral, loader):
    path = os.path.join(CKPT_OUT, f'ssmae_spectral{int(spectral*100):03d}.pth')
    if os.path.exists(path):
        print(f"  reusing {os.path.basename(path)}", flush=True)
        return path, json.load(open(path + '.json'))['losses']
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SSMAE(cfg_with(spectral)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PRETRAIN_EPOCHS)
    losses = []
    t0 = time.time()
    for ep in range(PRETRAIN_EPOCHS):
        model.train(); tot = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            loss, _, _ = model(imgs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sch.step(); losses.append(tot / len(loader))
        if (ep + 1) % 20 == 0:
            print(f"    spectral={spectral}  epoch {ep+1:>3}  loss={losses[-1]:.4f}"
                  f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(model.state_dict(), path)
    json.dump({'losses': losses}, open(path + '.json', 'w'))
    print(f"  pretrained spectral={spectral} in {(time.time()-t0)/60:.1f} min"
          f"  final loss={losses[-1]:.4f}", flush=True)
    return path, losses


# ---------------------------------------------------------------- readouts
def probe_classification(ckpt, spectral, tl, vl):
    """Linear probe: frozen encoder, single linear layer, EuroSAT 10-way."""
    torch.manual_seed(SEED)
    enc = SSMAE(cfg_with(spectral)).to(device)
    enc.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    clf = SSMAEClassifier(encoder=enc, num_classes=10, embed_dim=512).to(device)
    for p in clf.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(clf.classifier.parameters(), lr=CLS_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(CLS_EPOCHS):
        clf.train()
        for x, y in tl:
            opt.zero_grad(); crit(clf(x.to(device)), y.to(device)).backward(); opt.step()
        clf.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in vl:
                P.extend(clf(x.to(device)).argmax(1).cpu().numpy()); Y.extend(y.numpy())
        best = max(best, accuracy_score(Y, P))
    return float(best)


def probe_change(ckpt, spectral, frozen, tl, vl):
    torch.manual_seed(SEED); np.random.seed(SEED)
    enc = SSMAE(cfg_with(spectral)).to(device)
    enc.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    det = DiffChangeDetector(enc, embed_dim=512, freeze_encoder=frozen).to(device)
    if frozen:
        opt = torch.optim.AdamW([p for p in det.parameters() if p.requires_grad],
                                lr=HEAD_LR, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {'params': det.encoder.parameters(), 'lr': ENC_LR},
            {'params': det.change_head.parameters(), 'lr': HEAD_LR}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CD_EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    best = -1.0
    for _ in range(CD_EPOCHS):
        det.train()
        for a, b, y in tl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad(); crit(det(a, b), y).backward()
            nn.utils.clip_grad_norm_(det.parameters(), 1.0); opt.step()
        sch.step()
        det.eval(); P, Y = [], []
        with torch.no_grad():
            for a, b, y in vl:
                P.extend(torch.sigmoid(det(a.to(device), b.to(device))).cpu().numpy())
                Y.extend(y.numpy())
        best = max(best, f1_score(np.array(Y), (np.array(P) >= .5).astype(int)))
    return float(best)


def main():
    base = Config()
    t0 = time.time()
    print("building datasets...", flush=True)
    tr_ds = EuroSATDataset(base.TRAIN_CSV, base.IMG_DIR)
    va_ds = EuroSATDataset(base.VAL_CSV, base.IMG_DIR)
    pre_loader = DataLoader(tr_ds, batch_size=PRETRAIN_BS, shuffle=True,
                            num_workers=8, pin_memory=True, persistent_workers=True)
    cls_tl = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=8,
                        pin_memory=True, persistent_workers=True)
    cls_vl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=8,
                        pin_memory=True, persistent_workers=True)
    cd_tl = DataLoader(BalancedExploitationDataset(tr_ds, num_pairs=2000, seed=42),
                       batch_size=32, shuffle=True, num_workers=8,
                       pin_memory=True, persistent_workers=True)
    cd_vl = DataLoader(BalancedExploitationDataset(va_ds, num_pairs=600, seed=99),
                       batch_size=32, shuffle=False, num_workers=8,
                       pin_memory=True, persistent_workers=True)
    print(f"datasets ready in {(time.time()-t0)/60:.1f} min\n", flush=True)

    results = {'meta': {'pretrain_epochs': PRETRAIN_EPOCHS, 'spatial': base.SPATIAL_MASK_RATIO,
                        'seed': SEED, 'fusion': 'diff', 'ratios': RATIOS}}
    for sp in RATIOS:
        print(f"=== spectral = {sp} ===", flush=True)
        if abs(sp - 0.3) < 1e-9 and os.path.exists(PUBLISHED):
            ckpt, losses = PUBLISHED, None      # the published run
            print("  using the published epoch-100 checkpoint", flush=True)
        else:
            ckpt, losses = pretrain(sp, pre_loader)
        r = {'ckpt': ckpt, 'pretrain_final_loss': (losses[-1] if losses else None),
             'masked_fraction': 1 - (1 - base.SPATIAL_MASK_RATIO) * (1 - sp)}
        r['cls_linear_probe'] = probe_classification(ckpt, sp, cls_tl, cls_vl)
        print(f"  EuroSAT linear probe : {r['cls_linear_probe']*100:.2f}%", flush=True)
        r['cd_frozen'] = probe_change(ckpt, sp, True, cd_tl, cd_vl)
        print(f"  change det (frozen)  : {r['cd_frozen']:.4f}", flush=True)
        r['cd_finetuned'] = probe_change(ckpt, sp, False, cd_tl, cd_vl)
        print(f"  change det (finetune): {r['cd_finetuned']:.4f}", flush=True)
        results[str(sp)] = r
        json.dump(results, open(OUT, 'w'), indent=1)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
