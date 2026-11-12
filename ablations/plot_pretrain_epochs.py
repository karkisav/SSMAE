#!/usr/bin/env python
"""Figure for Ablation 2: downstream change-detection F1 vs SS-MAE pretraining epochs.

Two series (frozen / finetuned probe) against a random-init reference line.
Palette: validated categorical slots 1 and 2 (blue #2a78d6, orange #eb6834) --
CVD separation dE 24.7 protan, normal-vision dE 33.6, all six checks PASS.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = '/home/iit-dharwad/sav/ablations/results_pretrain_epochs.json'
OUT = '/home/iit-dharwad/sav/ablations/fig_pretrain_epochs.png'

# --- design tokens (light surface) ---
SURFACE   = '#fcfcfb'
TEXT_1    = '#0b0b0b'
TEXT_2    = '#52514e'
GRID      = '#e1e0d9'
S_FROZEN  = '#2a78d6'   # categorical slot 1
S_FINETUN = '#eb6834'   # categorical slot 2

d = json.load(open(RES))
eps = sorted(int(e) for e in d['frozen'] if e != '0')
froz = [d['frozen'][str(e)]['best_f1'] for e in eps]
fine = [d['finetuned'][str(e)]['best_f1'] for e in eps]
rand_f = d['frozen']['0']['best_f1']
rand_t = d['finetuned']['0']['best_f1']

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

# recessive grid, no top/right spines
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)

# random-init references: the thing both series must beat
ax.axhline(rand_t, color=S_FINETUN, linestyle=(0, (5, 4)), linewidth=1.4, alpha=.85, zorder=1)
ax.axhline(rand_f, color=S_FROZEN,  linestyle=(0, (5, 4)), linewidth=1.4, alpha=.85, zorder=1)
# labels sit at the left edge, above their line, clear of the series endpoints
ax.text(9, rand_t + .004, f'random init  {rand_t:.3f}', color=S_FINETUN,
        va='bottom', ha='left', fontsize=8.5, fontweight='bold')
ax.text(9, rand_f + .004, f'random init  {rand_f:.3f}', color=S_FROZEN,
        va='bottom', ha='left', fontsize=8.5, fontweight='bold')

# 2px lines, >=8px markers, 2px surface ring so overlaps stay legible
for ys, color, label in ((fine, S_FINETUN, 'Finetuned encoder'),
                         (froz, S_FROZEN,  'Frozen encoder (linear probe)')):
    ax.plot(eps, ys, color=color, linewidth=2, zorder=3, label=label,
            marker='o', markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=2)

# selective direct labels only: endpoint and worst point per series
for ys, color, dy in ((fine, S_FINETUN, -15), (froz, S_FROZEN, -15)):
    lo = min(range(len(ys)), key=lambda i: ys[i])
    for i in (lo, len(ys) - 1):
        ax.annotate(f'{ys[i]:.3f}', (eps[i], ys[i]), textcoords='offset points',
                    xytext=(-2 if i == len(ys) - 1 else 0, dy),
                    ha='right' if i == len(ys) - 1 else 'center',
                    fontsize=8.5, color=TEXT_2)

ax.set_xlabel('SS-MAE pretraining epochs', fontsize=10.5, color=TEXT_2)
ax.set_ylabel('Best validation F1 (change detection)', fontsize=10.5, color=TEXT_2)
ax.set_title('More SS-MAE pretraining does not improve change detection',
             fontsize=13, color=TEXT_1, pad=14, loc='left', fontweight='bold')
ax.text(0, 1.015, 'EuroSAT exploitation pairs · 2000 train / 600 val · 20 epochs · seed 42',
        transform=ax.transAxes, fontsize=9, color=TEXT_2)
ax.set_xticks(eps)
ax.set_xlim(5, 106)
ax.tick_params(colors=TEXT_2, labelsize=9.5)
leg = ax.legend(loc='lower right', frameon=False, fontsize=9.5)
for t in leg.get_texts():
    t.set_color(TEXT_2)

plt.tight_layout()
plt.savefig(OUT, facecolor=SURFACE, bbox_inches='tight')
print(f'saved -> {OUT}')

# table view (accessibility: identity never color-alone)
print(f"\n{'pretrain':>9} {'frozen F1':>10} {'finetuned F1':>13}")
print(f"{'random':>9} {rand_f:>10.4f} {rand_t:>13.4f}")
for e, a, b in zip(eps, froz, fine):
    print(f"{e:>9} {a:>10.4f} {b:>13.4f}")
