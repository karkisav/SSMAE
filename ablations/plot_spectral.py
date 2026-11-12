#!/usr/bin/env python
"""Figure for Ablation 1: spectral masking ratio vs downstream quality.

Two panels rather than one dual-axis chart -- the classification probe is a
percentage and change detection is an F1, so they do not share a scale.

Palette: validated categorical slots 1/2/3. Slot 3 (#1baf7a) carries a contrast
WARN against the light surface, so that series is direct-labelled at every point
and a table view is printed -- the required relief.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = '/home/iit-dharwad/sav/ablations/results_spectral.json'
OUT = '/home/iit-dharwad/sav/ablations/fig_spectral.png'

SURFACE, TEXT_1, TEXT_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
S_FROZEN, S_FINE, S_CLS = '#2a78d6', '#eb6834', '#1baf7a'

# random-init references (ablation 5, diff fusion, identical protocol)
RAND_FROZEN, RAND_FINE = 0.8744, 0.9309

d = json.load(open(RES))
ks = [k for k in ('0.0', '0.15', '0.3', '0.5') if k in d]
x = [float(k) for k in ks]
cls = [d[k]['cls_linear_probe'] * 100 for k in ks]
froz = [d[k]['cd_frozen'] for k in ks]
fine = [d[k]['cd_finetuned'] for k in ks]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)

for ax in (axL, axR):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, linewidth=.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.set_xticks(x)
    ax.set_xlabel('Spectral masking ratio', fontsize=10.5, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9.5)

# --- left: EuroSAT linear probe (single series -> no legend; title names it) ---
axL.plot(x, cls, color=S_CLS, linewidth=2, marker='o', markersize=5.5,
         markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
for xi, yi in zip(x, cls):                      # contrast relief: label every point
    axL.annotate(f'{yi:.1f}%', (xi, yi), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9, color=TEXT_2)
axL.axhline(10, color=GRID, linestyle=(0, (4, 4)), linewidth=1.2, zorder=1)
axL.annotate('chance (10 classes)', (0.5, 10), textcoords='offset points',
             xytext=(-4, 6), ha='right', fontsize=8.5, color=TEXT_2)
axL.set_ylabel('EuroSAT linear-probe accuracy (%)', fontsize=10.5, color=TEXT_2)
axL.set_title('Frozen features lose class information', fontsize=11.5,
              color=TEXT_1, loc='left', pad=10, fontweight='bold')
axL.set_ylim(0, 88)

# --- right: change detection F1, two series + random-init references ---
axR.axhline(RAND_FINE, color=S_FINE, linestyle=(0, (5, 4)), linewidth=1.4, alpha=.85)
axR.axhline(RAND_FROZEN, color=S_FROZEN, linestyle=(0, (5, 4)), linewidth=1.4, alpha=.85)
axR.annotate(f'random init  {RAND_FINE:.3f}', (0.5, RAND_FINE), textcoords='offset points',
             xytext=(-4, 5), ha='right', fontsize=8.5, color=S_FINE, fontweight='bold')
axR.annotate(f'random init  {RAND_FROZEN:.3f}', (0.5, RAND_FROZEN), textcoords='offset points',
             xytext=(-4, -14), ha='right', fontsize=8.5, color=S_FROZEN, fontweight='bold')
for ys, c, lab in ((fine, S_FINE, 'Finetuned encoder'),
                   (froz, S_FROZEN, 'Frozen encoder')):
    axR.plot(x, ys, color=c, linewidth=2, marker='o', markersize=5.5,
             markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label=lab)
for xi, yi, dx, ha in ((x[0], froz[0], 8, 'left'), (x[-1], froz[-1], -8, 'right')):
    axR.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                 xytext=(dx, -4), ha=ha, va='top', fontsize=9, color=TEXT_2)
axR.margins(y=.08)
axR.set_ylabel('Change-detection F1 (diff fusion)', fontsize=10.5, color=TEXT_2)
axR.set_title('Pretraining only beats random when spectral masking is off',
              fontsize=11.5, color=TEXT_1, loc='left', pad=10, fontweight='bold')
leg = axR.legend(loc='lower left', frameon=False, fontsize=9.5)
for t in leg.get_texts():
    t.set_color(TEXT_2)

fig.suptitle('Spectral masking monotonically degrades every downstream readout',
             fontsize=13.5, color=TEXT_1, x=.008, ha='left', y=1.0, fontweight='bold')
fig.text(.008, .945, 'SS-MAE, spatial masking fixed at 0.75, 100 pretraining epochs · '
                     'published setting is 0.30', fontsize=9, color=TEXT_2, ha='left')
plt.tight_layout(rect=[0, 0, 1, .935])
plt.savefig(OUT, facecolor=SURFACE, bbox_inches='tight')
print(f'saved -> {OUT}\n')

# table view (required relief for the contrast WARN)
print(f"{'spectral':>9} {'masked%':>8} {'EuroSAT probe':>14} {'CD frozen':>10} {'CD finetune':>12}")
for k in ks:
    r = d[k]
    print(f"{k:>9} {r['masked_fraction']*100:>7.1f}% {r['cls_linear_probe']*100:>13.2f}% "
          f"{r['cd_frozen']:>10.4f} {r['cd_finetuned']:>12.4f}")
print(f"{'random':>9} {'--':>8} {'--':>14} {RAND_FROZEN:>10.4f} {RAND_FINE:>12.4f}")
