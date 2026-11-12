#!/usr/bin/env python
"""Figure for Ablation 5: fusion operator vs downstream F1.

Grouped bars, two panels (frozen / finetuned) so each keeps one y-axis.
Palette: validated categorical slots 1 (#2a78d6) and 2 (#eb6834).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = '/home/iit-dharwad/sav/ablations/results_fusion.json'
OUT = '/home/iit-dharwad/sav/ablations/fig_fusion.png'

SURFACE, TEXT_1, TEXT_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
C_PRE, C_RND = '#2a78d6', '#eb6834'

d = json.load(open(RES))
F = ['abs', 'diff', 'concat', 'absprod', 'concat_abs']
LAB = ['|f1-f2|\n(current)', 'f1-f2\n(signed)', '[f1,f2]', '[|f1-f2|,\nf1*f2]', '[f1,f2,\n|f1-f2|]']

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
xs = np.arange(len(F)); w = 0.38

for ax, regime, title in zip(axes, ('frozen', 'finetuned'),
                             ('Frozen encoder (linear probe)', 'Finetuned encoder')):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, linewidth=.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    pre = [d[f'{regime}|pretrained'][f]['best_f1'] for f in F]
    rnd = [d[f'{regime}|random'][f]['best_f1'] for f in F]
    # 2px surface gap between adjacent bars
    ax.bar(xs - w/2 - .01, pre, w, color=C_PRE, label='SS-MAE pretrained',
           edgecolor=SURFACE, linewidth=2)
    ax.bar(xs + w/2 + .01, rnd, w, color=C_RND, label='Random init',
           edgecolor=SURFACE, linewidth=2)
    for xi, (a, b) in enumerate(zip(pre, rnd)):
        ax.annotate(f'{a:.3f}', (xi - w/2, a), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=7.5, color=TEXT_2)
        ax.annotate(f'{b:.3f}', (xi + w/2, b), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=7.5, color=TEXT_2)
    ax.set_xticks(xs); ax.set_xticklabels(LAB, fontsize=8.5)
    ax.set_ylim(0.6, 1.0)
    ax.set_title(title, fontsize=11.5, color=TEXT_1, loc='left', pad=10, fontweight='bold')
    ax.set_ylabel('Best validation F1', fontsize=10.5, color=TEXT_2)
    ax.set_xlabel('Fusion operator', fontsize=10.5, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)

leg = axes[0].legend(loc='upper left', frameon=False, fontsize=9.5)
for t in leg.get_texts():
    t.set_color(TEXT_2)

fig.suptitle('The fusion operator matters more than the encoder',
             fontsize=13.5, color=TEXT_1, x=.008, ha='left', y=1.0, fontweight='bold')
fig.text(.008, .945, 'Symmetric operators (|f1-f2|, f1*f2) cannot distinguish Forest->Urban '
                     'from Urban->Forest · 2000 pairs, 20 epochs, seed 42',
         fontsize=9, color=TEXT_2, ha='left')
plt.tight_layout(rect=[0, 0, 1, .935])
plt.savefig(OUT, facecolor=SURFACE, bbox_inches='tight')
print(f'saved -> {OUT}')
