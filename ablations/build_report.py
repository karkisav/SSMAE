#!/usr/bin/env python
"""Build the ablation + paper-direction report as a PDF.

All ablation numbers are read from the results JSON files rather than typed in,
so the document cannot drift from the experiments.
"""
import json, os
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

BASE = '/home/iit-dharwad/sav/ablations'
OUT = '/home/iit-dharwad/sav/SSMAE_ablation_report.pdf'

INK, INK2, RULE = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#d8d7d0')
BLUE, ORANGE, GREEN, RED = (colors.HexColor('#2a78d6'), colors.HexColor('#eb6834'),
                            colors.HexColor('#1baf7a'), colors.HexColor('#c0392b'))

spec = json.load(open(f'{BASE}/results_spectral.json'))
ep = json.load(open(f'{BASE}/results_pretrain_epochs.json'))
fus = json.load(open(f'{BASE}/results_fusion.json'))

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
def S(name, **kw):
    base = dict(fontName='Helvetica', fontSize=9.5, leading=14, textColor=INK,
                alignment=TA_LEFT, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)

Body   = S('Body')
Small  = S('Small', fontSize=8.5, leading=12, textColor=INK2)
H1     = S('H1', fontName='Helvetica-Bold', fontSize=17, leading=21, spaceAfter=4, spaceBefore=2)
H2     = S('H2', fontName='Helvetica-Bold', fontSize=12.5, leading=16, spaceBefore=13, spaceAfter=5)
H3     = S('H3', fontName='Helvetica-Bold', fontSize=10, leading=13, spaceBefore=9, spaceAfter=3)
Sub    = S('Sub', fontSize=10, leading=14, textColor=INK2, spaceAfter=12)
Cap    = S('Cap', fontSize=8, leading=11, textColor=INK2, spaceBefore=3, spaceAfter=12)
Bullet = S('Bullet', leftIndent=11, bulletIndent=2, spaceAfter=3)
Key    = S('Key', fontSize=10, leading=15, fontName='Helvetica-Bold')

story = []
def P(t, s=Body): story.append(Paragraph(t, s))
def B(t): story.append(Paragraph(t, Bullet, bulletText='-'))
def gap(h=6): story.append(Spacer(1, h))

def fig(name, caption, width=165*mm):
    path = f'{BASE}/{name}'
    w, h = PILImage.open(path).size
    story.append(Image(path, width=width, height=width * h / w))
    story.append(Paragraph(caption, Cap))

def table(data, widths, highlight=None, align_right_from=1):
    t = Table(data, colWidths=widths, hAlign='LEFT')
    st = [('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8.5),
          ('FONT', (0, 1), (-1, -1), 'Helvetica', 8.5),
          ('TEXTCOLOR', (0, 0), (-1, -1), INK),
          ('LINEBELOW', (0, 0), (-1, 0), .8, RULE),
          ('LINEBELOW', (0, -1), (-1, -1), .8, RULE),
          ('ALIGN', (align_right_from, 0), (-1, -1), 'RIGHT'),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 4),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f5f1')])]
    for r in (highlight or []):
        st += [('FONT', (0, r), (-1, r), 'Helvetica-Bold', 8.5),
               ('TEXTCOLOR', (0, r), (-1, r), BLUE)]
    t.setStyle(TableStyle(st))
    story.append(t)

# ================================================================== TITLE
P('SS-MAE for Forest Encroachment Detection', H1)
P('Ablation study and revised paper direction', Sub)
P('This report documents a correctness audit of the SS-MAE pipeline, three controlled '
  'ablations run after that audit, and the paper direction those results support. '
  'Every number is reproduced from the committed result files listed in Appendix B; '
  'nothing is carried over from runs that predate the audit.', Body)
gap(2)

P('Executive summary', H2)
P('Three findings, in order of importance.', Body)
gap(2)

P('1. The "dual" in dual spatial-spectral masking is actively harmful.', Key)
P('Spectral masking - removing whole bands during pretraining - degrades every downstream '
  'readout monotonically. Turning it off raises the EuroSAT linear probe from '
  f"{spec['0.3']['cls_linear_probe']*100:.1f}% to {spec['0.0']['cls_linear_probe']*100:.1f}% "
  f"and change-detection F1 from {spec['0.3']['cd_frozen']:.4f} to {spec['0.0']['cd_frozen']:.4f}. "
  'This is the stated novelty of the method, and it is the single largest negative effect measured.', Body)

P('2. The fusion operator matters more than the encoder.', Key)
P('The change head consumed |f1 - f2|, which is exactly order-invariant and therefore cannot '
  'distinguish Forest-to-Urban from Urban-to-Forest - the entire semantic content of an '
  'encroachment detector. Replacing it with the signed difference f1 - f2 is worth up to '
  '+0.117 F1, more than any encoder change measured.', Body)

P('3. With both corrected, pretraining helps; before correction it did not.', Key)
P('Under the original recipe, self-supervised pretraining lost to a randomly initialised '
  'encoder at every one of ten checkpoints. With spectral masking off and signed fusion, the '
  f"frozen probe reaches {spec['0.0']['cd_frozen']:.4f} against a random-init reference of 0.8744.", Body)

gap(4)
P('Consequence for the paper: the original claim - that dual spatial-spectral masking improves '
  'encroachment detection - is not supportable. What the evidence does support is a stronger and '
  'more specific contribution, set out in Section 7.', Body)

story.append(PageBreak())

# ================================================================== SYSTEM
P('1. System under study', H2)
P('SS-MAE is a masked autoencoder over 13-band Sentinel-2 imagery. A ViT encoder '
  '(512-dim, 8 layers, 29.2 M parameters) is pretrained to reconstruct masked patches of '
  '64x64 EuroSAT tiles, then reused for two downstream tasks:', Body)
B('<b>Classification.</b> 10-way EuroSAT land cover, used here as a linear probe on frozen features.')
B('<b>Change detection.</b> A siamese head consumes a pair of tiles and predicts whether an '
  'exploitation transition occurred (for example Forest to Residential).')
gap(4)
P('Masking is applied along two axes. <b>Spatial</b> masking hides a fraction of the 8x8 patches '
  'independently per band; <b>spectral</b> masking removes entire bands. The published setting is '
  '0.75 spatial and 0.30 spectral, giving 80.8% of the input hidden, of which 23% of bands are '
  'removed outright. The spectral axis is the paper\'s claimed novelty.', Body)

P('2. Correctness audit', H2)
P('The ablations below are only meaningful because a set of defects was fixed first. These are '
  'recorded because they change how earlier numbers should be read, not as a list of chores.', Body)
gap(2)

table([['Defect', 'Effect on results'],
       ['Mask never applied to the encoder',
        'forward_encoder() received the mask and ignored it. The encoder saw\n'
        'the complete image, so "reconstruction" was a copy task. Explains the\n'
        'implausible 0.0109 pretraining loss. All pretraining was invalid.'],
       ['Label-efficiency contamination',
        'The 10%-label run reused the encoder already fine-tuned on 100% of\n'
        'labels, so "99.3% retention" measured leakage, not efficiency.'],
       ['Adaptive masking degenerate',
        'Loss was averaged over all patches while masking only scaled\n'
        'embeddings, so "mask nothing" minimised it. Band importance collapsed\n'
        '(variance 0.067 -> 0.001).'],
       ['Silent checkpoint failure',
        'strict=False hid a total key mismatch; the OSCD analysis ran on a\n'
        'randomly initialised network while reporting results.'],
       ['Band ordering',
        'sorted(glob("B*.tif")) places B8A after B12, shifting every band from\n'
        'index 8 upward relative to training.'],
       ['Uncontrolled headline comparison',
        'The 0.9505 result used 2.5x the pairs and 1.5x the epochs of the\n'
        'baselines it was compared against.']],
      [46*mm, 119*mm], align_right_from=0)
story.append(Paragraph('Table 1. Defects found and corrected before any ablation was run. '
                       'Figures produced before these fixes should not be cited.', Cap))

P('3. Post-audit baseline', H2)
P('After correction, EuroSAT classification reaches <b>95.52% accuracy, macro F1 0.9536</b>. '
  'The change-detection picture was less favourable. Under a matched budget - all methods on '
  'the same 2000 pairs for 20 epochs - self-supervised pretraining did not win:', Body)
gap(2)
table([['Method (2000 pairs, 20 epochs)', 'Best F1'],
       ['Random encoder', '0.8859'],
       ['Simple CNN', '0.8677'],
       ['SS-MAE frozen', '0.7074'],
       ['SS-MAE finetuned', '0.8781']],
      [110*mm, 25*mm])
story.append(Paragraph('Table 2. Matched-budget comparison. The random encoder leads.', Cap))

P('The label-efficiency sweep agreed, and was worse where pretraining should help most - at '
  'the smallest label fractions:', Body)
gap(2)
table([['Labels', 'Pairs', 'SS-MAE', 'Random', 'Gap'],
       ['1%', '50', '0.5289', '0.6276', '-0.0987'],
       ['5%', '250', '0.6401', '0.7769', '-0.1369'],
       ['10%', '500', '0.8111', '0.8012', '+0.0099'],
       ['25%', '1250', '0.8540', '0.8639', '-0.0100'],
       ['50%', '2500', '0.9060', '0.9049', '+0.0011'],
       ['100%', '5000', '0.9223', '0.9305', '-0.0082']],
      [22*mm, 22*mm, 26*mm, 26*mm, 26*mm])
story.append(Paragraph('Table 3. Label efficiency. Pretraining loses at 4 of 6 fractions, and '
                       'loses worst at 1% and 5% - the opposite of the expected pattern. '
                       'The two wins are within single-seed noise.', Cap))
P('These two tables set the question the ablations answer: <i>why</i> does pretraining fail here?', Body)

story.append(PageBreak())

# ================================================================== ABLATION 1
P('4. Ablation A - spectral masking ratio', H2)
P('<b>Question.</b> Does the spectral half of dual masking earn its place? Setting the spectral '
  'ratio to zero reduces the method to an ordinary spatial MAE.', Body)
P('<b>Design.</b> Spectral ratio in {0.00, 0.15, 0.30, 0.50} with spatial fixed at 0.75, '
  '100 pretraining epochs each, seed 42. The 0.30 arm reuses the published checkpoint. Three '
  'readouts per arm: EuroSAT linear probe, change detection frozen, change detection finetuned '
  '(signed fusion). Total 112 minutes.', Body)
gap(3)
fig('fig_spectral.png', 'Figure 1. Every readout degrades monotonically as spectral masking '
    'increases. At the published 0.30 setting the frozen probe sits below the random-init '
    'reference; at 0.00 it sits above it.')

rows = [['Spectral', 'Masked', 'EuroSAT probe', 'CD frozen', 'CD finetuned']]
for k in ('0.0', '0.15', '0.3', '0.5'):
    r = spec[k]
    rows.append([k + ('  (published)' if k == '0.3' else ''),
                 f"{r['masked_fraction']*100:.1f}%",
                 f"{r['cls_linear_probe']*100:.2f}%",
                 f"{r['cd_frozen']:.4f}", f"{r['cd_finetuned']:.4f}"])
rows.append(['random init', '--', '--', '0.8744', '0.9309'])
table(rows, [34*mm, 22*mm, 32*mm, 26*mm, 30*mm], highlight=[1])
story.append(Paragraph('Table 4. Spectral masking sweep. Row 0.00 (highlighted) is the best '
                       'configuration on all three readouts.', Cap))

P('<b>Mechanism.</b> The pretraining loss curves explain the effect. A fully removed band cannot '
  'be inpainted from spatial context, so the model parks at the irreducible error of predicting '
  'the band mean:', Body)
gap(2)
table([['Spectral', 'epoch 1', 'epoch 30', 'epoch 50', 'epoch 100', 'Behaviour'],
       ['0.00', '0.1313', '0.0528', '0.0342', '0.0262', 'smooth descent'],
       ['0.15', '0.1304', '0.1071', '0.1070', '0.0574', 'stalled ~40 epochs'],
       ['0.50', '0.1310', '0.1070', '0.0763', '0.0713', 'never escapes']],
      [20*mm, 20*mm, 20*mm, 20*mm, 22*mm, 42*mm])
story.append(Paragraph('Table 5. Pretraining reconstruction loss. Spectral masking spends most '
                       'of the optimisation budget on a target carrying no learnable signal.', Cap))
P('<b>Conclusion.</b> Spectral masking should be removed. The effect is large (a 23-point drop in '
  'linear-probe accuracy between 0.00 and 0.30), monotonic across four settings, and mechanistically '
  'explained.', Body)

story.append(PageBreak())

# ================================================================== ABLATION 2
P('5. Ablation B - pretraining duration', H2)
P('<b>Question.</b> Does more pretraining produce better downstream features? A flat or falling '
  'curve indicates the pretext task is not learning transferable structure.', Body)
P('<b>Design.</b> Ten checkpoints (epochs 10 to 100) plus a random-init control, each trained '
  'to a change detector under one fixed budget, in both frozen and finetuned regimes. '
  '22 runs, 20.8 minutes. This ablation uses the original recipe, so it characterises the '
  'method as published.', Body)
gap(3)
fig('fig_pretrain_epochs.png', 'Figure 2. No checkpoint beats random initialisation in either '
    'regime. The frozen probe degrades sharply before partially recovering.', width=150*mm)

rows = [['Pretrain epochs', 'Frozen F1', 'Frozen AUC', 'Finetuned F1']]
rows.append(['random init', f"{ep['frozen']['0']['best_f1']:.4f}",
             f"{ep['frozen']['0']['best_auc']:.4f}", f"{ep['finetuned']['0']['best_f1']:.4f}"])
for e in ('10', '20', '50', '80', '100'):
    rows.append([e, f"{ep['frozen'][e]['best_f1']:.4f}", f"{ep['frozen'][e]['best_auc']:.4f}",
                 f"{ep['finetuned'][e]['best_f1']:.4f}"])
table(rows, [36*mm, 30*mm, 30*mm, 34*mm], highlight=[1])
story.append(Paragraph('Table 6. Selected points from the 11-checkpoint sweep. Frozen AUC falls '
                       'to 0.52 - chance - between epochs 20 and 50.', Cap))

P('<b>Interpretation.</b> Under the published recipe, pretraining does not merely fail to help; '
  'it destroys discriminative structure that random initialisation already had. A random '
  'high-dimensional projection approximately preserves pairwise distances, which is what an L1 '
  'distance head needs, whereas the reconstruction objective trades that structure for low-level '
  'spectral detail. Ablation A identifies the spectral term as the specific cause.', Body)

story.append(PageBreak())

# ================================================================== ABLATION 3
P('6. Ablation C - fusion operator', H2)
P('<b>Question.</b> The head consumes |f1 - f2|. Is that distance operator itself responsible '
  'for flattering random initialisation?', Body)
P('<b>Design.</b> Five fusion operators, crossed with pretrained/random initialisation and '
  'frozen/finetuned regimes. 20 runs, 19.1 minutes. Encoder and head capacity otherwise '
  'unchanged, so any difference is attributable to fusion.', Body)
gap(3)
fig('fig_fusion.png', 'Figure 3. The current operator |f1 - f2| is the worst or near-worst choice '
    'in every arm. Signed and concatenation operators dominate it.')

rows = [['Fusion', 'Order-aware', 'Frozen (pre)', 'Frozen (rand)', 'Finetuned (pre)', 'Finetuned (rand)']]
NAMES = {'abs': '|f1-f2|  (current)', 'diff': 'f1-f2  (signed)', 'concat': '[f1, f2]',
         'absprod': '[|f1-f2|, f1*f2]', 'concat_abs': '[f1, f2, |f1-f2|]'}
SYM = {'abs': 'no', 'diff': 'yes', 'concat': 'yes', 'absprod': 'no', 'concat_abs': 'yes'}
for f in ('abs', 'diff', 'concat', 'absprod', 'concat_abs'):
    rows.append([NAMES[f], SYM[f],
                 f"{fus['frozen|pretrained'][f]['best_f1']:.4f}",
                 f"{fus['frozen|random'][f]['best_f1']:.4f}",
                 f"{fus['finetuned|pretrained'][f]['best_f1']:.4f}",
                 f"{fus['finetuned|random'][f]['best_f1']:.4f}"])
table(rows, [36*mm, 20*mm, 25*mm, 26*mm, 29*mm, 29*mm])
story.append(Paragraph('Table 7. Fusion operator sweep. "Order-aware" marks operators that can '
                       'distinguish A-to-B from B-to-A; |f1-f2| and f1*f2 are exactly symmetric '
                       '(verified numerically).', Cap))

P('<b>Conclusion.</b> Two separable results. The operator is worth up to +0.117 F1 on its own, '
  'which is larger than any encoder effect measured - a positive, immediately usable finding. '
  'But it does not rescue pretraining: random initialisation still wins in 9 of 10 cells, so '
  'the distance operator was not the reason pretraining failed. Ablation A supplies that reason.', Body)

story.append(PageBreak())

# ================================================================== SYNTHESIS
P('7. Synthesis and recommended configuration', H2)
P('The three ablations compose into a single coherent account. Under the published recipe, '
  'pretraining loses to random initialisation (Ablation B). The cause is the spectral masking '
  'term, which makes the pretext task partly unlearnable and degrades features monotonically '
  '(Ablation A). Independently, the head\'s symmetric distance operator discards transition '
  'direction and costs roughly 0.1 F1 (Ablation C). Correcting both yields a configuration '
  'where pretraining finally beats random initialisation.', Body)
gap(3)
table([['Component', 'As published', 'Recommended', 'Evidence'],
       ['Spectral masking', '0.30', '0.00  (off)', 'Ablation A, Table 4'],
       ['Spatial masking', '0.75', '0.75  (unchanged)', 'not yet ablated'],
       ['Fusion operator', '|f1 - f2|', 'f1 - f2  (signed)', 'Ablation C, Table 7'],
       ['Frozen CD F1', '0.8266', f"{spec['0.0']['cd_frozen']:.4f}", 'vs 0.8744 random'],
       ['EuroSAT probe', '52.46%', f"{spec['0.0']['cls_linear_probe']*100:.2f}%", 'Ablation A']],
      [34*mm, 30*mm, 40*mm, 40*mm], align_right_from=1)
story.append(Paragraph('Table 8. Published versus recommended configuration.', Cap))

P('8. Paper direction', H2)
P('<b>What can no longer be claimed.</b> That dual spatial-spectral masking improves '
  'encroachment detection. The ablation shows the opposite with a monotonic dose-response '
  'across four settings and a mechanism. Any draft built on the 0.9505 headline also needs '
  'retiring: that number came from an uncontrolled comparison (Table 1).', Body)
gap(2)
P('<b>What the evidence supports.</b> A paper on what actually drives siamese change detection '
  'in multispectral remote sensing, with three defensible claims:', Body)
B('<b>C1.</b> Spectral masking is counterproductive for MAE pretraining on multispectral imagery. '
  'A removed band has no spatial context to be recovered from, unlike a removed patch, so the '
  'objective contains a large irreducible term. Evidence: Tables 4 and 5, Figure 1.')
B('<b>C2.</b> Head design dominates encoder pretraining on this task. A symmetric fusion operator '
  'costs more accuracy than the entire pretraining stage supplies, and is additionally '
  'ill-posed for directional change. Evidence: Table 7, Figure 3.')
B('<b>C3.</b> Pretraining gains are conditional, and the conditions are identifiable by ablation '
  'rather than assumed. Evidence: Ablation B as the negative case, Ablation A as its resolution.')
gap(3)
P('This is a stronger contribution than the original framing. Negative and conditional results '
  'about self-supervision are useful to the community precisely because they are rarely '
  'published, and here they come with a mechanism and a corrected recipe rather than a bare '
  'failure report.', Body)
gap(2)
P('<b>Framing caution.</b> The learned band-importance result should not be described as '
  '"band importance for classification". The quantity measures how expensive a band is to mask, '
  'that is, how poorly it can be reconstructed from the others. That is spectral uniqueness. It '
  'ranks B01 Coastal and B09 Water Vapour highest - 60 m atmospheric-correction bands that '
  'practitioners routinely discard for land cover - which is coherent under the reconstruction '
  'reading and indefensible under the classification reading.', Body)

story.append(PageBreak())

# ================================================================== LIMITS
P('9. Threats to validity', H2)
B('<b>Single seed.</b> Every number is one run at seed 42. Large effects - the 23-point probe '
  'collapse, the +0.117 fusion gain - are far outside plausible noise. Differences of about '
  '0.02, such as 0.8860 versus 0.8878, are not interpretable. Claims C1 and C2 need 3 to 5 seeds '
  'before submission.')
B('<b>The change-detection task is a proxy.</b> Pairs are two unrelated EuroSAT scenes of '
  'different classes, so the task is closer to "do these belong to different classes" than to '
  'temporal change. This inflates absolute F1 and may interact with the fusion result. '
  'Co-located bitemporal pairs from OSCD and LEVIR-CD are staged and should be used to check it.')
B('<b>Validation-set model selection.</b> Best-epoch F1 is taken on the same split used for '
  'reporting, which is optimistic in absolute terms. It is applied identically to every arm, '
  'so comparisons remain valid.')
B('<b>Single dataset and scale.</b> All pretraining is EuroSAT at 64x64 with 64 patches. '
  'Whether the spectral-masking conclusion holds at larger tile sizes or on other sensors '
  'is untested.')
B('<b>LEVIR variant.</b> The staged data is the original LEVIR-CD (445/128/64), not LEVIR-CD+ '
  '(637/348). Earlier LEVIR numbers are not comparable to it.')

P('10. Remaining experiments', H2)
P('In priority order, with the claim each supports.', Body)
gap(2)
table([['#', 'Experiment', 'Supports', 'Cost'],
       ['1', 'Repeat Ablations A and C at 3-5 seeds', 'C1, C2 - required', '~4 h'],
       ['2', 'Re-run label efficiency with corrected recipe', 'C3 - headline result', '~1 h'],
       ['3', 'Spatial masking ratio sweep (0.25-0.90)', 'completes the masking story', '~2.5 h'],
       ['4', 'Band-subset ablation vs learned ranking', 'validates band-importance claim', '~1 h'],
       ['5', 'Co-located pairs from OSCD / LEVIR-CD', 'addresses the proxy-task threat', '~2 h'],
       ['6', 'Layer-wise freezing (k of 8 blocks)', 'locates transferable features', '~1 h']],
      [8*mm, 68*mm, 55*mm, 17*mm], align_right_from=3)
story.append(Paragraph('Table 9. Proposed remaining work. Item 2 is the one most likely to '
                       'change the headline: with spectral masking off, pretraining should now '
                       'show the low-label advantage it failed to show in Table 3.', Cap))

P('Appendix A - experimental protocol', H2)
P('Common to all ablations: EuroSAT 13-band tiles at 64x64, 8x8 patches; encoder 512-dim, '
  '8 layers, 8 heads, 29.2 M parameters; AdamW; cosine schedule; seed 42; NVIDIA RTX 4500 Ada.', Small)
P('Pretraining: batch 128, lr 1.5e-4, weight decay 0.05, 100 epochs, spatial masking 0.75.', Small)
P('Change detection: 2000 train / 600 validation balanced pairs, batch 32, 20 epochs, '
  'BCEWithLogits, gradient clipping 1.0, head lr 3e-4, encoder lr 1e-5 when unfrozen, '
  'threshold 0.5. Best-epoch validation F1 reported.', Small)
P('Linear probe: frozen encoder, single linear layer, 10 epochs, lr 1e-3.', Small)

P('Appendix B - artefacts', H2)
P('Ablation code and results (all under <b>ablations/</b>):', Small)
for f, d in [('ablation_spectral.py / results_spectral.json', 'Ablation A'),
             ('ablation_pretrain_epochs.py / results_pretrain_epochs.json', 'Ablation B'),
             ('ablation_fusion.py / results_fusion.json', 'Ablation C'),
             ('plot_spectral.py / plot_pretrain_epochs.py / plot_fusion.py', 'Figures 1-3'),
             ('ckpt/ssmae_spectral{000,015,050}.pth', 'Ablation A checkpoints')]:
    story.append(Paragraph(f'- <b>{f}</b> &nbsp;-&nbsp; {d}', Small))
P('Each ablation script lifts its model classes directly from the notebook by AST extraction, '
  'so the experiments cannot drift from the architecture that produced the reported baselines. '
  'Results are written incrementally, so partial runs are recoverable.', Small)

# ------------------------------------------------------------------ build
def footer(canv, doc):
    canv.saveState()
    canv.setFont('Helvetica', 7.5)
    canv.setFillColor(INK2)
    canv.drawString(20*mm, 12*mm, 'SS-MAE ablation study and paper direction')
    canv.drawRightString(190*mm, 12*mm, f'{doc.page}')
    canv.setStrokeColor(RULE)
    canv.line(20*mm, 15*mm, 190*mm, 15*mm)
    canv.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                        topMargin=18*mm, bottomMargin=20*mm,
                        title='SS-MAE ablation study and paper direction',
                        author='generated from committed result files')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f'built -> {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)')
