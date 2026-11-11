"""Build final A2 test tables, figures, qualitative galleries and an evidence report.

    python scripts/a2_error_analysis.py

Consumes completed evaluator exports; never trains or selects a run by test score.
Groups are operational diagnostics and can overlap. Visual causes remain hypotheses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'dla261-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml
from src.a2.data import build_transforms

NAMES = {'cnn': 'CNN baseline', 'resnet50_frozen': 'ResNet-50 frozen', 'resnet50_full': 'ResNet-50 full'}
COLORS = {'cnn': '#8a94a6', 'resnet50_frozen': '#3976be', 'resnet50_full': '#159c8c'}
SHAPE_PAIR = {'Banh chung', 'Banh tet'}
SOUP_GROUP = {'Bun bo Hue', 'Bun rieu', 'Banh canh'}
EXTENDED_SOUP_GROUP = SOUP_GROUP | {'Hu tieu', 'Pho'}
WHITE_CAKE_GROUP = {'Banh duc', 'Banh cuon', 'Banh gio', 'Banh bot loc'}
FIGURES = ROOT/'results/a2/figures/test'
TABLES = ROOT/'results/a2/tables'
ANALYSIS = ROOT/'results/a2/error_analysis'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path, expected, classes):
    rows = read_csv(path)
    if len(rows) != len(expected):
        raise ValueError(f'Incomplete predictions: {path}')
    for index, (row, item) in enumerate(zip(rows, expected)):
        for key in ['index', 'true_label', 'pred_label', 'width', 'height']:
            row[key] = int(row[key])
        for key in ['confidence', 'true_class_probability', 'confidence_margin', 'aspect_ratio']:
            row[key] = float(row[key])
        row['correct'] = bool(int(row['correct']))
        if (row['index'] != index or row['path'] != item['path'] or row['md5'] != item['md5']
                or row['true_label'] != item['label'] or row['true_class'] != classes[item['label']]
                or row['pred_class'] != classes[row['pred_label']]
                or row['correct'] != (row['true_label'] == row['pred_label'])):
            raise ValueError(f'Prediction/split alignment mismatch: {path}, row {index}')
    return rows


def confusion_from_rows(rows, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for row in rows:
        matrix[row['true_label'], row['pred_label']] += 1
    return matrix


def wilson(errors, count):
    if not count:
        return None, None
    z = 1.96
    p = errors/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0, center-half), min(1, center+half)


def group_stat(model, name, rows):
    n = len(rows)
    errors = sum(not row['correct'] for row in rows)
    low, high = wilson(errors, n)
    return dict(model=model, group=name, support=n, errors=errors,
                error_rate=errors/n if n else None, accuracy=1-errors/n if n else None,
                error_rate_wilson_low=low, error_rate_wilson_high=high,
                mean_confidence=statistics.mean(row['confidence'] for row in rows) if rows else None)


def top_confusions(model, matrix, classes):
    total_errors = int(matrix.sum()-matrix.trace())
    result = []
    for i, true in enumerate(classes):
        for j, pred in enumerate(classes):
            if i != j and matrix[i, j]:
                count = int(matrix[i, j]); support = int(matrix[i].sum())
                result.append(dict(model=model, true_class=true, predicted_class=pred,
                                   count=count, true_class_support=support,
                                   rate_within_true_class=count/support,
                                   share_of_all_errors=count/total_errors if total_errors else 0))
    return sorted(result, key=lambda row: (-row['count'], row['true_class'], row['predicted_class']))


def diverse(rows, key, count=9):
    ordered = sorted(rows, key=key)
    chosen, used = [], set()
    for row in ordered:
        if row['true_class'] not in used:
            chosen.append(row);used.add(row['true_class'])
        if len(chosen) == count:
            return chosen
    chosen_paths = {row['path'] for row in chosen}
    return chosen + [row for row in ordered if row['path'] not in chosen_paths][:count-len(chosen)]


def stable_key(row):
    return hashlib.sha256(('a2-qualitative-42:'+row['path']).encode()).hexdigest()


def select_examples(rows, frozen_rows):
    wrong = [row for row in rows if not row['correct']]
    pair_counts = Counter((row['true_class'], row['pred_class']) for row in wrong)
    return {
        'correct': diverse([row for row in rows if row['correct'] and row['confidence_margin'] >= .5], stable_key),
        'hard_correct': diverse([row for row in rows if row['correct']], lambda row:(row['confidence_margin'], row['path'])),
        'wrong': diverse(wrong, lambda row:(-pair_counts[row['true_class'], row['pred_class']], -row['confidence'], row['path'])),
        'high_confidence_wrong': diverse([row for row in wrong if row['confidence'] >= .9], lambda row:(-row['confidence'], row['path'])),
        'improvements': diverse([row for row, other in zip(rows, frozen_rows) if row['correct'] and not other['correct']], stable_key),
        'regressions': diverse([row for row, other in zip(rows, frozen_rows) if not row['correct'] and other['correct']], stable_key),
    }


def input_image(data_root, row, config):
    with Image.open(data_root/row['path']) as original:
        original.draft('RGB', (320, 320))
        image = original.convert('RGB')
    model_config = config['models'][config['model']]
    # These are exactly the saved evaluation Resize and CenterCrop, before tensor conversion.
    for transform in build_transforms(config, model_config['normalize'], train=False).transforms[:2]:
        image = transform(image)
    return image


def save_figure(fig, name):
    fig.savefig(FIGURES/name, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_metrics(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
        for i, (model, values) in enumerate(summary.items()):
            mean = values[metric+'_mean']; std = values[metric+'_sample_std']
            ax.bar(i, mean, color=COLORS[model], width=.58,
                   yerr=std if std is not None else None, capsize=5)
            ax.text(i, mean+.027, f'{mean:.4f}', ha='center', fontsize=10)
        ax.set_xticks(range(len(summary)), ['CNN\n1 seed', 'Frozen\n3 seeds', 'Full\n3 seeds'])
        ax.set_ylim(0, 1);ax.set_ylabel(metric.replace('_', '-'));ax.grid(axis='y', alpha=.15)
    fig.suptitle('Final test results | 4,984 images | error bars: sample SD across seeds')
    save_figure(fig, 'a2_test_model_comparison.png')


def plot_confusion(model, matrix, classes):
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros(matrix.shape), where=matrix.sum(axis=1, keepdims=True)>0)
    fig, ax = plt.subplots(figsize=(13, 11), layout='constrained')
    image = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=65, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j or normalized[i, j] >= .05:
                ax.text(j, i, f'{normalized[i,j]*100:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if normalized[i,j]>.5 else '#172033')
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    ax.set_title(NAMES[model]+' | row-normalized test confusion (%)\nDiagonal and off-diagonal cells >=5% annotated')
    fig.colorbar(image, ax=ax, shrink=.7, label='Fraction of each true class')
    save_figure(fig, f'a2_test_confusion_{model}.png')


def plot_per_class(classes, matrices):
    fig, ax = plt.subplots(figsize=(11, 12), layout='constrained')
    positions = np.arange(len(classes))
    for offset, (model, matrix) in zip([-.24, 0, .24], matrices.items()):
        denominator = matrix.sum(axis=0)+matrix.sum(axis=1)
        values = np.divide(2*np.diag(matrix), denominator, out=np.zeros(len(classes)), where=denominator>0)
        ax.barh(positions+offset, values, height=.22, color=COLORS[model], label=NAMES[model])
    ax.set_yticks(positions, classes);ax.invert_yaxis();ax.set_xlim(0, 1)
    ax.set_xlabel('Per-class F1');ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower right');ax.set_title('Test per-class F1 | runs selected by validation, before test')
    save_figure(fig, 'a2_test_per_class_f1.png')


def gallery(category, rows, data_root, config, others):
    fig, axes = plt.subplots(3, 3, figsize=(12, 16), layout='constrained')
    fig.get_layout_engine().set(h_pad=.12, w_pad=.06, hspace=.08, wspace=.04)
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, rows):
        ax.imshow(input_image(data_root, row, config))
        frozen = others['resnet50_frozen'][row['index']]
        cnn = others['cnn'][row['index']]
        ax.set_title(f"#{row['index']} | True: {row['true_class']}\n"
                     f"Full: {row['pred_class']} ({row['confidence']:.1%})\n"
                     f"Frozen: {frozen['pred_class']} | CNN: {cnn['pred_class']}", fontsize=8.5, pad=10)
    labels = {'correct':'Correct, margin >=0.5', 'hard_correct':'Correct, smallest probability margins',
              'wrong':'Failures from frequent confusion pairs', 'high_confidence_wrong':'Failures, confidence >=90%',
              'improvements':'Full correct, frozen wrong', 'regressions':'Frozen correct, full wrong'}
    fig.suptitle(labels[category]+'\nExact model input crops; representative full run selected by validation', fontsize=13)
    save_figure(fig, f'a2_test_examples_{category}.png')


def crop_gallery(rows, data_root, config):
    selected = diverse([row for row in rows if (row['aspect_ratio']>2 or row['aspect_ratio']<.5) and not row['correct']], stable_key, 3)
    if not selected:
        fig,ax=plt.subplots(figsize=(10,3));ax.text(.5,.5,'No extreme-aspect failures in this run',ha='center');ax.axis('off')
        save_figure(fig,'a2_test_crop_cases.png')
        return []
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 4.5*len(selected)), squeeze=False, layout='constrained')
    for axes_row, row in zip(axes, selected):
        with Image.open(data_root/row['path']) as original:
            image = original.convert('RGB');image.thumbnail((700,700))
        axes_row[0].imshow(image);axes_row[0].set_title(f"#{row['index']} | original {row['width']}x{row['height']}")
        axes_row[1].imshow(input_image(data_root, row, config))
        axes_row[1].set_title(f"True: {row['true_class']}\nFull: {row['pred_class']} ({row['confidence']:.1%})")
        for ax in axes_row:
            ax.axis('off')
    fig.suptitle('Extreme-aspect errors | original versus actual evaluation crop')
    save_figure(fig, 'a2_test_crop_cases.png')
    return selected


def image_features(rows, data_root):
    features = []
    for index, row in enumerate(rows):
        with Image.open(data_root/row['path']) as source:
            source.draft('RGB', (128,128))
            pixels = np.asarray(source.convert('RGB').resize((64,64)), dtype=float)/255
        near_white = (pixels.min(axis=2)>=220/255) & ((pixels.max(axis=2)-pixels.min(axis=2))<=30/255)
        border = np.ones((64,64), dtype=bool);border[13:-13,13:-13] = False
        features.append(dict(path=row['path'], mean_rgb_brightness=float(pixels.mean()),
                             near_white_border_fraction=float(near_white[border].mean())))
        if (index+1)%1000 == 0:
            print('IMAGE_FEATURES', index+1, '/', len(rows), flush=True)
    return features


def error_groups(model, rows, features):
    predicates = {
        'all': lambda r,f:True,
        'extreme_aspect': lambda r,f:r['aspect_ratio']>2 or r['aspect_ratio']<.5,
        'normal_aspect': lambda r,f:.5<=r['aspect_ratio']<=2,
        'short_side_lt224': lambda r,f:min(r['width'],r['height'])<224,
        'short_side_ge224': lambda r,f:min(r['width'],r['height'])>=224,
        'dark_mean_lt025': lambda r,f:f['mean_rgb_brightness']<.25,
        'bright_mean_gt075': lambda r,f:f['mean_rgb_brightness']>.75,
        'confidence_ge09': lambda r,f:r['confidence']>=.9,
        'margin_le01': lambda r,f:r['confidence_margin']<=.1,
        'true_shape_pair': lambda r,f:r['true_class'] in SHAPE_PAIR,
        'true_soup_group': lambda r,f:r['true_class'] in SOUP_GROUP,
        'extended_soup_group_exploratory': lambda r,f:r['true_class'] in EXTENDED_SOUP_GROUP,
        'white_cake_family_exploratory': lambda r,f:r['true_class'] in WHITE_CAKE_GROUP,
        'pia_white_border_proxy': lambda r,f:r['true_class']=='Banh pia' and f['near_white_border_fraction']>=.5,
        'pia_other_border': lambda r,f:r['true_class']=='Banh pia' and f['near_white_border_fraction']<.5,
    }
    return [group_stat(model, name, [r for r,f in zip(rows,features) if predicate(r,f)])
            for name,predicate in predicates.items()]


def within_group_confusions(rows, members):
    subset = [row for row in rows if row['true_class'] in members]
    count = sum(not row['correct'] and row['pred_class'] in members for row in subset)
    return dict(classes=sorted(members), support=len(subset), within_group_confusions=count,
                confusion_rate=count/len(subset) if subset else None,
                total_errors=sum(not row['correct'] for row in subset))


def class_adjusted_extreme(rows):
    extreme = [row for row in rows if row['aspect_ratio']>2 or row['aspect_ratio']<.5]
    normal = [row for row in rows if .5<=row['aspect_ratio']<=2]
    total = 0; covered = 0
    for label in {row['true_label'] for row in extreme}:
        n = sum(row['true_label']==label for row in extreme)
        references = [row for row in normal if row['true_label']==label]
        if references:
            total += n*sum(not row['correct'] for row in references)/len(references);covered += n
    return dict(extreme_support=len(extreme), observed_errors=sum(not row['correct'] for row in extreme),
                covered_support=covered, expected_errors_using_same_class_normal_rates=total,
                note='Descriptive adjustment, not a causal test of center cropping')


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])


def write_report(summary, representatives, selected_rows, matrices, classes, groups, pairs, hypotheses):
    report = ROOT/'reports/a2/test_evaluation_analysis.md'
    protocol=json.loads((TABLES/'a2_test_protocol.json').read_text())
    sample_metrics=json.loads((ROOT/'results/a2/runs'/representatives['resnet50_full']['run_id']/'evaluation/test/metrics.json').read_text())
    def image(name):
        return f'![{name}](../../results/a2/figures/test/{name})'
    lines = ['# A2 — Final test evaluation and error analysis', '',
             'These results use fixed saved checkpoints on the cleaned test split: **4,984 images, 30 classes**. '
             'All seven checkpoints were evaluated. No training or hyperparameter selection used these test results.', '',
             '## Protocol and run selection', '',
             'Illustrative runs were fixed using the highest existing validation macro-F1 in each family, before test evaluation. '
             'All-seed means remain the main comparison; illustrative runs are not selected by test performance.', '',
             markdown_table(['Family','Illustrative run','Validation macro-F1'],
                            [[NAMES[m],v['run_id'],f"{v['validation_macro_f1']:.4f}"] for m,v in representatives.items()]), '',
             f"Inference: {sample_metrics['device']}, PyTorch {sample_metrics['env']['torch']}, batch {protocol['batch_size']}, zero loader workers. Normalization and transforms come from "
             'each run config. CNN uses train statistics; ResNet uses ImageNet statistics. Each checkpoint is selected '
             'by validation macro-F1. Test predictions, checkpoint/config/split hashes and the device are saved per run.', '',
             '## Quantitative comparison', '',
             markdown_table(['Family','Runs','Accuracy','Macro-F1','Top-3 accuracy'],
               [[NAMES[m],s['runs'],f"{s['accuracy_mean']:.4f}"+(f" ± {s['accuracy_sample_std']:.4f}" if s['runs']>1 else ''),
                 f"{s['macro_f1_mean']:.4f}"+(f" ± {s['macro_f1_sample_std']:.4f}" if s['runs']>1 else ''),
                 f"{s['top3_accuracy_mean']:.4f}"] for m,s in summary.items()]), '',
             '± denotes sample standard deviation over seeds, not a confidence interval. CNN has one draft run only.', '',
             image('a2_test_model_comparison.png'), '',
             '### Paired frozen/full comparison', '',
             markdown_table(['Seed','Frozen F1','Full F1','Full − frozen'],
                            [[p['seed'],f"{p['frozen_f1']:.4f}",f"{p['full_f1']:.4f}",f"{p['gain']:+.4f}"] for p in pairs]), '',
             f"Mean paired gain: **{statistics.mean(p['gain'] for p in pairs):+.4f} macro-F1**. "
             'Interpret consistency over these three seeds descriptively; no formal significance claim is made. '
             'This compares a frozen backbone (including fixed BatchNorm statistics) with full adaptation, not a learning-rate-only intervention.', '',
             'CNN versus ResNet also changes architecture/capacity and initialization, so that comparison does not isolate pretraining alone.', '',
             '## Per-class behavior and confusions', '', image('a2_test_per_class_f1.png'), '']
    for model,matrix in matrices.items():
        top = top_confusions(model,matrix,classes)[:8]
        lines += ['### '+NAMES[model], '', image(f'a2_test_confusion_{model}.png'), '',
                  markdown_table(['True → predicted','Errors / true support','Rate within true class'],
                    [[r['true_class']+' → '+r['predicted_class'],f"{r['count']} / {r['true_class_support']}",f"{r['rate_within_true_class']:.1%}"] for r in top]), '']
    compute=read_csv(TABLES/'a2_test_compute_cost.csv')
    lines += ['## Compute cost', '',markdown_table(
        ['Family','Total / trainable parameters','Training minutes / run','Peak training GPU GiB','Forward ms / image'],
        [[NAMES[r['model']],f"{int(r['parameters']):,} / {int(r['trainable_parameters']):,}",
          f"{float(r['training_minutes_mean']):.2f}",r['peak_training_gpu_gib'] or 'not recorded',
          f"{float(r['forward_ms_per_image_mean']):.3f}"] for r in compute]), '',
        'Training was measured on RTX 4060; current forward timing is on Apple MPS. Frozen and full use the same '
        'inference architecture and have similar measured forward costs. Small timing differences in a single pass '
        'do not establish a speed advantage. Training time includes validation; forward time excludes preprocessing and transfers.', '']
    primary = [g for g in groups if g['model']=='resnet50_full']
    by_group={g['group']:g for g in primary}
    agreement=read_csv(TABLES/'a2_test_prediction_agreement.csv')
    lines += ['## Paired illustrative predictions', '',markdown_table(['Outcome','Images','Fraction'],
        [[r['outcome'],r['count'],f"{float(r['fraction']):.1%}"] for r in agreement]), '',
        'These paired counts refer to the validation-selected illustrative frozen/full runs, not an ensemble. '
        'Both improvements and regressions appear in the galleries.', '']
    lines += ['## Error groups', '',
              f"The following group analysis concerns **{representatives['resnet50_full']['run_id']}**, with "
              f"**{by_group['all']['errors']} errors / {by_group['all']['support']} test images**. "
              'Group-level seed variability is not estimated.', '',
              'Groups below are overlapping diagnostics, not mutually exclusive semantic causes. '
              'Error rates use each group\'s own support, never raw error counts alone. Wilson intervals are descriptive '
              'binomial intervals within one selected run and do not represent seed variability.', '',
              markdown_table(['Group','Images','Errors','Error rate','95% Wilson interval'],
                 [[r['group'],r['support'],r['errors'],f"{r['error_rate']:.1%}" if r['support'] else 'n/a',
                   f"[{r['error_rate_wilson_low']:.1%}, {r['error_rate_wilson_high']:.1%}]" if r['support'] else 'n/a'] for r in primary]), '',
              'Operational thresholds: extreme aspect ratio >2 or <0.5; short side <224 px; brightness is mean RGB '
              'on a 64×64 thumbnail (<0.25 dark, >0.75 bright); confidence ≥0.9; top-1/top-2 probability margin ≤0.1. '
              'These thresholds describe test behavior and were not used to tune the classifier. '
              'The extended soup group adds Hu tieu and Pho to H2 after observing confusions; the exploratory white-cake '
              'family consists of Banh duc, Banh cuon, Banh gio and Banh bot loc. These are post-hoc descriptions.', '',
              '### Error interpretation and possible improvements', '',
              markdown_table(['Diagnostic group','Measured frequency','Example IDs','Hypothesized cause','Future improvement idea'],[
                  ['Extended soup family',f"{by_group['extended_soup_group_exploratory']['errors']} errors / {by_group['extended_soup_group_exploratory']['support']} true images",'568, 2973, 4289',
                   'Shared broth/toppings; discriminative noodle shape can be small or hidden',
                   'Review training labels and add varied, clearly identified examples of confusing dishes'],
                  ['Banh chung / Banh tet',f"{hypotheses['H1']['within_group_confusions']} within-pair confusions / {hypotheses['H1']['support']} true images",'650, 678',
                   'Wrapped or close-up views can hide whole-item geometry; some labels need expert review',
                   'Audit ambiguous training labels and include whole-item plus cut-section views'],
                  ['White-cake family',f"{by_group['white_cake_family_exploratory']['errors']} errors / {by_group['white_cake_family_exploratory']['support']} true images",'96, 978, 191',
                   'Pale wrappers and minced toppings overlap; a few illustrated labels are ambiguous',
                   'Review labels without automatically relabeling test images; collect clearer shape cues in training'],
                  ['High-confidence predictions',f"{by_group['confidence_ge09']['errors']} errors / {by_group['confidence_ge09']['support']} predictions >=0.9",'568, 96, 650',
                   'Confidence can remain high for visually similar classes or possible annotation ambiguity',
                   'Assess calibration and any review threshold on validation, not on this final test'],
                  ['Extreme-aspect inputs',f"{by_group['extreme_aspect']['errors']} errors / {by_group['extreme_aspect']['support']} images",'191, 3452, 442',
                   'Center crop removes context in individual examples; no elevated group error rate is established',
                   'A future validation-controlled resize/letterbox comparison can test crop effects'],
                  ['Low-margin predictions',f"{by_group['margin_le01']['errors']} errors / {by_group['margin_le01']['support']} margin <=0.1 cases",'4165, 4812, 3170',
                   'Several competing labels receive similar scores; blur, occlusion or atypical framing may contribute',
                   'Inspect ambiguous training cases; evaluate uncertainty handling on validation']]), '',
              f"High-confidence errors constitute {by_group['confidence_ge09']['errors']/by_group['all']['errors']:.1%} "
              'of this run\'s errors. The diagnostic groups overlap and these proposed causes are not causal findings.', '',
              '## EDA hypotheses H1–H4', '',
              f"**H1 (Banh chung / Banh tet):** {hypotheses['H1']['within_group_confusions']} within-pair confusions "
              f"among {hypotheses['H1']['support']} true images of these classes. See the directed confusion table for asymmetry.", '',
              f"**H2 (Bun bo Hue / Bun rieu / Banh canh):** {hypotheses['H2']['within_group_confusions']} within-group "
              f"confusions among {hypotheses['H2']['support']} images. Broth color or hidden noodles are possible visual explanations, "
              'not labels inferred automatically from the confusion counts.', '',
              f"**H3 (extreme aspect ratio):** {hypotheses['H3']['observed_errors']} errors among "
              f"{hypotheses['H3']['extreme_support']} extreme-aspect images. Using normal-aspect error rates within each "
              f"true class gives {hypotheses['H3']['expected_errors_using_same_class_normal_rates']:.2f} expected errors "
              f"for {hypotheses['H3']['covered_support']} covered images. This adjustment does not establish that cropping caused the errors.", '',
              f"Observed extreme/normal error rates are {by_group['extreme_aspect']['error_rate']:.1%} / "
              f"{by_group['normal_aspect']['error_rate']:.1%}. For this illustrative full run, the original claim that extreme-aspect "
              'images fail more often is **not supported**; the extreme subset is small and its interval is wide.', '',
              image('a2_test_crop_cases.png'), '',
              '**H4 (Banh pia packaging / white background):** the white-border proxy measures the fraction of near-white '
              'pixels in the outer 20% of a 64×64 image. Near-white means all RGB channels ≥220/255 and their range ≤30/255; '
              'a fraction ≥0.5 defines the proxy-positive group. Plates and food may also be white, so this is not a packaging annotation. '
              'Compare the two Banh pia groups above and inspect Grad-CAM examples. Without semantic region annotations or a controlled '
              'occlusion study, reliance on packaging/background remains unresolved.', '',
              '## Qualitative evaluation', '',
              'All galleries show the exact Resize/CenterCrop input, with predictions from the same three illustrative runs. '
              'Selection is deterministic: correct cases use margin ≥0.5 and path hashing with class diversity; hard-correct '
              'cases have the smallest top-1/top-2 margins; failure cases prioritize frequent confusion pairs and confidence. '
              'Improvements and regressions are both shown. Full manifests retain every selected image path and selection rule.', '']
    for category in ['correct','hard_correct','wrong','high_confidence_wrong','improvements','regressions']:
        lines += ['### '+category.replace('_',' ').title(), '', image(f'a2_test_examples_{category}.png'), '']
    lines += ['## Grad-CAM diagnostics', '',
              'Maps use gradient-weighted final-convolution activations followed by ReLU '
              '([Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391)). They are coarse explanatory maps, '
              'not segmentation masks or causal proof. Target classes are fixed from the saved predictions/ground truth. '
              'No weights or BatchNorm statistics are updated.', '']
    cam_manifest = ANALYSIS/'gradcam_manifest.json'
    if cam_manifest.exists():
        for entry in json.loads(cam_manifest.read_text()):
            lines += [image(entry['figure']), '']
    review_path = ANALYSIS/'qualitative_review.csv'
    reviews = read_csv(review_path) if review_path.exists() else []
    lines += ['## Visual observations and improvement ideas', '']
    if reviews:
        lines += ['The observations below come from inspection of the exported galleries/Grad-CAM figures. '
                  'They are illustrative rather than prevalence estimates; suspected annotation issues are not relabeled.', '',
                  markdown_table(['Image / group','Observation','Hypothesized cause','Improvement idea'],
                  [[r['path'],r['observation'],r['hypothesized_cause'],r['improvement_idea']] for r in reviews]), '']
    else:
        lines += ['Visual observations are pending inspection of the generated galleries; no causes have been fabricated.', '']
    lines += ['## Limitations and conclusion', '',
              'Results cover the cleaned test split and these fixed checkpoints. The CNN baseline is a single draft run. '
              'Error groups overlap; sparse groups have uncertain rates. Source-domain leakage and remaining label noise '
              'were not resolved by this analysis. The original uncleaned test set is not evaluated here. '
              'No test-driven model changes are made.', '',
              'Training costs come from RTX 4060 logs; current inference costs come from a different device and cannot be '
              'treated as measurements on one shared device. Forward timings use one warmup batch and a single streamed pass; '
              'they exclude preprocessing/transfers and are not repeated latency benchmarks.', '',
              'Frozen/full mean test macro-F1: '
              f"**{summary['resnet50_frozen']['macro_f1_mean']:.4f} / {summary['resnet50_full']['macro_f1_mean']:.4f}**. "
              'Use the paired results, error examples and compute costs together when drawing the final conclusion.', '',
              '## Reproduce', '', '```bash',
              'python scripts/a2_evaluate_all.py --split test --device auto --batch-size 32',
              'python scripts/a2_error_analysis.py', 'python scripts/a2_gradcam.py',
              'python scripts/a2_error_analysis.py', '```', '',
              'The batch evaluator reuses complete matching exports by default. The second analysis pass incorporates '
              'the Grad-CAM manifest and visual review CSV without rerunning test inference. Required packages are '
              'already listed in requirements.txt (torch, torchvision, numpy, Pillow through torchvision, matplotlib, PyYAML).', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for directory in [FIGURES,TABLES,ANALYSIS]:
        directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    protocol = json.loads((TABLES/'a2_test_protocol.json').read_text())
    summary = json.loads((TABLES/'a2_test_summary.json').read_text())
    split = json.loads((ROOT/'results/a2/splits/split_v2.json').read_text());classes=split['classes']
    all_rows = {};metrics={}
    for run_id in protocol['run_ids']:
        out=ROOT/'results/a2/runs'/run_id/'evaluation/test'
        all_rows[run_id]=load_predictions(out/'predictions.csv',split['test'],classes)
        metrics[run_id]=json.loads((out/'metrics.json').read_text())
        matrix=confusion_from_rows(all_rows[run_id],len(classes))
        if matrix.tolist()!=metrics[run_id]['confusion']:
            raise ValueError(f'Confusion mismatch: {run_id}')
    representatives=protocol['representatives']
    rows={model:all_rows[item['run_id']] for model,item in representatives.items()}
    matrices={model:confusion_from_rows(data,len(classes)) for model,data in rows.items()}
    primary=rows['resnet50_full'];frozen=rows['resnet50_frozen']
    config=yaml.safe_load((ROOT/'results/a2/runs'/representatives['resnet50_full']['run_id']/'config.yaml').read_text())
    data_root=Path(config['data_dir']);data_root=data_root if data_root.is_absolute() else ROOT/data_root
    feature_path=TABLES/'a2_test_image_features.csv'
    if feature_path.exists():
        features=read_csv(feature_path)
        if [f['path'] for f in features]!=[r['path'] for r in primary]:
            raise ValueError('Stale image feature cache')
        for f in features:
            for key in ['mean_rgb_brightness','near_white_border_fraction']:f[key]=float(f[key])
    else:
        features=image_features(primary,data_root);write_csv(feature_path,features)
    confusions=[record for model,matrix in matrices.items() for record in top_confusions(model,matrix,classes)]
    write_csv(TABLES/'a2_test_top_confusions.csv',confusions)
    groups=[g for model,data in rows.items() for g in error_groups(model,data,features)]
    write_csv(TABLES/'a2_test_error_groups.csv',groups)
    pairs=[]
    for seed in [42,43,44]:
        a=next(v for v in metrics.values() if v['model']=='resnet50_frozen' and v['seed']==seed)
        b=next(v for v in metrics.values() if v['model']=='resnet50_full' and v['seed']==seed)
        pairs.append(dict(seed=seed,frozen_f1=a['macro_f1'],full_f1=b['macro_f1'],gain=b['macro_f1']-a['macro_f1']))
    write_csv(TABLES/'a2_test_paired_seed_gains.csv',pairs)
    compute=[]
    for model in summary:
        model_runs=[run_id for run_id,m in metrics.items() if m['model']==model]
        training=[json.loads((ROOT/'results/a2/runs'/run_id/'metrics.json').read_text()) for run_id in model_runs]
        peaks=[t['peak_gpu_mem_gb'] for t in training if t.get('peak_gpu_mem_gb') is not None]
        compute.append(dict(model=model,parameters=training[0]['param'],trainable_parameters=training[0].get('param_trainable',training[0]['param']),
                            training_minutes_mean=statistics.mean(t['train_time']/60 for t in training),
                            peak_training_gpu_gib=statistics.mean(peaks) if peaks else None,
                            forward_ms_per_image_mean=summary[model]['inference_ms_per_image_mean']))
    write_csv(TABLES/'a2_test_compute_cost.csv',compute)
    comparisons=[]
    for label,condition in [('both_correct',lambda a,b:a and b),('full_only_correct',lambda a,b:a and not b),
                            ('frozen_only_correct',lambda a,b:not a and b),('both_wrong',lambda a,b:not a and not b)]:
        count=sum(condition(a['correct'],b['correct']) for a,b in zip(primary,frozen))
        comparisons.append(dict(outcome=label,count=count,fraction=count/len(primary)))
    write_csv(TABLES/'a2_test_prediction_agreement.csv',comparisons)
    hypotheses={'H1':within_group_confusions(primary,SHAPE_PAIR),
                'H2':within_group_confusions(primary,SOUP_GROUP),'H3':class_adjusted_extreme(primary)}
    (ANALYSIS/'hypothesis_results.json').write_text(json.dumps(hypotheses,indent=2)+'\n')
    plot_metrics(summary)
    for model,matrix in matrices.items():plot_confusion(model,matrix,classes)
    plot_per_class(classes,matrices)
    selected=select_examples(primary,frozen);manifest=[]
    for category,data in selected.items():
        gallery(category,data,data_root,config,rows)
        for row in data:manifest.append(dict(category=category,run_id=row['run_id'],index=row['index'],path=row['path'],
                                             true_class=row['true_class'],pred_class=row['pred_class'],confidence=row['confidence'],
                                             confidence_margin=row['confidence_margin']))
    for row in crop_gallery(primary,data_root,config):
        manifest.append(dict(category='crop_case',run_id=row['run_id'],index=row['index'],path=row['path'],
                             true_class=row['true_class'],pred_class=row['pred_class'],confidence=row['confidence'],confidence_margin=row['confidence_margin']))
    write_csv(ANALYSIS/'qualitative_manifest.csv',manifest)
    review=ANALYSIS/'qualitative_review.csv'
    if not review.exists():
        write_csv(review,[],['path','observation','hypothesized_cause','improvement_idea'])
    report=write_report(summary,representatives,rows,matrices,classes,groups,pairs,hypotheses)
    print('ANALYSIS_COMPLETE',report,flush=True)
    print('HYPOTHESES',json.dumps(hypotheses),flush=True)
    print('PRIMARY_ERROR_GROUPS',json.dumps([g for g in groups if g['model']=='resnet50_full']),flush=True)


if __name__=='__main__':
    main()
