#!/usr/bin/env python
# coding: utf-8
# Ablation Studies — LiveKVQuantP
# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Task: narrativeqa (LongBench v1, all samples)
# Figures saved to figures/

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

ABL_DIR = '../results/liveKVQuant/ablations'

def load(fname):
    with open(os.path.join(ABL_DIR, fname)) as f:
        d = json.load(f)
    return {
        'score':   d['avg_score'],
        'e2e':     d['avg_end_to_end_latency_ms'],
        'prefill': d['avg_prefill_latency_ms'],
        'decode':  d['avg_decode_latency_ms'],
        'mem':     d['max_peak_memory_mb'],
        'args':    d.get('args', {}),
    }


def load_all():
    baseline   = load('20260311_0311_v1_narrativeqa_baseline.json')
    no_warmup  = load('20260415_1249_v1_narrativeqa_wo_warmup.json')
    no_outlier = load('20260415_1332_v1_narrativeqa_wo_outlier.json')
    ema_minmax = load('20260415_1428_v1_narrativeqa_ema_minmax.json')
    qsl_0      = load('20260415_1521_v1_narrativeqa_wo_layerBypass.json')
    cs256      = load('20260415_1708_v1_narrativeqa_chunkSize256.json')
    cs1024     = load('20260415_1745_v1_narrativeqa_chunkSize1024.json')
    cs2048     = load('20260415_1817_v1_narrativeqa_chunkSize2048.json')
    cs4096     = load('20260415_1848_v1_narrativeqa_chunkSize4096.json')
    or_005     = load('20260415_1939_v1_narrativeqa_outlier0.005.json')
    or_050     = load('20260415_2030_v1_narrativeqa_outlier0.05.json')
    alpha_02   = load('20260415_2115_v1_narrativeqa_alpha0.2.json')
    alpha_03   = load('20260415_2116_v1_narrativeqa_alpha0.3.json')
    alpha_05   = load('20260415_2117_v1_narrativeqa_alpha0.5.json')
    clip_10    = load('20260415_2119_v1_narrativeqa_clip1.json')
    clip_30    = load('20260415_2120_v1_narrativeqa_clip3.json')
    clip_50    = load('20260415_2358_v1_narrativeqa_clip5.json')
    clip_70    = load('20260416_0048_v1_narrativeqa_clip7.json')

    return dict(
        baseline=baseline, no_warmup=no_warmup, no_outlier=no_outlier,
        ema_minmax=ema_minmax, qsl_0=qsl_0,
        cs256=cs256, cs512=baseline, cs1024=cs1024, cs2048=cs2048, cs4096=cs4096,
        or_005=or_005, or_010=baseline, or_050=or_050,
        alpha_01=baseline, alpha_02=alpha_02, alpha_03=alpha_03, alpha_05=alpha_05,
        clip_10=clip_10, clip_30=clip_30, clip_40=baseline, clip_50=clip_50, clip_70=clip_70,
    )


# ── Figure 1: Architectural Ablation ─────────────────────────────────────────

def plot_architecture(data):
    configs = [
        ('Baseline',             data['baseline']),
        ('w/o warmup_chunk',     data['no_warmup']),
        ('w/o outlier_isolation',data['no_outlier']),
        ('w/ ema_minmax',        data['ema_minmax']),
        ('w/o layer_bypass',     data['qsl_0']),
    ]
    labels = [c[0] for c in configs]
    scores = [c[1]['score'] for c in configs]
    e2e    = [c[1]['e2e'] / 1000 for c in configs]
    mem    = [c[1]['mem'] / 1024 for c in configs]
    colors = ['#2d7e4a', '#5bb37f', '#d45f5f', '#d4955f', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    bars = ax.bar(labels, scores, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('F1 Score'); ax.set_title('Score by Config'); ax.set_ylim(0.25, 0.31)
    for b, v in zip(bars, scores):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, e2e, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('E2E Latency (s)'); ax.set_title('Latency by Config')
    for b, v in zip(bars, e2e):
        ax.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    bars = ax.bar(labels, mem, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Peak Memory (GB)'); ax.set_title('Memory by Config'); ax.set_ylim(25.5, 27)
    for b, v in zip(bars, mem):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Layer 1: Architectural Ablation (narrativeqa)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/ablation_architecture.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {FIG_DIR}/ablation_architecture.png')


# ── Figure 2: Chunk Size Sensitivity ─────────────────────────────────────────

def plot_chunk_size(data):
    configs = [(256, data['cs256']), (512, data['cs512']), (1024, data['cs1024']),
               (2048, data['cs2048']), (4096, data['cs4096'])]
    labels  = [str(c[0]) for c in configs]
    scores  = [c[1]['score'] for c in configs]
    e2e     = [c[1]['e2e'] / 1000 for c in configs]
    prefill = [c[1]['prefill'] / 1000 for c in configs]
    decode  = [c[1]['decode'] / 1000 for c in configs]
    mem     = [c[1]['mem'] / 1024 for c in configs]
    color_main, color_hl = '#2d7e4a', '#f5a623'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(labels, scores, 'o-', color=color_main, markersize=8, linewidth=2)
    ax.scatter(['512'], [scores[1]], color=color_hl, s=120, zorder=5, edgecolors='black')
    ax.set_xlabel('Chunk Size'); ax.set_ylabel('F1 Score'); ax.set_title('Score vs Chunk Size')
    for l, v in zip(labels, scores):
        ax.annotate(f'{v:.4f}', (l, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)

    ax = axes[1]
    x = np.arange(len(labels))
    ax.bar(x, prefill, label='Prefill', color='#2d7e4a', alpha=0.88, edgecolor='white')
    ax.bar(x, decode, bottom=prefill, label='Decode', color='#a8d5b5', alpha=0.88, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Chunk Size'); ax.set_ylabel('Latency (s)'); ax.set_title('Prefill + Decode Latency')
    ax.legend(fontsize=9)
    for i, v in enumerate(e2e):
        ax.text(i, v+0.3, f'{v:.1f}s', ha='center', fontsize=9)

    ax = axes[2]
    ax.plot(labels, mem, 's-', color='#9b1c1c', markersize=8, linewidth=2)
    ax.set_xlabel('Chunk Size'); ax.set_ylabel('Peak Memory (GB)'); ax.set_title('Memory vs Chunk Size')
    for l, v in zip(labels, mem):
        ax.annotate(f'{v:.2f}', (l, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)

    fig.suptitle('Layer 2: Chunk Size Sensitivity (narrativeqa)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/ablation_chunk_size.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {FIG_DIR}/ablation_chunk_size.png')


# ── Figure 3: Outlier Ratio Sensitivity ──────────────────────────────────────

def plot_outlier_ratio(data):
    configs = [('0.5%', data['or_005']), ('1.0%\n(baseline)', data['or_010']), ('5.0%', data['or_050'])]
    labels  = [c[0] for c in configs]
    scores  = [c[1]['score'] for c in configs]
    e2e     = [c[1]['e2e'] / 1000 for c in configs]
    mem     = [c[1]['mem'] / 1024 for c in configs]
    colors  = ['#5bb37f', '#2d7e4a', '#d4955f']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, ylabel, title, fmt in [
        (axes[0], scores, 'F1 Score',       'Score vs Outlier Ratio',   '.4f'),
        (axes[1], e2e,   'E2E Latency (s)', 'Latency vs Outlier Ratio', '.1f'),
        (axes[2], mem,   'Peak Memory (GB)','Memory vs Outlier Ratio',  '.2f'),
    ]:
        bars = ax.bar(labels, vals, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
        ax.set_ylabel(ylabel); ax.set_xlabel('Outlier Ratio'); ax.set_title(title)
        for b, v in zip(bars, vals):
            label = f'{v:{fmt}}' + ('s' if 'Latency' in ylabel else '')
            ax.text(b.get_x()+b.get_width()/2, v+(max(vals)-min(vals))*0.02,
                    label, ha='center', va='bottom', fontsize=9)
    axes[0].set_ylim(0.27, 0.30)

    fig.suptitle('Layer 2: Outlier Ratio Sensitivity (narrativeqa)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/ablation_outlier_ratio.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {FIG_DIR}/ablation_outlier_ratio.png')


# ── Figure 4: EMA Alpha Sensitivity ──────────────────────────────────────────

def plot_ema_alpha(data):
    configs = [('0.1\n(baseline)', data['alpha_01']), ('0.3', data['alpha_03']), ('0.5', data['alpha_05'])]
    labels  = [c[0] for c in configs]
    scores  = [c[1]['score'] for c in configs]
    e2e     = [c[1]['e2e'] / 1000 for c in configs]
    mem     = [c[1]['mem'] / 1024 for c in configs]
    colors  = ['#f5a623', '#5bb37f', '#8fc9a8']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    bars = ax.bar(labels, scores, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('F1 Score'); ax.set_xlabel('EMA Alpha'); ax.set_title('Score vs EMA Alpha'); ax.set_ylim(0.27, 0.30)
    for b, v in zip(bars, scores):
        ax.text(b.get_x()+b.get_width()/2, v+0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, e2e, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('E2E Latency (s)'); ax.set_xlabel('EMA Alpha'); ax.set_title('Latency vs EMA Alpha')
    for b, v in zip(bars, e2e):
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    bars = ax.bar(labels, mem, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Peak Memory (GB)'); ax.set_xlabel('EMA Alpha'); ax.set_title('Memory vs EMA Alpha')
    ax.set_ylim(26.0, 26.5)
    for b, v in zip(bars, mem):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Layer 2: EMA Alpha Sensitivity (narrativeqa)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/ablation_ema_alpha.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {FIG_DIR}/ablation_ema_alpha.png')


# ── Figure 5: Clip Factor Sensitivity ────────────────────────────────────────

def plot_clip_factor(data):
    configs = [('1.0', data['clip_10']), ('3.0', data['clip_30']),
               ('4.0\n(baseline)', data['clip_40']), ('5.0', data['clip_50']), ('7.0', data['clip_70'])]
    labels  = [c[0] for c in configs]
    scores  = [c[1]['score'] for c in configs]
    e2e     = [c[1]['e2e'] / 1000 for c in configs]
    mem     = [c[1]['mem'] / 1024 for c in configs]
    colors  = ['#5bb37f', '#2d7e4a', '#d4955f', '#d45f5f', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    bars = ax.bar(labels, scores, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('F1 Score'); ax.set_xlabel('Clip Factor'); ax.set_title('Score vs Clip Factor'); ax.set_ylim(0.26, 0.30)
    for b, v in zip(bars, scores):
        ax.text(b.get_x()+b.get_width()/2, v+0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, e2e, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('E2E Latency (s)'); ax.set_xlabel('Clip Factor'); ax.set_title('Latency vs Clip Factor')
    for b, v in zip(bars, e2e):
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    bars = ax.bar(labels, mem, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Peak Memory (GB)'); ax.set_xlabel('Clip Factor'); ax.set_title('Memory vs Clip Factor')
    ax.set_ylim(26.0, 26.5)
    for b, v in zip(bars, mem):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Layer 2: Clip Factor Sensitivity (narrativeqa)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/ablation_clip_factor.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {FIG_DIR}/ablation_clip_factor.png')


# ── Figure 6: Score–Latency Trade-off ────────────────────────────────────────

def plot_tradeoff(data):
    all_configs = {
        'Baseline':      data['baseline'],
        'warmup=false':  data['no_warmup'],
        'outlier=false': data['no_outlier'],
        'ema_minmax':    data['ema_minmax'],
        'qsl=0':         data['qsl_0'],
        'cs=256':        data['cs256'],
        'cs=1024':       data['cs1024'],
        'cs=2048':       data['cs2048'],
        'cs=4096':       data['cs4096'],
        'or=0.5%':       data['or_005'],
        'or=5.0%':       data['or_050'],
        'α=0.2':         data['alpha_02'],
        'α=0.3':         data['alpha_03'],
        'α=0.5':         data['alpha_05'],
        'clip=3':        data['clip_30'],
        'clip=5':        data['clip_50'],
        'clip=7':        data['clip_70'],
    }
    group_colors = {
        'arch':  '#2d7e4a', 'chunk': '#1f77b4', 'ratio': '#d45f5f',
        'hybrid':'#9467bd', 'alpha': '#ff7f0e', 'clip':  '#8c564b',
    }
    config_groups = {
        'Baseline': 'arch', 'warmup=false': 'arch', 'outlier=false': 'arch',
        'ema_minmax': 'arch', 'qsl=0': 'hybrid',
        'cs=256': 'chunk', 'cs=1024': 'chunk', 'cs=2048': 'chunk', 'cs=4096': 'chunk',
        'or=0.5%': 'ratio', 'or=5.0%': 'ratio',
        'α=0.2': 'alpha', 'α=0.3': 'alpha', 'α=0.5': 'alpha',
        'clip=3': 'clip', 'clip=5': 'clip', 'clip=7': 'clip',
    }
    label_offsets = {
        # well-separated points — right side, simple nudge
        'cs=4096':       (  8, -14),
        'cs=2048':       (  8,   6),
        'cs=1024':       (  8,   6),
        'outlier=false': (  8,  -14),
        'α=0.2':         (  8,   6),
        'or=5.0%':       (  8,  -14),
        'or=0.5%':       (  8,   6),
        'qsl=0':         (  8,   6),
        'ema_minmax':    (  8,   6),
        'cs=256':        (  8,   6),
        # Baseline — highest score, right side pushed up
        'Baseline':      (  8,   6),
        # dense cluster — clip goes LEFT, alpha/warmup goes RIGHT
        'clip=5':        (-55,  28),   # left
        'clip=7':        (-55,   0),   # left
        'clip=3':        (-55, -28),   # left
        'α=0.5':         ( 55,  28),   # right
        'α=0.3':         ( 55,   0),   # right
        'warmup=false':  ( 55, -28),   # right
    }

    all_mems = [d['mem'] for d in all_configs.values()]
    mem_min, mem_range = min(all_mems), max(all_mems) - min(all_mems)
    def bubble_size(mem):
        return 40 + 260 * (mem - mem_min) / mem_range if mem_range > 0 else 100

    baseline = data['baseline']
    fig, ax = plt.subplots(figsize=(13, 6))

    arrow_props = dict(arrowstyle='->', color='gray', lw=0.8)
    for name, d in all_configs.items():
        x, y = d['e2e'] / 1000, d['score']
        ax.scatter(x, y, s=bubble_size(d['mem']), c=group_colors[config_groups[name]],
                   alpha=0.75, edgecolors='black', linewidth=0.5, zorder=3)
        dx, dy = label_offsets.get(name, (6, 6))
        # draw arrow only when label is far enough from the point
        arrowprops = arrow_props if (abs(dx) > 20 or abs(dy) > 20) else None
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(dx, dy),
                    fontsize=8, alpha=0.9, arrowprops=arrowprops)

    ax.axhline(y=baseline['score'], color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axvline(x=baseline['e2e']/1000, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('E2E Latency (s)'); ax.set_ylabel('F1 Score')
    ax.set_xlim(7, 35)
    y_min = min(d['score'] for d in all_configs.values()) - 0.003
    y_max = max(d['score'] for d in all_configs.values()) + 0.005
    ax.set_ylim(y_min, y_max)

    legend_patches = [mpatches.Patch(color=v, label=k.title(), alpha=0.8) for k, v in group_colors.items()]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, title='Group', title_fontsize=8, framealpha=0.6)

    fig.suptitle('F1 Score vs Latency Trade-off  (bubble size ∝ peak memory)', fontsize=12, y=1.01)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.1)
    fig.savefig(f'{FIG_DIR}/ablation_tradeoff.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {FIG_DIR}/ablation_tradeoff.png')


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary(data):
    rows = [
        ('Baseline (α=0.1)',   data['baseline']),
        ('warmup=false',       data['no_warmup']),
        ('outlier=false',      data['no_outlier']),
        ('ema_minmax',         data['ema_minmax']),
        ('chunk_size=256',     data['cs256']),
        ('chunk_size=1024',    data['cs1024']),
        ('chunk_size=2048',    data['cs2048']),
        ('chunk_size=4096',    data['cs4096']),
        ('outlier_ratio=0.5%', data['or_005']),
        ('outlier_ratio=5.0%', data['or_050']),
        ('quant_start_layer=0',data['qsl_0']),
        ('ema_alpha=0.1',      data['alpha_01']),
        ('ema_alpha=0.3',      data['alpha_03']),
        ('ema_alpha=0.5',      data['alpha_05']),
        ('clip_factor=3.0',    data['clip_30']),
        ('clip_factor=5.0',    data['clip_50']),
        ('clip_factor=7.0',    data['clip_70']),
    ]
    bs = data['baseline']['score']
    print(f'{"Config":<25s} {"Score":>8s} {"Δ Score":>9s} {"E2E (s)":>8s} {"Prefill":>8s} {"Decode":>8s} {"Mem (GB)":>9s}')
    print('-' * 78)
    for name, d in rows:
        delta = d['score'] - bs
        print(f'{name:<25s} {d["score"]:>8.4f} {delta:>+9.4f} {d["e2e"]/1000:>8.1f} {d["prefill"]/1000:>8.1f} {d["decode"]/1000:>8.1f} {d["mem"]/1024:>9.2f}')


# ── Main ──────────────────────────────────────────────────────────────────────

PLOTS = {
    'architecture':  plot_architecture,
    'chunk_size':    plot_chunk_size,
    'outlier_ratio': plot_outlier_ratio,
    'ema_alpha':     plot_ema_alpha,
    'clip_factor':   plot_clip_factor,
    'tradeoff':      plot_tradeoff,
    'summary':       print_summary,
}

if __name__ == '__main__':
    # ── Select which figures to generate ──────────────────────────────────────
    # Use 'all' to run everything, or list specific keys from PLOTS above.
    RUN = ['tradeoff']
    # RUN = 'all'
    # RUN = ['architecture', 'tradeoff', 'summary']
    # ──────────────────────────────────────────────────────────────────────────

    data = load_all()
    print('All ablation files loaded (17 configs).')

    targets = PLOTS.keys() if RUN == 'all' else RUN
    for key in targets:
        print(f'\n── {key} ──')
        PLOTS[key](data)
