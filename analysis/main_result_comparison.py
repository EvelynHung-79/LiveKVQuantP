#!/usr/bin/env python
# coding: utf-8
# KV Cache Compression Methods — Comparison on LongBench v1 & v2

import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

TASKS = [
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
    'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum',
    'passage_retrieval_en', 'passage_count', 'lcc', 'repobench-p',
]
TASK_LABELS = [t.replace('_', '\n') for t in TASKS]

# ── load FullKV ──────────────────────────────────────────────────────────────
fullkv = {}
for f in glob.glob('../results/fullKV/longbench_v1_results/*.json'):
    d = json.load(open(f))
    fullkv[d['task']] = {
        'score': d['avg_score'],
        'e2e':   d['avg_latency_ms'],
        'mem':   d['max_peak_memory_mb'],
    }

# ── load LiveKVQuantP ────────────────────────────────────────────────────────
lkp = {}
for f in glob.glob('../results/liveKVQuant/ema_alpha0.1/longbench_v1_results/*.json'):
    d = json.load(open(f))
    lkp[d['task']] = {
        'score':   d['avg_score'],
        'e2e':     d['avg_end_to_end_latency_ms'],
        'prefill': d['avg_prefill_latency_ms'],
        'decode':  d['avg_decode_latency_ms'],
        'mem':     d['max_peak_memory_mb'],
    }

# ── load KVQuant ─────────────────────────────────────────────────────────────
kvq = {}
for f in glob.glob('../../KVQuant/results/KVQuant/longbench_v1_results/*.json'):
    d = json.load(open(f))
    r = d['results']
    kvq[d['task']] = {
        'score':   r['avg_score'],
        'e2e':     r['avg_end_to_end_latency_ms'],
        'prefill': r['avg_prefill_latency_ms'],
        'decode':  r['avg_decode_latency_ms'],
        'mem':     r['max_peak_memory_mb'],
    }

# ── load Finch v3 ─────────────────────────────────────────────────────────────
finch = {}
v3_dir = '../../Finch/logs/longbench_v1_results/v3_result (new prompt template and penalty=1.0)/'
for f in glob.glob(v3_dir + '*.json'):
    d = json.load(open(f))
    task = d['configs']['task_name']
    r    = d['result']
    raw_score = r['avg_f1_score']
    finch[task] = {
        'score':   raw_score / 100.0,   # Finch stores as 0–100
        'e2e':     r['avg_e2e_latency_ms'],
        'prefill': r['avg_prefill_latency_ms'],
        'decode':  r['avg_decode_latency_ms'],
        'mem':     r['max_peak_memory'],
    }

print(f"Tasks loaded — FullKV: {len(fullkv)}, LKP: {len(lkp)}, KVQ: {len(kvq)}, Finch: {len(finch)}")
assert all(len(x) == 16 for x in [fullkv, lkp, kvq, finch]), "Missing tasks!"
print("All 16 tasks loaded ✓")

def build_df(metric, sources, labels):
    """Build a tasks × methods DataFrame for a given metric key."""
    rows = {}
    for task in TASKS:
        row = {}
        for src, lbl in zip(sources, labels):
            v = src.get(task, {}).get(metric)
            row[lbl] = v
        rows[task] = row
    df = pd.DataFrame(rows).T  # tasks as rows
    df.index.name = 'Task'
    return df[labels]   # column order

METHODS  = ['FullKV', 'LiveKVQuantP', 'KVQuant', 'Finch']
SOURCES  = [fullkv, lkp, kvq, finch]
COLORS   = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']
METHODS3 = ['LiveKVQuantP', 'KVQuant', 'Finch']
COLORS3  = ['#55A868', '#C44E52', '#DD8452']

# Model weight (Llama 3.1 8B, FP16) — subtracted from peak memory
MODEL_WEIGHT_MB = 14.96 * 1024   # 15 319.04 MB

df_score   = build_df('score', SOURCES, METHODS)
df_e2e     = build_df('e2e',   SOURCES, METHODS)
df_mem     = build_df('mem',   SOURCES, METHODS) - MODEL_WEIGHT_MB   # exclude model weight
df_prefill = build_df('prefill', [lkp, kvq, finch], METHODS3)
df_decode  = build_df('decode',  [lkp, kvq, finch], METHODS3)

# Average row (only numeric, skip NaN for FullKV e2e if any)
for df in [df_score, df_e2e, df_mem, df_prefill, df_decode]:
    df.loc['Average'] = df.mean(skipna=True)

print("DataFrames built ✓")
print(f"Model weight subtracted: {MODEL_WEIGHT_MB:.0f} MB ({MODEL_WEIGHT_MB/1024:.2f} GB)")

# Each task scored with its official LongBench v1 metric (F1 / ROUGE-L / Accuracy / Edit-sim).  
# Scores normalised to **0–1**.

def fmt_delta(val, ref):
    d = val - ref
    s = f'{d:+.4f}'
    return s

display_df = df_score.copy()
display_df['Δ LKP'] = df_score['LiveKVQuantP'] - df_score['FullKV']
display_df['Δ KVQ'] = df_score['KVQuant']      - df_score['FullKV']
display_df['Δ Finch']= df_score['Finch']        - df_score['FullKV']

styled = (display_df.style
    .format({
        'FullKV': '{:.4f}', 'LiveKVQuantP': '{:.4f}',
        'KVQuant': '{:.4f}', 'Finch': '{:.4f}',
        'Δ LKP': '{:+.4f}', 'Δ KVQ': '{:+.4f}', 'Δ Finch': '{:+.4f}',
    })
    .background_gradient(subset=['Δ LKP','Δ KVQ','Δ Finch'],
                         cmap='RdYlGn', vmin=-0.15, vmax=0.02)
    .set_caption('Table 1 — Score (0–1). Δ = method − FullKV.')
    .set_table_styles([{'selector': 'caption',
                        'props': [('font-size','13px'),('font-weight','bold'),
                                  ('text-align','left'),('margin-bottom','6px')]}])
)
styled

fig, ax = plt.subplots(figsize=(16, 4.5))
x     = np.arange(len(TASKS))
width = 0.20

for i, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals = [df_score.loc[t, method] for t in TASKS]
    bars = ax.bar(x + (i - 1.5) * width, vals, width,
                  label=method, color=color, alpha=0.88, edgecolor='white', linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Score (0–1)')
ax.set_title('Figure 1 — Per-task Score Comparison')
ax.legend(framealpha=0.3, fontsize=9)
ax.set_ylim(0, 1.12)
ax.axhline(0, color='k', linewidth=0.4)
plt.tight_layout()
plt.savefig('figures/score_comparison.png', bbox_inches='tight')
plt.show()

# Finch performs **attention-score based token dropping** — it permanently discards KV cache entries
# judged unimportant. Several structural mismatches with Llama 3.1 8B make this selection unreliable:
# 
# Tasks most affected: **passage_retrieval_en** (−0.83), **triviaqa** (−0.08), **hotpotqa** (−0.27) —
# all require locating a specific span or fact scattered across a long context.
# 
# KVQuant pre-allocates a **static KV buffer of `maxseqlen = 32 768` tokens**.  
# For tasks whose samples exceed that limit, the input is silently **truncated**, causing
# those samples to fail. `passage_count` has inputs that can exceed 32 k tokens, meaning
# several samples produce wrong answers regardless of quantisation quality.  
# *(See `KVQuant/docs/研究過程.md` — Bug 8: maxseqlen truncation.)*

# E2E latency = total wall-clock time per sample (prefill + decode).  
# Unit: **ms**. Lower is better.

display_e2e = df_e2e.copy()
for m in ['LiveKVQuantP', 'KVQuant', 'Finch']:
    display_e2e[f'{m} overhead'] = df_e2e[m] / df_e2e['FullKV']

styled_e2e = (display_e2e.style
    .format({
        'FullKV': '{:,.0f}', 'LiveKVQuantP': '{:,.0f}',
        'KVQuant': '{:,.0f}', 'Finch': '{:,.0f}',
        'LiveKVQuantP overhead': '{:.2f}x',
        'KVQuant overhead':      '{:.2f}x',
        'Finch overhead':        '{:.2f}x',
    })
    .background_gradient(subset=['LiveKVQuantP overhead','KVQuant overhead','Finch overhead'],
                         cmap='RdYlGn_r', vmin=0.8, vmax=4.0)
    .set_caption('Table 2 — E2E Latency (ms). Overhead = method / FullKV (>1.0x = slower).')
    .set_table_styles([{'selector':'caption',
                        'props':[('font-size','13px'),('font-weight','bold'),
                                 ('text-align','left'),('margin-bottom','6px')]}])
)
styled_e2e

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4.5))

# Left: absolute latency
x = np.arange(len(TASKS))
width = 0.20
for i, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals = [df_e2e.loc[t, method] / 1000 for t in TASKS]  # → seconds
    ax1.bar(x + (i-1.5)*width, vals, width, label=method, color=color, alpha=0.88,
            edgecolor='white', linewidth=0.4)
ax1.set_xticks(x); ax1.set_xticklabels(TASKS, rotation=35, ha='right', fontsize=9)
ax1.set_ylabel('E2E Latency (s)'); ax1.set_title('Figure 2a — E2E Latency (absolute)')
ax1.legend(framealpha=0.3, fontsize=9)

# Right: overhead vs FullKV (3 compressed methods only)
for i, (method, color) in enumerate(zip(METHODS3, COLORS3)):
    overheads = [df_e2e.loc[t, method] / df_e2e.loc[t, 'FullKV'] for t in TASKS]
    ax2.bar(x + (i-1)*width, overheads, width, label=method, color=color, alpha=0.88,
            edgecolor='white', linewidth=0.4)
ax2.axhline(1.0, color='k', linewidth=1, linestyle='--', alpha=0.5, label='FullKV baseline')
ax2.set_xticks(x); ax2.set_xticklabels(TASKS, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('Overhead vs FullKV'); ax2.set_title('Figure 2b — E2E Overhead (ratio vs FullKV)')
ax2.legend(framealpha=0.3, fontsize=9)

plt.tight_layout()
plt.savefig('figures/e2e_latency.png', bbox_inches='tight')
plt.show()

# Peak GPU memory (MB) measured during inference, **minus model weight (14.96 GB)**.
# This isolates KV cache + activation overhead. Lower is better.

display_mem = df_mem.copy()
for m in ['LiveKVQuantP', 'KVQuant', 'Finch']:
    display_mem[f'Δ {m} (MB)'] = df_mem[m] - df_mem['FullKV']

styled_mem = (display_mem.style
    .format({
        'FullKV': '{:,.0f}', 'LiveKVQuantP': '{:,.0f}',
        'KVQuant': '{:,.0f}', 'Finch': '{:,.0f}',
        'Δ LiveKVQuantP (MB)': '{:+,.0f}',
        'Δ KVQuant (MB)':      '{:+,.0f}',
        'Δ Finch (MB)':        '{:+,.0f}',
    })
    .background_gradient(subset=['Δ LiveKVQuantP (MB)','Δ KVQuant (MB)','Δ Finch (MB)'],
                         cmap='RdYlGn', vmin=-10000, vmax=3000)
    .set_caption('Table 3 — Peak GPU Memory excl. model weight (MB). Δ = method − FullKV.')
    .set_table_styles([{'selector':'caption',
                        'props':[('font-size','13px'),('font-weight','bold'),
                                 ('text-align','left'),('margin-bottom','6px')]}])
)
styled_mem

fig, ax = plt.subplots(figsize=(16, 4.5))
x = np.arange(len(TASKS))
width = 0.20
for i, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals = [df_mem.loc[t, method] / 1024 for t in TASKS]  # → GB
    ax.bar(x + (i-1.5)*width, vals, width, label=method, color=color, alpha=0.88,
           edgecolor='white', linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(TASKS, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Memory excl. model weight (GB)')
ax.set_title('Figure 3 — Peak GPU Memory excl. Model Weight (14.96 GB)')
ax.legend(framealpha=0.3, fontsize=9)
plt.tight_layout()
plt.savefig('figures/memory.png', bbox_inches='tight')
plt.show()

# KVQuant pre-allocates a **static INT4 KV buffer of `maxseqlen = 32 768` tokens** at model load time,
# regardless of the actual input length.  
# Even for a 2 k-token input (e.g. `qasper`), the full 32 k INT4 buffer is resident on the GPU,
# leaving peak memory close to FullKV.
# In contrast:
# - **LiveKVQuantP** allocates KV buffers chunk-by-chunk; only the actual sequence length is stored → memory scales with input.
# - **Finch** drops tokens (reducing KV cache size), but stores the remaining tokens in **FP16**, giving moderate but consistent memory savings.

# Comparing **LiveKVQuantP**, **KVQuant**, and **Finch** only  
# (FullKV does not have prefill/decode split instrumentation).
# 

# Build detailed breakdown DataFrame
rows = {}
for task in TASKS:
    row = {}
    for src, m in zip([lkp, kvq, finch], METHODS3):
        e2e     = src[task]['e2e']
        prefill = src[task]['prefill']
        decode  = src[task]['decode']
        row[f'{m} Prefill'] = prefill
        row[f'{m} Decode']  = decode
        row[f'{m} E2E']     = e2e
        row[f'{m} Prefill%']= prefill / e2e * 100
    rows[task] = row

df_breakdown = pd.DataFrame(rows).T
df_breakdown.index.name = 'Task'

# Average row
df_breakdown.loc['Average'] = df_breakdown.mean()

# Display summary: Prefill / Decode / Prefill% for each method
cols_show = []
for m in METHODS3:
    cols_show += [f'{m} Prefill', f'{m} Decode', f'{m} Prefill%']

fmt_dict = {}
for m in METHODS3:
    fmt_dict[f'{m} Prefill']  = '{:,.0f}'
    fmt_dict[f'{m} Decode']   = '{:,.0f}'
    fmt_dict[f'{m} Prefill%'] = '{:.1f}%'

(df_breakdown[cols_show].style
    .format(fmt_dict)
    .background_gradient(subset=[f'{m} Prefill%' for m in METHODS3],
                         cmap='Oranges', vmin=20, vmax=100)
    .set_caption('Table 4 — Prefill / Decode Latency (ms) and Prefill% per method.')
    .set_table_styles([{'selector':'caption',
                        'props':[('font-size','13px'),('font-weight','bold'),
                                 ('text-align','left'),('margin-bottom','6px')]}])
)

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
PREFILL_COLORS = {'LiveKVQuantP': '#2d7e4a', 'KVQuant': '#9b1c1c', 'Finch': '#a0522d'}
DECODE_COLORS  = {'LiveKVQuantP': '#a8d5b5', 'KVQuant': '#f5a0a0', 'Finch': '#f5c79e'}

x = np.arange(len(TASKS))
w = 0.55

for ax, m in zip(axes, METHODS3):
    prefills = [df_breakdown.loc[t, f'{m} Prefill'] / 1000 for t in TASKS]
    decodes  = [df_breakdown.loc[t, f'{m} Decode']  / 1000 for t in TASKS]
    bar1 = ax.bar(x, prefills, w, label='Prefill', color=PREFILL_COLORS[m], alpha=0.9)
    bar2 = ax.bar(x, decodes,  w, bottom=prefills, label='Decode',
                  color=DECODE_COLORS[m], alpha=0.9)
    ax.set_title(f'{m}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Latency (s)')
    ax.legend(fontsize=9, framealpha=0.3)

fig.suptitle('Figure 4 — Prefill vs Decode Latency Breakdown (s)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('figures/prefill_decode_breakdown.png', bbox_inches='tight')
plt.show()

# Average Prefill% per method
avg_prefill_pct = {m: df_breakdown.loc['Average', f'{m} Prefill%'] for m in METHODS3}
avg_e2e         = {m: df_breakdown.loc['Average', f'{m} E2E'] / 1000 for m in METHODS3}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Bar: average prefill%
bars = ax1.bar(METHODS3, [avg_prefill_pct[m] for m in METHODS3],
               color=COLORS3, alpha=0.88, edgecolor='white', width=0.5)
ax1.set_ylabel('Avg Prefill Latency (%)')
ax1.set_title('Figure 5a — Average Prefill % of E2E')
ax1.set_ylim(0, 105)
for bar, m in zip(bars, METHODS3):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{avg_prefill_pct[m]:.1f}%', ha='center', va='bottom', fontsize=10)

# Bar: average E2E
bars2 = ax2.bar(METHODS3, [avg_e2e[m] for m in METHODS3],
                color=COLORS3, alpha=0.88, edgecolor='white', width=0.5)
ax2.set_ylabel('Avg E2E Latency (s)')
ax2.set_title('Figure 5b — Average E2E Latency')
for bar, m in zip(bars2, METHODS3):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{avg_e2e[m]:.1f}s', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/avg_latency_summary.png', bbox_inches='tight')
plt.show()

# KVQuant applies quantisation **inline during prefill**: for every chunk of the input,
# it runs a FP16 → INT4 CUDA kernel to compress the K/V cache before continuing.  
# This means:
# 1. **Every prefill step writes to a compressed buffer** using custom CUDA kernels (`quant_and_pack_vcache` / `quant_and_pack_kcache`).
# 2. The kernels were originally written for **MHA** (1 KV head per attention head); supporting GQA required a `num_groups` patch, adding branching overhead.
# 3. The **RoPE scaling fix** for Llama 3.1 (per-frequency scaling inside the K-cache kernel) adds an extra computation pass during prefill.
# Net effect: KVQuant's prefill is **~10× longer** than LiveKVQuantP on narrativeqa despite both operating on the same sequence length.  
# *(See `KVQuant/docs/研究過程.md` §2026-03-19-140000 and `docs/重點.md`)*
# 
# Finch does **not** reduce the prefill computation — it must still attend over the full context
# to compute attention scores for token selection.  
# In fact, prefill is *slightly heavier* than FullKV because:
# - Each split-size chunk (512 tokens) triggers an extra attention pass to score tokens.
# - The score aggregation across GQA heads adds a small but non-zero per-chunk overhead.
# Finch saves memory (by discarding tokens from the KV cache), but the saving only materialises
# 
# LiveKVQuantP's chunk-wise design keeps quantisation in a **streaming pipeline**:
# - Quantisation happens per chunk and immediately frees the FP16 activations.
# - Uses SDPA (Flash-Attention) for the attention computation, avoiding the explicit O(n²) attention matrix.
# - INT4 storage is allocated dynamically, scaling with actual sequence length.
# *(See `LiveKVQuantP/docs/memory_optimization_analysis.md` — Proposal A: SDPA fix, ~8 GB saved.)*

# 
# - **LiveKVQuantP** has the best accuracy–efficiency balance: only −1.2% score with moderate latency overhead.
# - **KVQuant** preserves accuracy well (−0.5%) but pays a large prefill cost (+2.24× E2E) due to inline FP16→INT4 kernels, and *uses more memory than FullKV* because of static `maxseqlen` pre-allocation.
# - **Finch** gives the best memory savings (−3.4 GB avg) but accuracy drops sharply (−17.9%) due to GQA voting noise and flat attention distributions under Llama 3.1's large RoPE θ.

print('=== Average Scores ===')
for m in METHODS:
    avg = df_score.loc[TASKS, m].mean()
    print(f'  {m:16s}: {avg:.4f}')

print()
print('=== Average E2E Latency (s) ===')
for m in METHODS:
    avg = df_e2e.loc[TASKS, m].mean() / 1000
    print(f'  {m:16s}: {avg:.2f}s')

print()
print('=== Average Prefill % ===')
for m in METHODS3:
    avg = df_breakdown.loc[TASKS, f'{m} Prefill%'].mean()
    print(f'  {m:16s}: {avg:.1f}%')

print()
print(f'=== Average Memory excl. model weight ({MODEL_WEIGHT_MB/1024:.2f} GB) ===')
for m in METHODS:
    avg = df_mem.loc[TASKS, m].mean() / 1024
    print(f'  {m:16s}: {avg:.2f} GB')

# 
# 
# 1. Accuracy (Score)
# 2. End-to-End Latency
# 3. Peak GPU Memory
# 4. Prefill / Decode Breakdown
# 5. Summary

import json, glob, numpy as np, pandas as pd

V2_TASKS = [
    'Single-Document QA', 'Multi-Document QA', 'Long In-context Learning',
    'Long-dialogue History Understanding', 'Code Repository Understanding',
    'Long Structured Data Understanding',
]
V2_SHORT = ['Single\nDoc QA', 'Multi\nDoc QA', 'Long\nICL',
            'Long\nDialogue', 'Code\nRepo', 'Long\nStruct Data']

# ── load FullKV v2 ────────────────────────────────────────────────────────────
fullkv_v2 = {}
for f in glob.glob('../results/fullKV/longbench_v2_results/*.json'):
    d = json.load(open(f))
    fullkv_v2[d['task']] = {
        'score': d['avg_score'], 'e2e': d['avg_end_to_end_latency_ms'],
        'mem': d['max_peak_memory_mb'], 'prefill': 0.0, 'decode': 0.0,
    }

# ── load LiveKVQuantP v2 ──────────────────────────────────────────────────────
lkp_v2 = {}
for f in glob.glob('../results/liveKVQuant/ema_alpha0.1/longbench_v2_results/*.json'):
    d = json.load(open(f))
    lkp_v2[d['task']] = {
        'score': d['avg_score'], 'e2e': d['avg_end_to_end_latency_ms'],
        'prefill': d['avg_prefill_latency_ms'], 'decode': d['avg_decode_latency_ms'],
        'mem': d['max_peak_memory_mb'],
    }

# ── load KVQuant v2 ───────────────────────────────────────────────────────────
kvq_v2 = {}
for f in glob.glob('../../KVQuant/results/KVQuant/longbench_v2_results/*.json'):
    d = json.load(open(f))
    r = d['results']
    kvq_v2[d['domain']] = {
        'score': r['avg_accuracy'], 'e2e': r['avg_end_to_end_latency_ms'],
        'prefill': r['avg_prefill_latency_ms'], 'decode': r['avg_decode_latency_ms'],
        'mem': r['max_peak_memory_mb'],
    }

# ── load Finch v2 ─────────────────────────────────────────────────────────────
FINCH_V2_MAP = {
    'Single_Document_QA':                  'Single-Document QA',
    'Multi_Document_QA':                   'Multi-Document QA',
    'Long_In_context_Learning':            'Long In-context Learning',
    'Long_dialogue_History_Understanding': 'Long-dialogue History Understanding',
    'Code_Repository_Understanding':       'Code Repository Understanding',
    'Long_Structured_Data_Understanding':  'Long Structured Data Understanding',
}
finch_v2 = {}
for f in glob.glob('../../Finch/logs/longbench_v2_results/*.json'):
    d = json.load(open(f))
    task = FINCH_V2_MAP[d['configs']['task_name']]
    r = d['result']
    finch_v2[task] = {
        'score': r['avg_f1_score'] / 100.0, 'e2e': r['avg_e2e_latency_ms'],
        'prefill': r['avg_prefill_latency_ms'], 'decode': r['avg_decode_latency_ms'],
        'mem': r['max_peak_memory'],
    }

assert all(len(x) == 6 for x in [fullkv_v2, lkp_v2, kvq_v2, finch_v2])
print("All 6 v2 tasks loaded ✓")

def v2arr(d, metric):
    return np.array([d[t][metric] for t in V2_TASKS])

score_v2  = {m: v2arr(src, 'score') for m, src in zip(METHODS, [fullkv_v2, lkp_v2, kvq_v2, finch_v2])}
e2e_v2    = {m: v2arr(src, 'e2e')   for m, src in zip(METHODS, [fullkv_v2, lkp_v2, kvq_v2, finch_v2])}
mem_v2    = {m: v2arr(src, 'mem') - MODEL_WEIGHT_MB for m, src in zip(METHODS, [fullkv_v2, lkp_v2, kvq_v2, finch_v2])}
prefill_v2= {m: v2arr(src, 'prefill') for m, src in zip(METHODS3, [lkp_v2, kvq_v2, finch_v2])}
decode_v2 = {m: v2arr(src, 'decode')  for m, src in zip(METHODS3, [lkp_v2, kvq_v2, finch_v2])}

# Build summary DataFrame
df_v2 = pd.DataFrame({
    m: {t: score_v2[m][i] for i, t in enumerate(V2_TASKS)} for m in METHODS
})
df_v2.index.name = 'Task'
for m in METHODS3:
    df_v2[f'Δ {m}'] = df_v2[m] - df_v2['FullKV']
df_v2.loc['Average'] = df_v2.mean()

fmt = {m: '{:.4f}' for m in METHODS}
fmt.update({f'Δ {m}': '{:+.4f}' for m in METHODS3})
(df_v2.style.format(fmt)
 .background_gradient(subset=[f'Δ {m}' for m in METHODS3], cmap='RdYlGn', vmin=-0.6, vmax=0.02)
 .set_caption('Table 5 — LongBench v2 Score (0–1). Δ = method − FullKV.')
 .set_table_styles([{'selector':'caption',
                     'props':[('font-size','13px'),('font-weight','bold'),
                               ('text-align','left'),('margin-bottom','6px')]}])
)

# Multiple-choice accuracy. Scores 0–1.

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(6); w = 0.20
for i, (m, color) in enumerate(zip(METHODS, COLORS)):
    ax.bar(x + (i-1.5)*w, score_v2[m], w, label=m, color=color, alpha=0.88,
           edgecolor='white', linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(V2_SHORT, fontsize=9)
ax.set_ylabel('Score (0–1)')
ax.set_title('Figure 6 — Per-task Score Comparison (LongBench v2)')
ax.legend(framealpha=0.3, fontsize=9); ax.set_ylim(0, 1.12)
plt.tight_layout()
plt.savefig('figures/v2_score_comparison.png', bbox_inches='tight')
plt.show()

# 
# - Dynamic chunk-wise INT4 allocation avoids static pre-allocation OOM.
# - All tokens are retained (quantised, not dropped), preserving the full context.
# - Memory savings (~8 GB) come from INT4 compression, not token eviction.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
x = np.arange(6); w = 0.20; w3 = 0.25

for i, (m, color) in enumerate(zip(METHODS, COLORS)):
    ax1.bar(x + (i-1.5)*w, e2e_v2[m]/1000, w, label=m, color=color, alpha=0.88,
            edgecolor='white', linewidth=0.4)
ax1.set_xticks(x); ax1.set_xticklabels(V2_SHORT, fontsize=9)
ax1.set_ylabel('E2E Latency (s)'); ax1.set_title('Figure 7a — E2E Latency (LBv2, abs)')
ax1.legend(framealpha=0.3, fontsize=9)

for i, (m, color) in enumerate(zip(METHODS3, COLORS3)):
    overheads = e2e_v2[m] / e2e_v2['FullKV']
    ax2.bar(x + (i-1)*w3, overheads, w3, label=m, color=color, alpha=0.88,
            edgecolor='white', linewidth=0.4)
ax2.axhline(1.0, color='k', linewidth=1, linestyle='--', alpha=0.5, label='FullKV')
ax2.set_xticks(x); ax2.set_xticklabels(V2_SHORT, fontsize=9)
ax2.set_ylabel('Overhead vs FullKV'); ax2.set_title('Figure 7b — E2E Overhead (LBv2)')
ax2.legend(framealpha=0.3, fontsize=9)
plt.tight_layout()
plt.savefig('figures/v2_e2e_latency.png', bbox_inches='tight')
plt.show()

# E2E table
df_e2e_v2 = pd.DataFrame({m: {t: e2e_v2[m][i] for i, t in enumerate(V2_TASKS)} for m in METHODS})
df_e2e_v2.index.name = 'Task'
for m in METHODS3:
    df_e2e_v2[f'{m} OH'] = df_e2e_v2[m] / df_e2e_v2['FullKV']
df_e2e_v2.loc['Average'] = df_e2e_v2.mean()
fmt2 = {m: '{:,.0f}' for m in METHODS}
fmt2.update({f'{m} OH': '{:.2f}x' for m in METHODS3})
(df_e2e_v2.style.format(fmt2)
 .background_gradient(subset=[f'{m} OH' for m in METHODS3], cmap='RdYlGn_r', vmin=0.3, vmax=5.0)
 .set_caption('Table 6 — E2E Latency (ms). OH = method / FullKV.')
 .set_table_styles([{'selector':'caption','props':[('font-size','13px'),('font-weight','bold'),
                     ('text-align','left'),('margin-bottom','6px')]}])
)

# KVQuant's E2E appears fast because **many samples OOM and return immediately** (no generation).  
# A failed sample logs `OOM(prefill)`, contributes 0 to the score, and its latency reflects only  
# the failed prefill attempt (often shorter than a complete run). This artificially deflates avg E2E.

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(6); w = 0.20
for i, (m, color) in enumerate(zip(METHODS, COLORS)):
    ax.bar(x + (i-1.5)*w, mem_v2[m]/1024, w, label=m, color=color, alpha=0.88,
           edgecolor='white', linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(V2_SHORT, fontsize=9)
ax.set_ylabel('Memory excl. model weight (GB)')
ax.set_title('Figure 8 — Peak GPU Memory excl. Model Weight (LongBench v2)')
ax.legend(framealpha=0.3, fontsize=9)
plt.tight_layout()
plt.savefig('figures/v2_memory.png', bbox_inches='tight')
plt.show()

# Memory table
df_mem_v2 = pd.DataFrame({m: {t: mem_v2[m][i] for i, t in enumerate(V2_TASKS)} for m in METHODS})
df_mem_v2.index.name = 'Task'
for m in METHODS3:
    df_mem_v2[f'Δ {m} (MB)'] = df_mem_v2[m] - df_mem_v2['FullKV']
df_mem_v2.loc['Average'] = df_mem_v2.mean()
fmt3 = {m: '{:,.0f}' for m in METHODS}
fmt3.update({f'Δ {m} (MB)': '{:+,.0f}' for m in METHODS3})
(df_mem_v2.style.format(fmt3)
 .background_gradient(subset=[f'Δ {m} (MB)' for m in METHODS3], cmap='RdYlGn', vmin=-15000, vmax=5000)
 .set_caption('Table 7 — Peak GPU Memory excl. model weight (MB). Δ = method − FullKV.')
 .set_table_styles([{'selector':'caption','props':[('font-size','13px'),('font-weight','bold'),
                     ('text-align','left'),('margin-bottom','6px')]}])
)

PREFILL_COLORS = {'LiveKVQuantP': '#2d7e4a', 'KVQuant': '#9b1c1c', 'Finch': '#a0522d'}
DECODE_COLORS  = {'LiveKVQuantP': '#a8d5b5', 'KVQuant': '#f5a0a0', 'Finch': '#f5c79e'}
x = np.arange(6)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
for ax, m in zip(axes, METHODS3):
    ax.bar(x, prefill_v2[m]/1000, 0.55, label='Prefill',
           color=PREFILL_COLORS[m], alpha=0.9)
    ax.bar(x, decode_v2[m]/1000, 0.55, bottom=prefill_v2[m]/1000,
           label='Decode', color=DECODE_COLORS[m], alpha=0.9)
    ax.set_title(m, fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(V2_SHORT, fontsize=8)
    ax.set_ylabel('Latency (s)'); ax.legend(fontsize=9, framealpha=0.3)
fig.suptitle('Figure 9 — Prefill vs Decode Breakdown (LongBench v2)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('figures/v2_prefill_decode_breakdown.png', bbox_inches='tight')
plt.show()

# Prefill% summary
for m in METHODS3:
    pct = (prefill_v2[m] / e2e_v2[m] * 100).mean()
    avg_e2e_m = e2e_v2[m].mean() / 1000
    print(f'  {m:18s}: avg prefill% = {pct:.1f}%,  avg E2E = {avg_e2e_m:.1f}s')

avg_score_v2 = {m: score_v2[m].mean() for m in METHODS}
avg_e2e_v2   = {m: e2e_v2[m].mean()/1000 for m in METHODS}
avg_mem_v2   = {m: mem_v2[m].mean()/1024 for m in METHODS}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, metric, vals, ylabel, title, extra in [
    (axes[0], 'score', avg_score_v2, 'Avg Score (0–1)', 'Figure 10a — Avg Score (LBv2)', '{:.3f}'),
    (axes[1], 'e2e',   avg_e2e_v2,   'Avg E2E (s)',     'Figure 10b — Avg E2E (LBv2)',   '{:.1f}s'),
    (axes[2], 'mem',   avg_mem_v2,   'Mem excl. weight (GB)', 'Figure 10c — Avg Memory excl. Weight (LBv2)', '{:.1f}GB'),
]:
    bars = ax.bar(METHODS, [vals[m] for m in METHODS], color=COLORS, alpha=0.88, width=0.5)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for bar, m in zip(bars, METHODS):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                extra.format(vals[m]), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('figures/v2_summary.png', bbox_inches='tight')
plt.show()

# 
# - **LiveKVQuantP** is the only method that generalises to v2 long contexts: −5.9% score vs −48% for KVQuant/Finch.
# - **KVQuant**'s apparent speed (18.8s avg) and low score are both artefacts of OOM: samples > 32 k tokens crash immediately, score = 0, latency = partial prefill time.
# - **Finch** suffers even more from long inputs: with 99.2% of time in prefill, token-dropping gives no latency benefit, and the score collapses due to excessive context eviction.
# - v2 prefill% is much higher than v1 (≥96% vs ~63%) across all methods, confirming that v2 is a **prefill-dominated** workload.
