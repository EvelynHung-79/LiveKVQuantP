#!/usr/bin/env python3
"""Compare fullKV and LiveKVQuant results side by side, save to overall_result.json."""

import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FULLKV_DIR = os.path.join(RESULTS_DIR, "baselines", "fullKV")
LIVEKV_DIR = os.path.join(RESULTS_DIR, "liveKVQuant")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "./results/overall_result.json")


def extract_task_name(filename):
    """Extract task name from filename like '20260304_2048_v1_narrativeqa_runpod.json'."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'^\d{8}_\d{4}_', '', name)
    name = re.sub(r'^v\d+_', '', name)
    name = re.sub(r'(_runpod|_2warmupchunk)+$', '', name)
    return name


def load_results(directory):
    """Load all JSON result files from a directory, keyed by task name."""
    results = {}
    if not os.path.isdir(directory):
        return results
    for f in os.listdir(directory):
        if not f.endswith('.json'):
            continue
        path = os.path.join(directory, f)
        with open(path) as fp:
            data = json.load(fp)
        if 'avg_score' not in data:
            continue
        task = extract_task_name(f)
        if task not in results or f > results[task]['_filename']:
            data['_filename'] = f
            results[task] = data
    return results


def main():
    fullkv = load_results(FULLKV_DIR)
    livekv = load_results(LIVEKV_DIR)

    all_tasks = sorted(set(fullkv.keys()) | set(livekv.keys()))

    categories = [
        ("Single-Doc QA", ["narrativeqa", "qasper", "multifieldqa_en"]),
        ("Multi-Doc QA", ["hotpotqa", "2wikimqa", "musique"]),
        ("Summarization", ["gov_report", "qmsum", "multi_news"]),
        ("Few-shot", ["trec", "triviaqa", "samsum"]),
        ("Synthetic", ["passage_count", "passage_retrieval_en"]),
        ("Code", ["lcc", "repobench-p"]),
    ]

    task_to_cat = {}
    for cat, tasks in categories:
        for t in tasks:
            task_to_cat[t] = cat

    ordered_tasks = []
    for cat, tasks in categories:
        for t in tasks:
            if t in fullkv or t in livekv:
                ordered_tasks.append(t)
    for t in all_tasks:
        if t not in task_to_cat:
            ordered_tasks.append(t)

    # Build per-task comparison
    per_task = []
    for task in ordered_tasks:
        fk = fullkv.get(task)
        lk = livekv.get(task)

        entry = {
            "task": task,
            "category": task_to_cat.get(task, "Other"),
            "fullKV": {
                "score": round(fk['avg_score'] * 100, 2) if fk else None,
                "latency_ms": round(fk['avg_latency_ms'], 1) if fk and 'avg_latency_ms' in fk else None,
                "max_memory_mb": round(fk['max_peak_memory_mb'], 1) if fk and 'max_peak_memory_mb' in fk else None,
            },
            "LiveKVQuant": {
                "score": round(lk['avg_score'] * 100, 2) if lk else None,
                "latency_ms": round(lk['avg_latency_ms'], 1) if lk and 'avg_latency_ms' in lk else None,
                "max_memory_mb": round(lk['max_peak_memory_mb'], 1) if lk and 'max_peak_memory_mb' in lk else None,
            },
            "diff_score": round((lk['avg_score'] - fk['avg_score']) * 100, 2) if fk and lk else None,
        }
        per_task.append(entry)

    # Overall averages (only tasks with both results)
    both_tasks = [t for t in ordered_tasks if t in fullkv and t in livekv]
    n = len(both_tasks)
    overall = {}
    if n > 0:
        overall = {
            "num_common_tasks": n,
            "fullKV_avg_score": round(sum(fullkv[t]['avg_score'] for t in both_tasks) / n * 100, 2),
            "LiveKVQuant_avg_score": round(sum(livekv[t]['avg_score'] for t in both_tasks) / n * 100, 2),
            "diff_avg_score": round((sum(livekv[t]['avg_score'] for t in both_tasks) - sum(fullkv[t]['avg_score'] for t in both_tasks)) / n * 100, 2),
            "fullKV_avg_latency_ms": round(sum(fullkv[t].get('avg_latency_ms', 0) for t in both_tasks) / n, 1),
            "LiveKVQuant_avg_latency_ms": round(sum(livekv[t].get('avg_latency_ms', 0) for t in both_tasks) / n, 1),
            "fullKV_avg_memory_mb": round(sum(fullkv[t].get('max_peak_memory_mb', 0) for t in both_tasks) / n, 1),
            "LiveKVQuant_avg_memory_mb": round(sum(livekv[t].get('max_peak_memory_mb', 0) for t in both_tasks) / n, 1),
        }

    # Per-category averages
    per_category = []
    for cat, cat_tasks in categories:
        ct = [t for t in cat_tasks if t in fullkv and t in livekv]
        if not ct:
            continue
        cn = len(ct)
        per_category.append({
            "category": cat,
            "num_tasks": cn,
            "fullKV_avg_score": round(sum(fullkv[t]['avg_score'] for t in ct) / cn * 100, 2),
            "LiveKVQuant_avg_score": round(sum(livekv[t]['avg_score'] for t in ct) / cn * 100, 2),
            "diff_avg_score": round((sum(livekv[t]['avg_score'] for t in ct) - sum(fullkv[t]['avg_score'] for t in ct)) / cn * 100, 2),
        })

    output = {
        "description": "fullKV vs LiveKVQuant comparison. Score = avg_score * 100 (higher is better). diff_score = LiveKVQuant - fullKV.",
        "overall": overall,
        "per_category": per_category,
        "per_task": per_task,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
