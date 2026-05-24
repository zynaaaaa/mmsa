#!/usr/bin/env python3
"""
Extract and summarize experiment results from log files
"""
import re
import sys
from pathlib import Path
import numpy as np


ALL_METRICS = [
    'Has0_acc_2', 'Has0_F1_score',
    'Non0_acc_2', 'Non0_F1_score',
    'Mult_acc_5', 'Mult_acc_7',
    'MAE', 'Corr',
]


def _parse_value(raw):
    """Parse a value string that may be nan, np.float64(...), or plain number."""
    raw = raw.strip()
    if raw == 'nan':
        return float('nan')
    return float(raw)


def extract_seed_results(log_file):
    """Extract results for each seed from log file"""
    results = {}

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Match either "Seed N result: {...}" or "Result for seed N: {...}"
        m = re.search(r"(?:Seed (\d+) result|Result for seed (\d+)):\s*\{(.+)\}", line)
        if not m:
            continue

        seed = int(m.group(1) if m.group(1) is not None else m.group(2))
        dict_str = m.group(3)

        row = {}
        for metric in ALL_METRICS:
            # Match 'metric': value  or  'metric': np.float64(value)
            p = re.search(
                rf"'{metric}':\s*(?:[\w.]+\()?([\d.eE+-]+|nan)\)?",
                dict_str
            )
            if p:
                row[metric] = _parse_value(p.group(1))
            else:
                row[metric] = float('nan')

        results[seed] = row

    return results


def calculate_statistics(results):
    """Calculate mean and std from seed results, skipping nan."""
    if not results:
        return None

    stats = {}
    for metric in ALL_METRICS:
        values = [r[metric] for r in results.values() if not np.isnan(r[metric])]
        stats[metric] = {
            'mean': np.mean(values) if values else float('nan'),
            'std': np.std(values) if values else float('nan'),
            'n': len(values),
        }
    return stats


def is_failed(row):
    corr = row.get('Corr', float('nan'))
    acc = row.get('Has0_acc_2', row.get('Non0_acc_2', 1.0))
    return np.isnan(corr) or corr < 0.3 or acc < 0.6


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <results_directory>")
        print("Example: python summarize_results.py experiment_results")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    log_files = sorted(results_dir.glob("*.log"))

    if not log_files:
        print(f"No log files found in {results_dir}")
        sys.exit(1)

    print("=" * 120)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 120)
    print()

    all_results = {}

    for log_file in log_files:
        model_name = log_file.stem.rsplit('_', 1)[0]

        seed_results = extract_seed_results(log_file)

        if not seed_results:
            print(f"Model: {model_name}")
            print("-" * 120)
            print("  Warning: No results found in log file")
            print()
            continue

        print(f"Model: {model_name}")
        print("-" * 120)
        print(f"  Seeds: {sorted(seed_results.keys())} ({len(seed_results)} total)")

        # Header
        hdr = f"    {'Seed':>6}"
        for metric in ALL_METRICS:
            hdr += f"  {metric:>15}"
        hdr += "  Status"
        print(hdr)

        for seed in sorted(seed_results.keys()):
            r = seed_results[seed]
            failed = is_failed(r)
            row_str = f"    {seed:>6}"
            for metric in ALL_METRICS:
                v = r[metric]
                if np.isnan(v):
                    row_str += f"  {'nan':>15}"
                else:
                    row_str += f"  {v:>15.4f}"
            if failed:
                row_str += "  <-- FAILED"
            print(row_str)

        stats = calculate_statistics(seed_results)

        print()
        print("  Mean +/- Std:")
        for metric in ALL_METRICS:
            mean = stats[metric]['mean']
            std = stats[metric]['std']
            n = stats[metric]['n']
            if np.isnan(mean):
                print(f"    {metric:20s}: nan")
            else:
                print(f"    {metric:20s}: {mean:.4f} +/- {std:.4f}  (n={n})")

        failures = [seed for seed, r in seed_results.items() if is_failed(r)]
        if failures:
            print()
            print(f"  Warning: Failed seeds: {failures}")
            print(f"  Failure rate: {len(failures)}/{len(seed_results)} "
                  f"({100*len(failures)/len(seed_results):.1f}%)")

        print()
        all_results[model_name] = {
            'seeds': seed_results,
            'stats': stats
        }

    if not all_results:
        print("No results to summarize.")
        return

    # Comparison table
    paper_metrics = ['Has0_acc_2', 'Has0_F1_score', 'Mult_acc_7', 'MAE', 'Corr']

    print("=" * 120)
    print("COMPARISON TABLE (paper metrics)")
    print("=" * 120)
    print()

    header = f"{'Model':<25}"
    for metric in paper_metrics:
        header += f"  {metric:>20}"
    print(header)
    print("-" * len(header))

    for model_name in sorted(all_results.keys()):
        stats = all_results[model_name]['stats']
        name = model_name.replace('_', ' ').title()
        row = f"{name:<25}"
        for metric in paper_metrics:
            mean = stats[metric]['mean']
            std = stats[metric]['std']
            if np.isnan(mean):
                row += f"  {'nan':>20}"
            else:
                row += f"  {mean:.4f}+/-{std:.4f}"
        print(row)

    # LaTeX table
    print()
    print("=" * 120)
    print("LATEX TABLE")
    print("=" * 120)
    print()

    ncols = len(paper_metrics)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l" + "c" * ncols + "}")
    print("\\hline")
    col_names = " & ".join(["Model"] + [m.replace('_', '\\_') for m in paper_metrics]) + " \\\\"
    print(col_names)
    print("\\hline")

    for model_name in sorted(all_results.keys()):
        stats = all_results[model_name]['stats']
        display_name = model_name.replace('_', ' ').title()
        cells = [display_name]
        for metric in paper_metrics:
            mean = stats[metric]['mean']
            std = stats[metric]['std']
            if np.isnan(mean):
                cells.append("--")
            else:
                cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        print(" & ".join(cells) + " \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Experimental results on MOSI dataset.}")
    print("\\end{table}")
    print()


if __name__ == "__main__":
    main()
