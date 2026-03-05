"""
A3_merge_and_analyze.py
Merge E1/E1b (indices 0-49) and E1aug (indices 50-99) into n=100 dataset.
Compute peak CIF rates, 95% Wilson CIs, and GAF.

Usage:
    python A3_merge_and_analyze.py --e1b data/E1_E1b/combined_summary.json \
                                    --aug data/E1aug/summary.json \
                                    --output data/merged_n100.json
"""

import json
import math
import argparse


def wilson_ci(k, n, z=1.96):
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - margin), min(1, center + margin)


def merge_and_analyze(e1b_path, aug_path, output_path):
    with open(e1b_path) as f:
        orig = json.load(f)
    with open(aug_path) as f:
        aug = json.load(f)

    MODELS = ["sonnet4", "gpt4o", "haiku35", "gpt4omini", "gpt35"]
    DOMAINS = ["gsm8k", "csqa", "boolq"]
    LAMBDAS = ["0.0", "0.4", "0.8", "1.0"]

    merged = {}
    for m in MODELS:
        merged[m] = {}
        for d in DOMAINS:
            o_ls = orig[m][d]["lambda_sweep"]
            a_ls = aug[m][d]["lambda_sweep"]

            o_base = orig[m][d]["baseline_accuracy"]
            a_base = aug[m][d]["baseline_accuracy"]
            merged_base = (o_base * 50 + a_base * 50) / 100

            lam_data = {}
            for lam in LAMBDAS:
                if lam in o_ls and lam in a_ls:
                    n_elig = o_ls[lam]["n_eligible"] + a_ls[lam]["n_eligible"]
                    n_cif = o_ls[lam]["n_cif"] + a_ls[lam]["n_cif"]
                    rate, lo, hi = wilson_ci(n_cif, n_elig)
                    lam_data[lam] = {
                        "cif_rate": rate,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_eligible": n_elig,
                        "n_cif": n_cif,
                    }

            peak_lam = max(lam_data.keys(), key=lambda l: lam_data[l]["cif_rate"])
            peak = lam_data[peak_lam]

            merged[m][d] = {
                "baseline": merged_base,
                "lambda_sweep": lam_data,
                "peak_cif": peak["cif_rate"],
                "peak_ci_lo": peak["ci_lo"],
                "peak_ci_hi": peak["ci_hi"],
                "peak_n_eligible": peak["n_eligible"],
                "peak_lambda": peak_lam,
            }

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Print summary
    print(f"{'Model':15s} {'Domain':8s} {'CIF':>7s} {'95% CI':>18s} {'n':>5s}")
    print("-" * 58)
    for m in MODELS:
        for d in DOMAINS:
            md = merged[m][d]
            print(
                f"{m:15s} {d:8s} {md['peak_cif']:6.1%}  "
                f"[{md['peak_ci_lo']:5.1%}, {md['peak_ci_hi']:5.1%}]  "
                f"n={md['peak_n_eligible']:3d}"
            )

    # GAF
    print("\nGap Amplification (vs Sonnet 4):")
    for d in DOMAINS:
        s4 = max(merged["sonnet4"][d]["peak_cif"], 0.001)
        for m in MODELS:
            if m != "sonnet4":
                gaf = merged[m][d]["peak_cif"] / s4
                print(f"  {d:6s} {m:15s}: {gaf:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1b", required=True)
    parser.add_argument("--aug", required=True)
    parser.add_argument("--output", default="merged_n100.json")
    args = parser.parse_args()
    merge_and_analyze(args.e1b, args.aug, args.output)
