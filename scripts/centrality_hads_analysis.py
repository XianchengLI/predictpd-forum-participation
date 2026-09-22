"""Engagement subgroup contrasts with HADS anxiety and depression change (Results)."""

import pandas as pd
from scipy import stats

from engagement_common import METRICS, WINDOWS, build_pairs, high_flag, load
from paths import PROC

fp_df, centrality = load()

rows = []
for outcome in ["hads_anxiety", "hads_depression"]:
    for window, max_days in WINDOWS.items():
        d = build_pairs(fp_df, centrality, max_days, outcome=outcome)
        print(f"\n{outcome} / {window}: n = {len(d)}")
        for metric, (rule, low_lab, high_lab) in METRICS.items():
            hi = high_flag(d, metric)
            low, high = d.loc[~hi, "change"], d.loc[hi, "change"]
            _, p = stats.ttest_ind(high, low, equal_var=False)
            rows.append({
                "outcome": outcome, "config": window, "metric": metric,
                "low_label": low_lab, "high_label": high_lab,
                "n_low": len(low), "n_high": len(high),
                "mean_low": low.mean(), "mean_high": high.mean(),
                "diff": high.mean() - low.mean(), "welch_p": p,
            })
            print(f"  {metric:<12} n={len(low)}/{len(high)}  low {low.mean():+.2f}  "
                  f"high {high.mean():+.2f}  Welch p={p:.3f}")

out = pd.DataFrame(rows)
out.to_csv(PROC / "centrality_hads_comparison.csv", index=False)
print(f"\nSmallest Welch p across all HADS contrasts: {out['welch_p'].min():.3f}")
print("Saved: centrality_hads_comparison.csv")
