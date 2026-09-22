"""Engagement subgroup contrasts (Table 5) with Welch CIs, plus the Methods descriptives and the age check."""

import pandas as pd
from scipy import stats

from engagement_common import METRICS, WINDOWS, build_pairs, high_flag, load
from paths import PROC

fp_df, centrality = load()

rows, desc = [], []
for window, max_days in WINDOWS.items():
    d = build_pairs(fp_df, centrality, max_days)
    print(f"\n{'=' * 70}\n{window}: n = {len(d)} FP with a pre/post risk-score pair\n{'=' * 70}")
    for metric, (rule, low_lab, high_lab) in METRICS.items():
        hi = high_flag(d, metric)
        low, high = d.loc[~hi, "change"], d.loc[hi, "change"]
        res = stats.ttest_ind(high, low, equal_var=False)  # Welch
        ci = res.confidence_interval(0.95)
        rows.append({
            "config": window, "metric": metric, "rule": rule,
            "low_label": low_lab, "high_label": high_lab,
            "n_low": len(low), "n_high": len(high),
            "mean_low": low.mean(), "mean_high": high.mean(),
            "diff": high.mean() - low.mean(),
            "welch_ci_lo": ci.low, "welch_ci_hi": ci.high, "welch_p": res.pvalue,
        })
        print(f"  {metric:<12} {low_lab} n={len(low):>3} mean={low.mean():+.2f} | "
              f"{high_lab} n={len(high):>3} mean={high.mean():+.2f} | "
              f"diff {high.mean() - low.mean():+.2f} [{ci.low:+.2f}, {ci.high:+.2f}] "
              f"Welch p={res.pvalue:.3f}")

    desc.append({"config": window, "statistic": "n", "value": len(d)})
    desc.append({"config": window, "statistic": "pct_posted_once",
                 "value": 100 * (d["post_count"] == 1).mean()})
    desc.append({"config": window, "statistic": "n_distinct_post_count",
                 "value": d["post_count"].nunique()})
    for m in ["in_degree", "betweenness", "clustering"]:
        desc.append({"config": window, "statistic": f"pct_zero_{m}",
                     "value": 100 * (d[m] == 0).mean()})
    desc.append({"config": window, "statistic": "closeness_median",
                 "value": d["closeness"].median()})
    desc.append({"config": window, "statistic": "closeness_p75",
                 "value": d["closeness"].quantile(0.75)})
    hi = high_flag(d, "post_count")
    a_lo, a_hi = d.loc[~hi, "age_at_cutoff"].dropna(), d.loc[hi, "age_at_cutoff"].dropna()
    desc.append({"config": window, "statistic": "age_mean_1_post", "value": a_lo.mean()})
    desc.append({"config": window, "statistic": "age_mean_2plus_posts", "value": a_hi.mean()})
    desc.append({"config": window, "statistic": "age_welch_p",
                 "value": stats.ttest_ind(a_hi, a_lo, equal_var=False)[1]})

pd.DataFrame(rows).to_csv(PROC / "centrality_temporal_comparison.csv", index=False)
pd.DataFrame(desc).to_csv(PROC / "engagement_descriptives.csv", index=False)
print("\nDescriptives:")
print(pd.DataFrame(desc).round(3).to_string(index=False))
print("\nSaved: centrality_temporal_comparison.csv, engagement_descriptives.csv")
