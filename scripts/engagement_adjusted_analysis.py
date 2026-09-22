"""Covariate-adjusted engagement contrasts and the Spearman correlation matrix (MA6)."""

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from engagement_common import METRICS, WINDOWS, build_pairs, high_flag, load
from paths import PROC

fp_df, centrality = load()

results = []
nowin_sample = None
for window, max_days in WINDOWS.items():
    d = build_pairs(fp_df, centrality, max_days)
    if window == "No window":
        nowin_sample = d.copy()
    print(f"\n{'=' * 78}\n{window}: n={len(d)} FP "
          f"(age available {d['age_at_cutoff'].notna().sum()})\n{'=' * 78}")
    for metric, (rule, low_lab, high_lab) in METRICS.items():
        d["high"] = high_flag(d, metric).astype(int)
        sub = d.dropna(subset=["change", "pre_value", "age_at_cutoff", "interval_y"])
        lo_v, hi_v = sub.loc[sub["high"] == 0, "change"], sub.loc[sub["high"] == 1, "change"]
        _, p_welch = stats.ttest_ind(hi_v, lo_v, equal_var=False)
        m = smf.ols("change ~ high + pre_value + age_at_cutoff + interval_y", data=sub).fit()
        row = {
            "config": window, "metric": metric, "low_label": low_lab, "high_label": high_lab,
            "n_low": len(lo_v), "n_high": len(hi_v),
            "unadj_diff": hi_v.mean() - lo_v.mean(), "welch_p": p_welch,
            "adj_diff": m.params["high"], "adj_se": m.bse["high"],
            "adj_ci_lo": m.conf_int().loc["high", 0], "adj_ci_hi": m.conf_int().loc["high", 1],
            "adj_p": m.pvalues["high"],
            "beta_baseline": m.params["pre_value"], "p_baseline": m.pvalues["pre_value"],
        }
        results.append(row)
        print(f"  {metric:<12} n={len(lo_v)}/{len(hi_v):<4} unadj {row['unadj_diff']:+.3f} "
              f"(Welch p={p_welch:.3f})  adj {row['adj_diff']:+.3f} "
              f"[{row['adj_ci_lo']:+.2f}, {row['adj_ci_hi']:+.2f}] p={row['adj_p']:.3f}  "
              f"baseline coef {row['beta_baseline']:+.3f} (p={row['p_baseline']:.2f})")

pd.DataFrame(results).to_csv(PROC / "engagement_adjusted_results.csv", index=False)

mcols = list(METRICS)
corr = pd.DataFrame(index=mcols, columns=mcols, dtype=float)
for a in mcols:
    for b in mcols:
        corr.loc[a, b] = stats.spearmanr(nowin_sample[a], nowin_sample[b])[0]
print(f"\nSpearman correlations, no-window sample (n={len(nowin_sample)}):")
print(corr.round(2).to_string())
corr.round(4).to_csv(PROC / "engagement_correlation_matrix.csv")
print("\nSaved: engagement_adjusted_results.csv, engagement_correlation_matrix.csv")
