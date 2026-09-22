"""Engagement contrasts under closest vs earliest/latest session pairs (MA6)."""

import pandas as pd
from scipy import stats

from engagement_common import METRICS, WINDOWS, build_pairs, high_flag, load
from paths import PROC

fp_df, centrality = load()

rows = []
for window, max_days in WINDOWS.items():
    d = build_pairs(fp_df, centrality, max_days)
    d["closest"] = d["change"]
    d["widest"] = d["widest_change"]
    affected = int(((d["n_pre_candidates"] > 1) | (d["n_post_candidates"] > 1)).sum())
    for metric in METRICS:
        hi = high_flag(d, metric)
        for rule in ("closest", "widest"):
            lo_v, hi_v = d.loc[~hi, rule], d.loc[hi, rule]
            _, p = stats.ttest_ind(hi_v, lo_v, equal_var=False)
            rows.append({
                "config": window, "metric": metric, "pair_rule": rule,
                "n": len(d), "n_pairs_affected": affected,
                "n_low": len(lo_v), "n_high": len(hi_v),
                "mean_low": lo_v.mean(), "mean_high": hi_v.mean(),
                "diff": hi_v.mean() - lo_v.mean(), "welch_p": p,
            })

out = pd.DataFrame(rows)
out.to_csv(PROC / "engagement_pair_rule_sensitivity.csv", index=False)
pd.set_option("display.width", 170)
print(out.round(3).to_string(index=False))
