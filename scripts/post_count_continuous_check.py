"""Post count as an ordered variable (MA6)."""

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from engagement_common import WINDOWS, build_pairs, load
from paths import PROC

fp_df, centrality = load()

tests, by_value = [], []
for window, max_days in WINDOWS.items():
    d = build_pairs(fp_df, centrality, max_days)
    rho, p_rho = stats.spearmanr(d["post_count"], d["change"])
    tests.append({"config": window, "n": len(d), "test": "Spearman rho", "est": rho, "p": p_rho})

    m = smf.ols("change ~ post_count", data=d).fit()
    ci = m.conf_int().loc["post_count"]
    tests.append({"config": window, "n": int(m.nobs), "test": "OLS per post",
                  "est": m.params["post_count"], "ci_lo": ci[0], "ci_hi": ci[1],
                  "p": m.pvalues["post_count"]})

    multi = d[d["post_count"] >= 2]
    rho2, p2 = stats.spearmanr(multi["post_count"], multi["change"])
    tests.append({"config": window, "n": len(multi), "test": "Spearman rho, 2+ posters only",
                  "est": rho2, "p": p2})

    g = d.assign(pc_cap=d["post_count"].clip(upper=4))  # 1, 2, 3, 4+
    g = g.groupby("pc_cap")["change"].agg(["size", "mean"]).reset_index()
    g.insert(0, "config", window)
    by_value.append(g)

tests = pd.DataFrame(tests)
by_value = pd.concat(by_value, ignore_index=True)
tests.to_csv(PROC / "post_count_continuous_check.csv", index=False)
by_value.to_csv(PROC / "post_count_by_value.csv", index=False)

pd.set_option("display.width", 160)
print(tests.round(4).to_string(index=False))
print()
print(by_value.round(3).to_string(index=False))
