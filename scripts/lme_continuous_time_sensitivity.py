"""Model 1 with elapsed time in years instead of the 0/1 indicator (Results; MA6)."""

import pandas as pd
import statsmodels.formula.api as smf

from paths import PROC

RUNS = [
    ("primary_clean", "risk_score"),
    ("primary_clean", "hads_anxiety"),
    ("primary_clean", "hads_depression"),
    ("eligible_clean", "risk_score"),
]

rows = []
for config, outcome in RUNS:
    pairs = pd.read_csv(PROC / f"preforum_pairs_{config}_{outcome}.csv")
    long = []
    for r in pairs.itertuples(index=False):
        base = {"pid": r.HashFinal, "group": r.group}
        long.append({**base, "t_years": 0.0, "y": r.pre_transformed})
        long.append({**base, "t_years": r.interval_days / 365.25, "y": r.post_transformed})
    d = pd.DataFrame(long)
    d["group"] = pd.Categorical(d["group"], categories=["NP", "FP"])
    m = smf.mixedlm("y ~ t_years * group", data=d, groups=d["pid"]).fit()
    k = "t_years:group[T.FP]"
    ci = m.conf_int().loc[k]
    rows.append({
        "config": config, "outcome": outcome,
        "n_fp": int((pairs["group"] == "FP").sum()), "n_np": int((pairs["group"] == "NP").sum()),
        "mean_interval_years": pairs["interval_days"].mean() / 365.25,
        "coef_per_year": m.fe_params[k], "se": m.bse[k], "ci_lo": ci[0], "ci_hi": ci[1],
        "p": m.pvalues[k], "converged": bool(m.converged),
    })
    r = rows[-1]
    print(f"[{config} | {outcome}] nFP={r['n_fp']} nNP={r['n_np']}  per-year diff "
          f"{r['coef_per_year']:+.4f} (95% CI {r['ci_lo']:+.4f} to {r['ci_hi']:+.4f}, "
          f"p={r['p']:.4f}); mean interval {r['mean_interval_years']:.2f} y")

pd.DataFrame(rows).to_csv(PROC / "lme_continuous_time_sensitivity.csv", index=False)
print(f"\nSaved {PROC / 'lme_continuous_time_sensitivity.csv'}")
