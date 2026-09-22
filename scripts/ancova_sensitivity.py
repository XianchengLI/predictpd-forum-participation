"""ANCOVA specification: follow-up ~ baseline + group (+ interval) (MA6)."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from paths import PROC

RUNS = [
    ("primary_clean", "risk_score"),
    ("primary_clean", "hads_anxiety"),
    ("primary_clean", "hads_depression"),
    ("eligible_clean", "risk_score"),
]

lme = pd.read_csv(PROC / "preforum_main_results.csv").set_index(["config", "outcome"])

rows = []
for config, outcome in RUNS:
    pairs = pd.read_csv(PROC / f"preforum_pairs_{config}_{outcome}.csv")
    d = pairs[["group", "pre_transformed", "post_transformed", "interval_days"]].copy()
    d["group"] = pd.Categorical(d["group"], categories=["NP", "FP"])
    d["interval_c"] = d["interval_days"] - d["interval_days"].mean()

    for label, formula in [
        ("A1", "post_transformed ~ pre_transformed + group"),
        ("A2", "post_transformed ~ pre_transformed + group + interval_c"),
    ]:
        m = smf.ols(formula, data=d).fit()
        k = "group[T.FP]"
        rows.append({
            "config": config, "outcome": outcome, "model": label,
            "n_fp": int((d["group"] == "FP").sum()),
            "n_np": int((d["group"] == "NP").sum()),
            "beta_group": m.params[k], "se": m.bse[k],
            "ci_lo": m.conf_int().loc[k, 0], "ci_hi": m.conf_int().loc[k, 1],
            "p": m.pvalues[k],
            "beta_baseline": m.params["pre_transformed"],
            "lme_m1_beta3": lme.loc[(config, outcome), "m1_beta3"],
            "lme_m1_p": lme.loc[(config, outcome), "m1_p"],
            "lme_m2_beta3": lme.loc[(config, outcome), "m2_beta3"],
            "lme_m2_p": lme.loc[(config, outcome), "m2_p"],
        })
        print(f"[{config} | {outcome} | {label}] "
              f"nFP={rows[-1]['n_fp']} nNP={rows[-1]['n_np']}  "
              f"ANCOVA group beta={m.params[k]:+.4f} "
              f"(SE {m.bse[k]:.4f}, 95% CI {rows[-1]['ci_lo']:+.4f} to "
              f"{rows[-1]['ci_hi']:+.4f}, p={m.pvalues[k]:.4f})  "
              f"baseline coef={m.params['pre_transformed']:.3f}  "
              f"| LME M1 b3={rows[-1]['lme_m1_beta3']:+.4f} (p={rows[-1]['lme_m1_p']:.4f})")

out = pd.DataFrame(rows)
out.to_csv(PROC / "ancova_sensitivity.csv", index=False)
print(f"\nSaved to {PROC / 'ancova_sensitivity.csv'}")
