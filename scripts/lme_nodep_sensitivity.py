"""LME on the depression-excluded risk score, same pairs as the primary analysis (MA6)."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from paths import PROC

nodep = pd.read_csv(PROC / "riskscore_final_nodep.csv").set_index("HashFinal")

rows = []
for config in ["primary_clean", "eligible_clean"]:
    pairs = pd.read_csv(PROC / f"preforum_pairs_{config}_risk_score.csv")

    def nodep_val(pid, session):
        try:
            return nodep.at[pid, f"riskscore_final_nodep_{int(session)}"]
        except KeyError:
            return np.nan

    pairs["pre_nodep"] = [nodep_val(p, s) for p, s in zip(pairs["HashFinal"], pairs["pre_session"])]
    pairs["post_nodep"] = [nodep_val(p, s) for p, s in zip(pairs["HashFinal"], pairs["post_session"])]
    n_missing = int(pairs["pre_nodep"].isna().sum() + pairs["post_nodep"].isna().sum())
    pairs = pairs.dropna(subset=["pre_nodep", "post_nodep"])
    pairs = pairs[(pairs["pre_nodep"] > 0) & (pairs["post_nodep"] > 0)]

    long_rows = []
    for r in pairs.itertuples(index=False):
        base_rec = {"pid": r.HashFinal, "group": r.group, "interval_days": r.interval_days}
        long_rows.append({**base_rec, "time": 0, "y": np.log(r.pre_nodep)})
        long_rows.append({**base_rec, "time": 1, "y": np.log(r.post_nodep)})
    d = pd.DataFrame(long_rows)
    d["group"] = pd.Categorical(d["group"], categories=["NP", "FP"])
    d["interval_centered"] = d["interval_days"] - d["interval_days"].mean()

    for label, formula in [
        ("M1", "y ~ time * group"),
        ("M2", "y ~ time * group + interval_centered + time:interval_centered"),
    ]:
        m = smf.mixedlm(formula, data=d, groups=d["pid"]).fit()
        key = "time:group[T.FP]"
        n_fp = int((pairs["group"] == "FP").sum())
        n_np = int((pairs["group"] == "NP").sum())
        rows.append({
            "config": config, "model": label, "n_fp": n_fp, "n_np": n_np,
            "n_cells_missing_nodep": n_missing,
            "beta3": m.fe_params[key], "se": m.bse[key],
            "ci_lo": m.fe_params[key] - 1.96 * m.bse[key],
            "ci_hi": m.fe_params[key] + 1.96 * m.bse[key],
            "p": m.pvalues[key],
        })
        print(f"[{config} | {label}] nFP={n_fp} nNP={n_np} "
              f"(missing nodep cells: {n_missing})  "
              f"beta3={m.fe_params[key]:+.4f} "
              f"[{rows[-1]['ci_lo']:+.4f}, {rows[-1]['ci_hi']:+.4f}]  "
              f"p={m.pvalues[key]:.4f}")

out = pd.DataFrame(rows)
out.to_csv(PROC / "lme_nodep_sensitivity.csv", index=False)
print("\nSaved lme_nodep_sensitivity.csv")
print("Reference (full score): primary M1 beta3=-0.463 p=0.016, M2 -0.480 p=0.013; "
      "eligible M1 -0.328 p=0.065")
