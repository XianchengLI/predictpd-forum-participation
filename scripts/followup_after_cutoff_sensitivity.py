"""Primary comparison restricted to participants whose follow-up postdates the cutoff (Results; MA6)."""

import pandas as pd
import statsmodels.formula.api as smf

from paths import PROC

OUTCOMES = ["risk_score", "hads_anxiety", "hads_depression"]

def to_long(pairs):
    rows = []
    for r in pairs.itertuples(index=False):
        base = {"pid": r.HashFinal, "group": r.group, "interval_days": r.interval_days}
        rows.append({**base, "time": 0, "y": r.pre_transformed})
        rows.append({**base, "time": 1, "y": r.post_transformed})
    d = pd.DataFrame(rows)
    d["group"] = pd.Categorical(d["group"], categories=["NP", "FP"])
    d["interval_centered"] = d["interval_days"] - d["interval_days"].mean()
    return d

rows = []
for config in ["primary_clean", "eligible_clean"]:
    for outcome in OUTCOMES:
        pairs = pd.read_csv(
            PROC / f"preforum_pairs_{config}_{outcome}.csv", parse_dates=["post_date", "cutoff"]
        )
        spans = pairs["post_date"] > pairs["cutoff"].dt.normalize()
        n_before = pairs.loc[~spans].groupby("group").size()
        kept = pairs[spans]
        d = to_long(kept)
        for model, formula in [
            ("M1", "y ~ time * group"),
            ("M2", "y ~ time * group + interval_centered + time:interval_centered"),
        ]:
            m = smf.mixedlm(formula, data=d, groups=d["pid"]).fit()
            k = "time:group[T.FP]"
            rows.append({
                "config": config, "outcome": outcome, "model": model,
                "n_fp": int((kept["group"] == "FP").sum()),
                "n_np": int((kept["group"] == "NP").sum()),
                "excluded_fp_followup_before_cutoff": int(n_before.get("FP", 0)),
                "excluded_np_followup_before_cutoff": int(n_before.get("NP", 0)),
                "beta3": m.fe_params[k], "se": m.bse[k],
                "ci_lo": m.fe_params[k] - 1.96 * m.bse[k],
                "ci_hi": m.fe_params[k] + 1.96 * m.bse[k],
                "p": m.pvalues[k],
            })
            r = rows[-1]
            print(f"[{config} | {outcome} | {model}] nFP={r['n_fp']} nNP={r['n_np']} "
                  f"(excluded {r['excluded_fp_followup_before_cutoff']} FP, "
                  f"{r['excluded_np_followup_before_cutoff']} NP)  "
                  f"beta3={r['beta3']:+.4f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] p={r['p']:.4f}")

out = pd.DataFrame(rows)
out.to_csv(PROC / "followup_after_cutoff_sensitivity.csv", index=False)
print("\nSaved followup_after_cutoff_sensitivity.csv")
