"""Piecewise models with a random slope, LR tests, sessions per participant and the stability check (MA5)."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from paths import NP_CUTOFF as NP_MEDIAN
from paths import PROC as OUT
from paths import first_post_dates, read_cohort

SESSIONS = list(range(9))
PW = {
    "risk_score": ("predictpd_riskscore_final_2024_{}", [0, 1, 2, 3, 4, 5, 7, 8], "log"),
    "hads_anxiety": ("hads_anxiety_score_final_{}", SESSIONS, "none"),
    "hads_depression": ("hads_depression_score_final_{}", SESSIONS, "none"),
}
FORMULA = "outcome ~ pre_time + post_time + group + pre_time:group + post_time:group"
MODELS = {"RI": None, "RS1": "~years_from_cutoff"}
KEYS = {"pre": "pre_time:group[T.FP]", "post": "post_time:group[T.FP]"}
OPTIMISERS = [None, "lbfgs", "powell"]

needed = {"HashFinal"}
for i in SESSIONS:
    needed.add(f"date_session_{i}")
for tmpl, sess, _ in PW.values():
    for s in sess:
        needed.add(tmpl.format(s))

df = read_cohort(needed)
first_post = first_post_dates()
df["first_post"] = df["HashFinal"].map(first_post)
df["is_fp"] = df["first_post"].notna()
df["cutoff_own"] = df["first_post"].fillna(NP_MEDIAN)

def build_long(outcome):
    tmpl, sess, tr = PW[outcome]
    roster = pd.read_csv(OUT / f"preforum_pairs_primary_clean_{outcome}.csv")
    ids = set(roster["HashFinal"])
    records = []
    for s in sess:
        sub = df[["HashFinal", "is_fp", "cutoff_own", f"date_session_{s}", tmpl.format(s)]].copy()
        sub.columns = ["pid", "is_fp", "cutoff", "sess_date", "value"]
        sub = sub.dropna(subset=["sess_date", "value"])
        if tr == "log":
            sub = sub[sub["value"] > 0]
            sub["outcome"] = np.log(sub["value"])
        else:
            sub["outcome"] = sub["value"]
        records.append(sub)
    long = pd.concat(records, ignore_index=True)
    long = long[long["pid"].isin(ids)].copy()
    long["t"] = (long["sess_date"] - long["cutoff"]).dt.days / 365.25
    long["pre_time"] = long["t"].clip(upper=0.0)
    long["post_time"] = long["t"].clip(lower=0.0)
    long["years_from_cutoff"] = long["pre_time"] + long["post_time"]
    long["group"] = pd.Categorical(
        long["is_fp"].map({True: "FP", False: "NP"}), categories=["NP", "FP"]
    )
    return long

def fit(long, re_formula, reml):
    last = None
    for method in OPTIMISERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = smf.mixedlm(
                    FORMULA, data=long, groups=long["pid"], re_formula=re_formula
                )
                kw = {"reml": reml}
                if method is not None:
                    kw["method"] = method
                res = model.fit(**kw)
                err = ""
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}"
        msgs = sorted({f"{w.category.__name__}: {str(w.message)[:120]}" for w in caught})
        attempt = {
            "res": res,
            "method": method or "default",
            "warnings": " | ".join(msgs),
            "error": err,
        }
        last = attempt
        if res is not None and res.converged:
            return attempt
    return last

def describe(attempt, outcome, model_name, reml, n_fp, n_np, n_obs):
    res = attempt["res"]
    row = {
        "outcome": outcome,
        "model": model_name,
        "fit": "REML" if reml else "ML",
        "n_fp": n_fp,
        "n_np": n_np,
        "n_obs": n_obs,
        "optimiser": attempt["method"],
        "error": attempt["error"],
        "warnings": attempt["warnings"],
    }
    if res is None:
        row["converged"] = False
        return row
    row.update({"converged": bool(res.converged), "llf": res.llf})
    for tag, key in KEYS.items():
        b, se = res.fe_params[key], res.bse[key]
        row.update(
            {
                f"{tag}_est": b,
                f"{tag}_se": se,
                f"{tag}_ci_lo": b - 1.96 * se,
                f"{tag}_ci_hi": b + 1.96 * se,
                f"{tag}_p": res.pvalues[key],
            }
        )
    return row

rows, sess_rows = [], []
for outcome in PW:
    long = build_long(outcome)
    n_fp = long.loc[long["group"] == "FP", "pid"].nunique()
    n_np = long.loc[long["group"] == "NP", "pid"].nunique()

    per = long.groupby(["pid", "group"], observed=True).agg(
        n_total=("t", "size"),
        n_pre=("t", lambda x: int((x < 0).sum())),
        n_post=("t", lambda x: int((x >= 0).sum())),
    )
    for grp, g in per.reset_index().groupby("group", observed=True):
        for col in ["n_total", "n_pre", "n_post"]:
            for val, cnt in g[col].value_counts().sort_index().items():
                sess_rows.append(
                    {"outcome": outcome, "group": grp, "count_type": col, "n_sessions": val,
                     "n_participants": cnt}
                )

    for model_name, re_formula in MODELS.items():
        for reml in (True, False):
            attempt = fit(long, re_formula, reml)
            rows.append(describe(attempt, outcome, model_name, reml, n_fp, n_np, len(long)))

# Stability check (table note c): drop the five FP with the steepest pre-forum declines
long = build_long("risk_score")
pre_fp = long[(long["group"] == "FP") & (long["t"] < 0)]
slopes = {}
for pid, g in pre_fp.groupby("pid"):
    if len(g) >= 2 and (g["t"].max() - g["t"].min()) >= 0.25:
        slopes[pid] = np.polyfit(g["t"], g["outcome"], 1)[0]
slopes = pd.Series(slopes)
drop = set(slopes.nsmallest(5).index)
sub = long[~long["pid"].isin(drop)]
attempt = fit(sub, MODELS["RS1"], True)
row = describe(attempt, "risk_score", "RS1_drop5_steepest_FP", True,
               sub.loc[sub["group"] == "FP", "pid"].nunique(),
               sub.loc[sub["group"] == "NP", "pid"].nunique(), len(sub))
row["dropped_pre_slopes"] = "; ".join(f"{v:+.2f}" for v in slopes[list(drop)])
rows.append(row)

res_df = pd.DataFrame(rows)

# LR tests on ML fits (boundary test, naive chi-square p is conservative)
lr_rows = []
for outcome in PW:
    ml = res_df[(res_df["outcome"] == outcome) & (res_df["fit"] == "ML")].set_index("model")
    for big, small, ddf in [("RS1", "RI", 2)]:
        if "llf" in ml.columns and pd.notna(ml.at[big, "llf"]) and pd.notna(ml.at[small, "llf"]):
            lr = 2 * (ml.at[big, "llf"] - ml.at[small, "llf"])
            lr_rows.append(
                {"outcome": outcome, "comparison": f"{big} vs {small}", "lr_stat": lr,
                 "df": ddf, "p_naive": stats.chi2.sf(max(lr, 0), ddf)}
            )
lr_df = pd.DataFrame(lr_rows)
res_df = res_df.merge(
    lr_df.assign(model=lr_df["comparison"].str.split(" vs ").str[0]).assign(fit="ML")[
        ["outcome", "model", "fit", "comparison", "lr_stat", "df", "p_naive"]
    ],
    on=["outcome", "model", "fit"],
    how="left",
)
res_df.to_csv(OUT / "piecewise_random_slope_check.csv", index=False)
pd.DataFrame(sess_rows).to_csv(OUT / "piecewise_random_slope_sessions.csv", index=False)

pub = pd.read_csv(OUT / "preforum_piecewise.csv").set_index("outcome")
print("=" * 90)
print("GATE: RI (REML) vs preforum_piecewise.csv (random-intercept columns of MA5)")
print("=" * 90)
for outcome in PW:
    r = res_df[(res_df["outcome"] == outcome) & (res_df["model"] == "RI")
               & (res_df["fit"] == "REML")].iloc[0]
    p = pub.loc[outcome]
    print(f"  {outcome:<16} n {r['n_fp']}/{r['n_np']} [pub {p['n_fp']}/{p['n_np']}]  "
          f"pre {r['pre_est']:+.3f} p={r['pre_p']:.3f} [pub {p['pre_slope_diff']:+.3f} "
          f"p={p['pre_p']:.3f}]  post {r['post_est']:+.3f} p={r['post_p']:.3f} "
          f"[pub {p['post_slope_diff']:+.3f} p={p['post_p']:.3f}]")

print("\nSESSIONS PER PARTICIPANT (median [min-max]; share with >=2 pre / >=2 post)")
for outcome in PW:
    long = build_long(outcome)
    per = long.groupby(["pid", "group"], observed=True)["t"].agg(
        n_total="size", n_pre=lambda x: (x < 0).sum(), n_post=lambda x: (x >= 0).sum()
    ).reset_index()
    for grp, g in per.groupby("group", observed=True):
        print(f"  {outcome:<16} {grp}: total {g['n_total'].median():.0f} "
              f"[{g['n_total'].min()}-{g['n_total'].max()}], "
              f"pre {g['n_pre'].median():.0f} [{g['n_pre'].min()}-{g['n_pre'].max()}], "
              f"post {g['n_post'].median():.0f} [{g['n_post'].min()}-{g['n_post'].max()}]; "
              f">=2 pre {100 * (g['n_pre'] >= 2).mean():.0f}%, "
              f">=2 post {100 * (g['n_post'] >= 2).mean():.0f}%")
        print(f"      total dist {g['n_total'].value_counts().sort_index().to_dict()}  "
              f"post dist {g['n_post'].value_counts().sort_index().to_dict()}")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 200)
cols = ["outcome", "model", "fit", "optimiser", "converged", "llf", "pre_est", "pre_se",
        "pre_p", "post_est", "post_se", "post_p"]
print("\n" + res_df[cols].round(4).to_string(index=False))
for _, r in res_df.iterrows():
    if r["warnings"] or r["error"]:
        print(f"  {r['outcome']} {r['model']} {r['fit']}: {r['warnings']} {r['error']}")
print("\nLikelihood-ratio tests (ML fits):")
print(lr_df.round(4).to_string(index=False))
