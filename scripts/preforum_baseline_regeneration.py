"""Primary two-point LME analysis: Figure 1 counts, Tables 2-4, MA3, MA4, MA5 (random intercept) and the pair files used downstream."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from paths import NP_CUTOFF as NP_MEDIAN
from paths import PROC as OUT
from paths import first_post_dates, read_cohort

warnings.filterwarnings("ignore")

SESSIONS = list(range(9))

OUTCOMES = {
    "risk_score": {
        "col_template": "predictpd_riskscore_final_2024_{}",
        "sessions": [0, 1, 2, 3, 4, 5, 7, 8],
        "sessions_s4plus": [4, 5, 7, 8],
        "transform": "log",
    },
    "hads_anxiety": {
        "col_template": "hads_anxiety_score_final_{}",
        "sessions": list(range(9)),
        "sessions_s4plus": [4, 5, 6, 7, 8],
        "transform": "none",
    },
    "hads_depression": {
        "col_template": "hads_depression_score_final_{}",
        "sessions": list(range(9)),
        "sessions_s4plus": [4, 5, 6, 7, 8],
        "transform": "none",
    },
}

print("=" * 78)
print("LOADING")
print("=" * 78)

needed = {"HashFinal", "gender"}
for i in SESSIONS:
    needed.add(f"date_session_{i}")
    needed.add(f"age_session_{i}")
for cfg in OUTCOMES.values():
    for s in cfg["sessions"]:
        needed.add(cfg["col_template"].format(s))

df = read_cohort(needed)
first_post = first_post_dates()
df["first_post"] = df["HashFinal"].map(first_post)
df["is_fp"] = df["first_post"].notna()
df["group"] = df["is_fp"].map({True: "FP", False: "NP"})
df["cutoff_own"] = df["first_post"].fillna(NP_MEDIAN)  # FP: first post; NP: median first post

for s in [4, 5, 7, 8]:
    v = f"predictpd_riskscore_final_2024_{s}"
    df[f"_rv_{s}"] = df[v].notna() & (df[v] > 0) & df[f"date_session_{s}"].notna()
df["_n_risk"] = df[[f"_rv_{s}" for s in [4, 5, 7, 8]]].sum(axis=1)
active_ids = set(df.loc[~df["is_fp"] & (df["_n_risk"] >= 3), "HashFinal"])
fp_ids = set(df.loc[df["is_fp"], "HashFinal"])
raw_idx = df.drop_duplicates("HashFinal").set_index("HashFinal")

print(f"Participants {len(df)}; FP {len(fp_ids)}; active NP {len(active_ids)}")
print(f"NP cutoff {NP_MEDIAN.date()}")

def find_pairs(df_sub, outcome_key, window_days=None, use_s4plus=False, cutoff_col="cutoff_own"):
    cfg = OUTCOMES[outcome_key]
    pre_w = post_w = window_days
    sessions = cfg["sessions_s4plus"] if use_s4plus else cfg["sessions"]

    records = []
    for s in sessions:
        dcol, vcol = f"date_session_{s}", cfg["col_template"].format(s)
        sub = df_sub[["HashFinal", "group", cutoff_col, dcol, vcol]].copy()
        sub.columns = ["HashFinal", "group", "cutoff", "sess_date", "sess_val"]
        sub["session"] = s
        sub = sub.dropna(subset=["sess_date", "sess_val"])
        if outcome_key == "risk_score":
            sub = sub[sub["sess_val"] > 0]
        records.append(sub)
    long = pd.concat(records, ignore_index=True)
    long["diff_days"] = (long["sess_date"] - long["cutoff"]).dt.days

    if pre_w is not None:
        pre = long[(long["diff_days"] < 0) & (long["diff_days"].abs() <= pre_w)].copy()
        post = long[(long["diff_days"] >= 0) & (long["diff_days"] <= post_w)].copy()
        pre["abs_diff"] = pre["diff_days"].abs()
        pre = pre.sort_values("abs_diff").groupby("HashFinal").first().reset_index()
        post = post.sort_values("diff_days").groupby("HashFinal").first().reset_index()
    else:
        long = long.sort_values(["HashFinal", "sess_date"])
        counts = long.groupby("HashFinal").size()
        long = long[long["HashFinal"].isin(counts[counts >= 2].index)]
        pre = long.groupby("HashFinal").first().reset_index()
        post = long.groupby("HashFinal").last().reset_index()

    pairs = pre[["HashFinal", "group", "session", "sess_date", "sess_val"]].merge(
        post[["HashFinal", "session", "sess_date", "sess_val"]],
        on="HashFinal", suffixes=("_pre", "_post"))
    pairs = pairs[pairs["session_pre"] != pairs["session_post"]]

    if cfg["transform"] == "log":
        pairs["pre_transformed"] = np.log(pairs["sess_val_pre"])
        pairs["post_transformed"] = np.log(pairs["sess_val_post"])
    else:
        pairs["pre_transformed"] = pairs["sess_val_pre"]
        pairs["post_transformed"] = pairs["sess_val_post"]
    pairs["change"] = pairs["post_transformed"] - pairs["pre_transformed"]
    pairs["interval_days"] = (pairs["sess_date_post"] - pairs["sess_date_pre"]).dt.days
    pairs = pairs.rename(columns={
        "session_pre": "pre_session", "session_post": "post_session",
        "sess_date_pre": "pre_date", "sess_date_post": "post_date"})
    pairs["cutoff"] = pairs["HashFinal"].map(raw_idx["cutoff_own"])
    pairs["preforum_ok"] = pairs["pre_date"] < pairs["cutoff"]  # strictly before
    return pairs

def to_long(pairs, extra=()):
    rows = []
    for r in pairs.itertuples(index=False):
        base = {"pid": r.HashFinal, "group": r.group, "interval_days": r.interval_days}
        for c in extra:
            base[c] = getattr(r, c)
        rows.append({**base, "time": 0, "y": r.pre_transformed})
        rows.append({**base, "time": 1, "y": r.post_transformed})
    d = pd.DataFrame(rows)
    d["group"] = pd.Categorical(d["group"], categories=["NP", "FP"])
    d["interval_centered"] = d["interval_days"] - d["interval_days"].mean()
    return d

def fit_full(d, formula, model_label, config, outcome):
    m = smf.mixedlm(formula, data=d, groups=d["pid"]).fit()
    coef_rows = []
    for k in m.fe_params.index:
        coef_rows.append({
            "config": config, "outcome": outcome, "model": model_label, "param": k,
            "beta": m.fe_params[k], "se": m.bse[k],
            "ci_lo": m.fe_params[k] - 1.96 * m.bse[k],
            "ci_hi": m.fe_params[k] + 1.96 * m.bse[k],
            "p": m.pvalues[k],
        })
    key = next(k for k in m.fe_params.index
               if "time" in k.lower() and "group" in k.lower() and ":" in k)
    return coef_rows, m.fe_params[key], m.bse[key], m.pvalues[key]

def df_for(np_sel):
    if np_sel == "active":
        return df[df["HashFinal"].isin(fp_ids | active_ids)]
    return df

CONFIGS = [
    # name, np_sel, window_days, s4plus
    ("primary_clean", "active", None, True),
    ("eligible_clean", "all", None, True),
    ("1y_median", "active", 365, False),
]

main_rows, coef_rows_all = [], []
kept_pairs = {}  # (config, outcome) -> pairs after rule

for name, np_sel, window, s4plus in CONFIGS:
    print(f"\n{'=' * 78}\nCONFIG: {name}\n{'=' * 78}")
    dsub = df_for(np_sel)
    for ok in OUTCOMES:
        pairs = find_pairs(dsub, ok, window_days=window, use_s4plus=s4plus)
        n_late_np = int(((pairs["group"] == "NP") & ~pairs["preforum_ok"]).sum())
        n_late_fp = int(((pairs["group"] == "FP") & ~pairs["preforum_ok"]).sum())
        pairs = pairs[pairs["preforum_ok"]]
        kept_pairs[(name, ok)] = pairs

        fp = pairs[pairs["group"] == "FP"]
        npg = pairs[pairs["group"] == "NP"]
        d_long = to_long(pairs)
        c1, b1, se1, p1 = fit_full(d_long, "y ~ time * group", "M1", name, ok)
        c2, b2, se2, p2 = fit_full(
            d_long, "y ~ time * group + interval_centered + time:interval_centered",
            "M2", name, ok)
        coef_rows_all.extend(c1 + c2)

        row = {
            "config": name, "outcome": ok,
            "n_fp": len(fp), "n_np": len(npg),
            "excluded_baseline_after_cutoff_fp": n_late_fp,
            "excluded_baseline_after_cutoff_np": n_late_np,
            "fp_change": fp["change"].mean(), "np_change": npg["change"].mean(),
            "m1_beta3": b1, "m1_se": se1, "m1_p": p1,
            "m2_beta3": b2, "m2_se": se2, "m2_p": p2,
        }
        if ok == "risk_score":  # Results text: ratio of fold-changes on the odds scale
            row.update({"m1_fold_change": np.exp(b1), "m1_fc_ci_lo": np.exp(b1 - 1.96 * se1),
                        "m1_fc_ci_hi": np.exp(b1 + 1.96 * se1)})
        main_rows.append(row)
        print(f"  [{ok}] nFP={len(fp)} nNP={len(npg)} "
              f"(baseline after cutoff excluded: {n_late_fp} FP, {n_late_np} NP)")
        print(f"    dFP={row['fp_change']:+.3f} dNP={row['np_change']:+.3f}  "
              f"M1 b3={b1:+.4f} (SE {se1:.3f}, p={p1:.4f})  "
              f"M2 b3={b2:+.4f} (SE {se2:.3f}, p={p2:.4f})")
        if name in ("primary_clean", "eligible_clean"):
            pairs.to_csv(OUT / f"preforum_pairs_{name}_{ok}.csv", index=False)

pd.DataFrame(main_rows).to_csv(OUT / "preforum_main_results.csv", index=False)
pd.DataFrame(coef_rows_all).to_csv(OUT / "preforum_lme_full_coefficients.csv", index=False)

print(f"\n{'=' * 78}\nMODEL 3 (new primary sample)\n{'=' * 78}")
pp = kept_pairs[("primary_clean", "risk_score")].copy()
ac = []  # age at the baseline session projected to the participant's forum cutoff
for r in pp.itertuples(index=False):
    s = int(r.pre_session)
    a = raw_idx.at[r.HashFinal, f"age_session_{s}"]
    dts = raw_idx.at[r.HashFinal, f"date_session_{s}"]
    cut = raw_idx.at[r.HashFinal, "cutoff_own"]
    ac.append(np.nan if pd.isna(a) or pd.isna(dts) else float(a) + (cut - dts).days / 365.25)
pp["age_at_cutoff"] = ac

m3_rows = []
for col in ["age_at_cutoff"]:
    sub = pp.dropna(subset=[col])
    d = to_long(sub, extra=(col,))
    d["age_c"] = d[col] - d[col].mean()
    m = smf.mixedlm("y ~ time * group + age_c + time:age_c", data=d, groups=d["pid"]).fit()
    for k in m.fe_params.index:
        m3_rows.append({"age_definition": col, "n": len(sub), "param": k,
                        "beta": m.fe_params[k], "se": m.bse[k],
                        "ci_lo": m.fe_params[k] - 1.96 * m.bse[k],
                        "ci_hi": m.fe_params[k] + 1.96 * m.bse[k],
                        "p": m.pvalues[k]})
    key = "time:group[T.FP]"
    print(f"  [{col}] n={len(sub)}  b3={m.fe_params[key]:+.4f}  "
          f"SE={m.bse[key]:.3f}  p={m.pvalues[key]:.4f}  "
          f"age={m.fe_params['age_c']:+.4f} (p={m.pvalues['age_c']:.2e})  "
          f"time:age={m.fe_params['time:age_c']:+.4f} (p={m.pvalues['time:age_c']:.2e})")
pd.DataFrame(m3_rows).to_csv(OUT / "preforum_model3.csv", index=False)

# Model 3 variants quoted in MA4
sub = pp.dropna(subset=["age_at_cutoff"])
d = to_long(sub, extra=("age_at_cutoff",))
d["age_c"] = d["age_at_cutoff"] - d["age_at_cutoff"].mean()
d["age_c2"] = d["age_c"] ** 2
var_rows = []
for label, formula in [
    ("M1 on Model 3 sample", "y ~ time * group"),
    ("age main effect only", "y ~ time * group + age_c"),
    ("age + age x time (Model 3)", "y ~ time * group + age_c + time:age_c"),
    ("quadratic age x time", "y ~ time * group + age_c + time:age_c + age_c2 + time:age_c2"),
]:
    m = smf.mixedlm(formula, data=d, groups=d["pid"]).fit()
    k = "time:group[T.FP]"
    var_rows.append({"variant": label, "n": len(sub), "beta3": m.fe_params[k],
                     "se": m.bse[k], "p": m.pvalues[k]})
    print(f"  [{label}] n={len(sub)} b3={m.fe_params[k]:+.4f} p={m.pvalues[k]:.4f}")
pd.DataFrame(var_rows).to_csv(OUT / "preforum_model3_variants.csv", index=False)

# Age-gradient attribution of the baseline gap (MA4)
np_sub = pp[(pp["group"] == "NP") & pp["age_at_cutoff"].notna()]
grad = np.polyfit(np_sub["age_at_cutoff"], np_sub["pre_transformed"], 1)[0]
fp_sub = pp[(pp["group"] == "FP") & pp["age_at_cutoff"].notna()]
age_gap = fp_sub["age_at_cutoff"].mean() - np_sub["age_at_cutoff"].mean()
base_gap = fp_sub["pre_transformed"].mean() - np_sub["pre_transformed"].mean()
sub = sub.assign(group=pd.Categorical(sub["group"], categories=["NP", "FP"]))
ols_adj = smf.ols("pre_transformed ~ group + age_at_cutoff", data=sub).fit()
ols_int = smf.ols("pre_transformed ~ age_at_cutoff * group", data=sub).fit()
attr = {
    "n": len(sub),
    "np_age_gradient_per_year": grad,
    "fp_slope": np.polyfit(fp_sub["age_at_cutoff"], fp_sub["pre_transformed"], 1)[0],
    "age_gap_years": age_gap,
    "predicted_gap_from_age": grad * age_gap,
    "observed_baseline_gap": base_gap,
    "share_explained": grad * age_gap / base_gap,
    "residual_group_diff_age_adjusted": ols_adj.params["group[T.FP]"],
    "residual_group_diff_p": ols_adj.pvalues["group[T.FP]"],
    "age_slope_interaction_p": ols_int.pvalues["age_at_cutoff:group[T.FP]"],
}
pd.DataFrame([attr]).to_csv(OUT / "preforum_age_attribution.csv", index=False)
print(f"  MA4 attribution: NP age gradient {grad:+.4f}/y x gap {age_gap:+.2f}y "
      f"= {grad * age_gap:+.3f} of baseline gap {base_gap:+.3f} "
      f"({100 * grad * age_gap / base_gap:.0f}%); residual group diff "
      f"{attr['residual_group_diff_age_adjusted']:+.3f} (p={attr['residual_group_diff_p']:.4f}); "
      f"age-slope interaction p={attr['age_slope_interaction_p']:.2f}")

print(f"\n{'=' * 78}\nPIECEWISE (new primary rosters)\n{'=' * 78}")
pw_rows = []
PW = {"risk_score": ("predictpd_riskscore_final_2024_{}", [0, 1, 2, 3, 4, 5, 7, 8], "log"),
      "hads_anxiety": ("hads_anxiety_score_final_{}", SESSIONS, "none"),
      "hads_depression": ("hads_depression_score_final_{}", SESSIONS, "none")}
for ok, (tmpl, sess, tr) in PW.items():
    ids = set(kept_pairs[("primary_clean", ok)]["HashFinal"])
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
    long["group"] = pd.Categorical(long["is_fp"].map({True: "FP", False: "NP"}),
                                   categories=["NP", "FP"])
    m = smf.mixedlm("outcome ~ pre_time + post_time + group + pre_time:group + post_time:group",
                    data=long, groups=long["pid"]).fit()
    row = {"outcome": ok,
           "n_fp": long.loc[long["group"] == "FP", "pid"].nunique(),
           "n_np": long.loc[long["group"] == "NP", "pid"].nunique(),
           "pre_slope_diff": m.fe_params["pre_time:group[T.FP]"],
           "pre_se": m.bse["pre_time:group[T.FP]"],
           "pre_p": m.pvalues["pre_time:group[T.FP]"],
           "post_slope_diff": m.fe_params["post_time:group[T.FP]"],
           "post_se": m.bse["post_time:group[T.FP]"],
           "post_p": m.pvalues["post_time:group[T.FP]"]}
    pw_rows.append(row)
    print(f"  {ok:<16} nFP={row['n_fp']} nNP={row['n_np']}  "
          f"PRE {row['pre_slope_diff']:+.4f} (p={row['pre_p']:.3f})  "
          f"POST {row['post_slope_diff']:+.4f} (p={row['post_p']:.3f})")
pd.DataFrame(pw_rows).to_csv(OUT / "preforum_piecewise.csv", index=False)

print(f"\n{'=' * 78}\nBALANCE TABLE (new primary sample)\n{'=' * 78}")
bal_rows = []
rp = pp  # risk pairs with ages
rp = rp.assign(is_female=rp["HashFinal"].map(raw_idx["gender"]).eq("Female").where(
    rp["HashFinal"].map(raw_idx["gender"]).isin(["Female", "Male"])))

def wrow(name, fpv, npv, fmt=".2f"):
    fpv, npv = fpv.dropna(), npv.dropna()
    p = stats.ttest_ind(fpv, npv, equal_var=False)[1]
    smd = (fpv.mean() - npv.mean()) / np.sqrt((fpv.var(ddof=1) + npv.var(ddof=1)) / 2)
    bal_rows.append({"variable": name, "fp_n": len(fpv), "fp_mean": fpv.mean(),
                     "fp_sd": fpv.std(ddof=1), "np_n": len(npv), "np_mean": npv.mean(),
                     "np_sd": npv.std(ddof=1), "smd": smd, "p": p})
    print(f"  {name:<30} FP {fpv.mean():{fmt}} +/- {fpv.std(ddof=1):{fmt}} (n={len(fpv)})  "
          f"NP {npv.mean():{fmt}} +/- {npv.std(ddof=1):{fmt}} (n={len(npv)})  "
          f"SMD {smd:+.3f}  p={p:.4f}")

fpm, npm = rp["group"] == "FP", rp["group"] == "NP"
wrow("Age at forum cutoff (y)", rp.loc[fpm, "age_at_cutoff"], rp.loc[npm, "age_at_cutoff"], ".1f")
g = rp.dropna(subset=["is_female"])
tab = pd.crosstab(g["group"], g["is_female"])
chi2, pg, _, _ = stats.chi2_contingency(tab, correction=False)
p1, p2 = tab.loc["FP", True] / tab.loc["FP"].sum(), tab.loc["NP", True] / tab.loc["NP"].sum()
bal_rows.append({"variable": "Female sex (%)",
                 "fp_n": int(tab.loc['FP'].sum()), "fp_mean": 100 * p1,
                 "np_n": int(tab.loc['NP'].sum()), "np_mean": 100 * p2,
                 "fp_sd": np.nan, "np_sd": np.nan,
                 "smd": (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2), "p": pg})
print(f"  {'Female sex (%)':<30} FP {tab.loc['FP', True]}/{tab.loc['FP'].sum()} "
      f"({100 * tab.loc['FP', True] / tab.loc['FP'].sum():.1f}%)  "
      f"NP {tab.loc['NP', True]}/{tab.loc['NP'].sum()} "
      f"({100 * tab.loc['NP', True] / tab.loc['NP'].sum():.1f}%)  p={pg:.4f}")
wrow("Baseline log risk score", rp.loc[fpm, "pre_transformed"], rp.loc[npm, "pre_transformed"])
ap = kept_pairs[("primary_clean", "hads_anxiety")]
wrow("Baseline HADS Anxiety", ap.loc[ap["group"] == "FP", "pre_transformed"],
     ap.loc[ap["group"] == "NP", "pre_transformed"])
dp = kept_pairs[("primary_clean", "hads_depression")]
wrow("Baseline HADS Depression", dp.loc[dp["group"] == "FP", "pre_transformed"],
     dp.loc[dp["group"] == "NP", "pre_transformed"])
wrow("Observation interval (y)", rp.loc[fpm, "interval_days"] / 365.25,
     rp.loc[npm, "interval_days"] / 365.25)
pd.DataFrame(bal_rows).to_csv(OUT / "preforum_balance_table.csv", index=False)

print(f"\n{'=' * 78}\nFLOW DIAGRAM COUNTS (risk score)\n{'=' * 78}")
flow = []
elig_all = find_pairs(df, "risk_score", window_days=None, use_s4plus=True)
n = {"level": ">=2 valid S4+ sessions (pairs)",
     "fp": int((elig_all["group"] == "FP").sum()), "np": int((elig_all["group"] == "NP").sum())}
flow.append(n)
print(f"  {n['level']:<42} FP {n['fp']}  NP {n['np']}")
ok_pairs = elig_all[elig_all["preforum_ok"]]
n = {"level": "+ baseline strictly before forum cutoff",
     "fp": int((ok_pairs["group"] == "FP").sum()), "np": int((ok_pairs["group"] == "NP").sum())}
flow.append(n)
print(f"  {n['level']:<42} FP {n['fp']}  NP {n['np']}")
act = ok_pairs[ok_pairs["group"].eq("FP") | ok_pairs["HashFinal"].isin(active_ids)]
n = {"level": "+ active NP retention filter",
     "fp": int((act["group"] == "FP").sum()), "np": int((act["group"] == "NP").sum())}
flow.append(n)
print(f"  {n['level']:<42} FP {n['fp']}  NP {n['np']}")
for ok in ["hads_anxiety", "hads_depression"]:
    p = kept_pairs[("primary_clean", ok)]
    n = {"level": f"primary_clean {ok}",
         "fp": int((p["group"] == "FP").sum()), "np": int((p["group"] == "NP").sum())}
    flow.append(n)
    print(f"  {n['level']:<42} FP {n['fp']}  NP {n['np']}")
pd.DataFrame(flow).to_csv(OUT / "preforum_flow_counts.csv", index=False)

print(f"\nAll outputs written to {OUT}/preforum_*.csv")
