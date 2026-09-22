"""1:k nearest-neighbour matching on standardised baseline log risk and age, one-year window (MA2)."""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from paths import NP_CUTOFF, PROC, first_post_dates, read_cohort

warnings.filterwarnings("ignore")

SESSIONS = list(range(9))
RISK_SESSIONS = [0, 1, 2, 3, 4, 5, 7, 8]
WINDOW_DAYS = 365
KS = [1, 2, 4]
BALANCE_VARS = ["baseline", "age", "interval"]

needed = {"HashFinal"}
for i in SESSIONS:
    needed.add(f"date_session_{i}")
    needed.add(f"age_session_{i}")
for s in RISK_SESSIONS:
    needed.add(f"predictpd_riskscore_final_2024_{s}")
df = read_cohort(needed)
df["first_post"] = df["HashFinal"].map(first_post_dates())
df["is_fp"] = df["first_post"].notna()

df["age"] = np.nan
for s in [4, 5, 7, 8]:
    m = df["age"].isna() & df[f"age_session_{s}"].notna() & df[f"date_session_{s}"].notna()
    df.loc[m, "age"] = (
        pd.to_numeric(df.loc[m, f"age_session_{s}"], errors="coerce")
        + (NP_CUTOFF - df.loc[m, f"date_session_{s}"]).dt.days / 365.25
    )

# reported only; the matching does not apply the retention filter
n_valid = sum(
    (df[f"predictpd_riskscore_final_2024_{s}"] > 0) & df[f"date_session_{s}"].notna()
    for s in [4, 5, 7, 8]
)
df["active_np"] = ~df["is_fp"] & (n_valid >= 3)

def windowed_pairs(cutoff_np):
    dd = df.copy()
    dd["cutoff"] = dd["first_post"].where(dd["is_fp"], cutoff_np)
    records = []
    for s in RISK_SESSIONS:
        sub = dd[["HashFinal", "is_fp", "active_np", "cutoff", "age",
                  f"date_session_{s}", f"predictpd_riskscore_final_2024_{s}"]].copy()
        sub.columns = ["HashFinal", "is_fp", "active_np", "cutoff", "age", "sess_date", "risk"]
        sub = sub.dropna(subset=["sess_date", "risk"])
        sub = sub[sub["risk"] > 0]
        sub["session"] = s
        sub["diff_days"] = (sub["sess_date"] - sub["cutoff"]).dt.days
        records.append(sub)
    long = pd.concat(records, ignore_index=True)
    pre = long[(long["diff_days"] < 0) & (long["diff_days"].abs() <= WINDOW_DAYS)].copy()
    pre["ad"] = pre["diff_days"].abs()
    pre = pre.sort_values("ad").groupby("HashFinal").first().reset_index()
    post = long[(long["diff_days"] >= 0) & (long["diff_days"] <= WINDOW_DAYS)].copy()
    post = post.sort_values("diff_days").groupby("HashFinal").first().reset_index()
    pairs = pre[["HashFinal", "is_fp", "active_np", "age", "session", "sess_date", "risk"]].merge(
        post[["HashFinal", "session", "sess_date", "risk", "diff_days"]],
        on="HashFinal", suffixes=("_pre", "_post"))
    pairs = pairs[pairs["session_pre"] != pairs["session_post"]]
    pairs["baseline_log_risk"] = np.log(pairs["risk_pre"])
    pairs["log_risk_change"] = np.log(pairs["risk_post"]) - pairs["baseline_log_risk"]
    pairs["interval_days"] = (pairs["sess_date_post"] - pairs["sess_date_pre"]).dt.days
    return pairs

def euclid_score(fp_row, cand, scale):
    zr = (cand["baseline_log_risk"] - fp_row["baseline_log_risk"]) / scale["risk_sd"]
    if pd.notna(fp_row["age"]):
        za = (cand["age"] - fp_row["age"]) / scale["age_sd"]
        return np.sqrt(zr**2 + za**2)
    return zr.abs()

def match(fp_df, np_df, k, scale):
    rows = []
    for _, fp_row in fp_df.iterrows():
        score = euclid_score(fp_row, np_df, scale)
        cand = np_df.assign(_s=score.values).nsmallest(min(k, len(np_df)), "_s")
        for rank, (_, nr) in enumerate(cand.iterrows(), start=1):
            rows.append({
                "fp_id": fp_row["HashFinal"], "np_id": nr["HashFinal"], "rank": rank,
                "distance": nr["_s"],
                "fp_baseline": fp_row["baseline_log_risk"], "np_baseline": nr["baseline_log_risk"],
                "fp_age": fp_row["age"], "np_age": nr["age"],
                "fp_interval": fp_row["interval_days"], "np_interval": nr["interval_days"],
                "fp_change": fp_row["log_risk_change"], "np_change": nr["log_risk_change"],
                "np_active": nr["active_np"],
            })
    return pd.DataFrame(rows)

def summarise(mp, k, pooled_sd):
    fl = (
        mp.groupby("fp_id")
        .agg(fp_baseline=("fp_baseline", "first"), np_baseline=("np_baseline", "mean"),
             fp_age=("fp_age", "first"), np_age=("np_age", "mean"),
             fp_interval=("fp_interval", "first"), np_interval=("np_interval", "mean"),
             fp_change=("fp_change", "first"), np_change=("np_change", "mean"))
        .reset_index()
    )
    did = fl["fp_change"] - fl["np_change"]
    n = len(did)
    _, p = stats.ttest_1samp(did, 0)
    se = did.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    out = {
        "k": k, "n_fp": n, "n_np_unique": mp["np_id"].nunique(),
        "share_matched_controls_active": mp["np_active"].mean(),
        "did": did.mean(), "ci_low": did.mean() - tcrit * se, "ci_high": did.mean() + tcrit * se,
        "p": p,
    }
    # SMD denominator: pooled pre-matching SD, fixed across k
    for var in BALANCE_VARS:
        v = fl.dropna(subset=[f"fp_{var}", f"np_{var}"])
        out[f"bal_{var}_smd"] = (v[f"fp_{var}"] - v[f"np_{var}"]).mean() / pooled_sd[var]
    return out

pairs = windowed_pairs(NP_CUTOFF)
fp_p = pairs[pairs["is_fp"]].reset_index(drop=True)
np_p = pairs[~pairs["is_fp"]].reset_index(drop=True)

scale = {"risk_sd": pairs["baseline_log_risk"].std(ddof=1), "age_sd": pairs["age"].std(ddof=1)}
col_of = {"baseline": "baseline_log_risk", "age": "age", "interval": "interval_days"}
pooled_sd = {
    var: np.sqrt((fp_p[c].var(ddof=1) + np_p[c].var(ddof=1)) / 2) for var, c in col_of.items()
}

results = []
for k in KS:
    mp = match(fp_p, np_p, k, scale)
    mp.to_csv(PROC / f"matching_euclid_pairs_k{k}.csv", index=False)
    results.append(summarise(mp, k, pooled_sd))
summary = pd.DataFrame(results)
summary.to_csv(PROC / "matching_distance_comparison.csv", index=False)

print("=" * 78)
print("SAMPLE (1-year symmetric window, NP cutoff 2021-05-08)")
print("=" * 78)
print(f"FP with pair: {len(fp_p)};  NP candidate pool: {len(np_p)}")
print(f"  of which active NP (>=3 of S4/5/7/8): {int(np_p['active_np'].sum())}")
print(f"FP missing age: {int(fp_p['age'].isna().sum())};  NP missing age: "
      f"{int(np_p['age'].isna().sum())}")
print(f"z-scale (pooled FP+NP candidates): log-risk SD {scale['risk_sd']:.3f}, "
      f"age SD {scale['age_sd']:.2f}")
print(f"SMD denominators: {({k: round(v, 3) for k, v in pooled_sd.items()})}")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print("\n" + summary.round(3).to_string(index=False))
print("\nSaved: matching_euclid_pairs_k{1,2,4}.csv, matching_distance_comparison.csv")
