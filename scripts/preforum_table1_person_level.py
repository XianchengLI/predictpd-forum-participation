"""Table 1, its Retention paragraph and the counts in Methods / Figure 1. Output: preforum_table1_person_level.csv"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from paths import FORUM_XLSX, NP_CUTOFF, PROC, read_cohort

warnings.filterwarnings("ignore")
RS = [4, 5, 7, 8]
HS = [4, 5, 6, 7, 8]

need = {"HashFinal", "gender"}
for s in range(9):
    need |= {f"date_session_{s}", f"age_session_{s}",
             f"hads_anxiety_score_final_{s}", f"hads_depression_score_final_{s}"}
for s in RS:
    need.add(f"predictpd_riskscore_final_2024_{s}")

df = read_cohort(need)
n_rows, n_with_id, n_unique = len(df), int(df.HashFinal.notna().sum()), df.HashFinal.nunique()
print(f"rows {n_rows}; rows with HashFinal {n_with_id}; unique participants {n_unique}")
df = df[df.HashFinal.notna()].copy()
n_dup_people = int((df.groupby("HashFinal").size() > 1).sum())

# merge duplicate records session-wise: first non-null value per column per person
merged = df.groupby("HashFinal", as_index=False).first()
print(f"after merging duplicate records: {len(merged)} people "
      f"({n_dup_people} appeared on two rows)")

forum = pd.read_excel(FORUM_XLSX)
fp_ids = set(forum["CreatedBy"].dropna().unique())
merged["is_fp"] = merged["HashFinal"].isin(fp_ids)

merged["age"] = np.nan
for s in RS:
    m = merged["age"].isna() & merged[f"age_session_{s}"].notna() & merged[f"date_session_{s}"].notna()
    merged.loc[m, "age"] = (pd.to_numeric(merged.loc[m, f"age_session_{s}"], errors="coerce")
                            + (NP_CUTOFF - merged.loc[m, f"date_session_{s}"]).dt.days / 365.25)

for s in RS:
    v = f"predictpd_riskscore_final_2024_{s}"
    merged[f"_v{s}"] = merged[v].notna() & (merged[v] > 0) & merged[f"date_session_{s}"].notna()
merged["n_s4"] = merged[[f"_v{s}" for s in RS]].sum(axis=1)

pool = merged[merged["n_s4"] >= 1]
fp, npg = pool[pool["is_fp"]], pool[~pool["is_fp"]]
n_forum_accounts = len(fp_ids)
n_linked = int(merged["is_fp"].sum())
print(f"\nForum accounts {n_forum_accounts}; linked to a cohort record {n_linked}; "
      f"with >=1 valid S4-8 risk score {len(fp)}")
print(f"S4-8 pool (>=1 valid risk): FP {len(fp)}, NP {len(npg)}, total {len(pool)}")

rows = []

def add(name, f, n, p, fmt=".2f", **extra):
    rows.append({"row": name, "fp": f, "np": n, "p": p, **extra})
    ps = "<0.001" if p is not None and p < 0.001 else (f"{p:.3f}" if p is not None else "--")
    print(f"  {name:<32} FP {f:{fmt}}  NP {n:{fmt}}  p={ps}")

add("Cohort N (unique participants)", n_unique, n_unique, None, ".0f")
add("Participants on two records", n_dup_people, n_dup_people, None, ".0f")
add("Forum accounts (incl. moderator)", n_forum_accounts, np.nan, None, ".0f")
add("Forum accounts linked to cohort", n_linked, np.nan, None, ".0f")
add("n (>=1 valid S4-8 risk score)", len(fp), len(npg), None, ".0f")
add("FP share of cohort (%)", 100 * len(fp) / n_unique, np.nan, None, ".1f")

print("\n--- Demographics ---")
add("Age at forum (y)", fp["age"].mean(), npg["age"].mean(),
    stats.ttest_ind(fp["age"].dropna(), npg["age"].dropna(), equal_var=False)[1], ".1f",
    fp_sd=fp["age"].std(), np_sd=npg["age"].std())
print(f"     (SD: FP {fp['age'].std():.1f}, NP {npg['age'].std():.1f})")
g = pool[pool["gender"].isin(["Female", "Male"])]
tab = pd.crosstab(g["is_fp"], g["gender"] == "Female")
pg = stats.chi2_contingency(tab, correction=False)[1]
add("Female sex (%)", 100 * tab.loc[True, True] / tab.loc[True].sum(),
    100 * tab.loc[False, True] / tab.loc[False].sum(), pg, ".1f")

print("\n--- Retention ---")
for k in [2, 3]:
    ct = np.array([[int((fp["n_s4"] >= k).sum()), int((fp["n_s4"] < k).sum())],
                   [int((npg["n_s4"] >= k).sum()), int((npg["n_s4"] < k).sum())]])
    add(f">={k} sessions (%)", 100 * (fp["n_s4"] >= k).mean(), 100 * (npg["n_s4"] >= k).mean(),
        stats.chi2_contingency(ct, correction=False)[1], ".1f")

# Retention paragraph: first/last valid risk-score date per person
dates = pd.concat([merged.loc[merged[f"_v{s}"], ["HashFinal", f"date_session_{s}"]]
                   .rename(columns={f"date_session_{s}": "d"}) for s in RS])
per = dates.groupby("HashFinal")["d"].agg(first="min", last="max", n="count")
per["int_y"] = (per["last"] - per["first"]).dt.days / 365.25
per["is_fp"] = per.index.isin(fp_ids)
np1 = per[~per["is_fp"]]
fp2, np2 = per[per["is_fp"] & (per["n"] >= 2)], per[~per["is_fp"] & (per["n"] >= 2)]
np_pre = np2[np2["first"] < NP_CUTOFF]
add("NP pool with no measurement before forum (%)", np.nan,
    100 * (np1["first"] >= NP_CUTOFF).mean(), None, ".1f")
add("Interval (y): FP vs NP started before forum", fp2["int_y"].mean(), np_pre["int_y"].mean(),
    None, ".2f", n_np=len(np_pre))

def pairs_for(tmpl, sessions, log=False):
    recs = []
    for s in sessions:
        sub = merged[["HashFinal", "is_fp", f"date_session_{s}", tmpl.format(s)]].copy()
        sub.columns = ["HashFinal", "is_fp", "d", "v"]
        sub = sub.dropna(subset=["d", "v"])
        if log:
            sub = sub[sub["v"] > 0]
        sub["session"] = s
        recs.append(sub)
    lg = pd.concat(recs, ignore_index=True).sort_values(["HashFinal", "d"])
    cnt = lg.groupby("HashFinal").size()
    lg = lg[lg["HashFinal"].isin(cnt[cnt >= 2].index)]
    a = lg.groupby("HashFinal").first().reset_index()
    b = lg.groupby("HashFinal").last().reset_index()
    p = a.merge(b, on="HashFinal", suffixes=("_pre", "_post"))
    p = p[p["session_pre"] != p["session_post"]]
    p["is_fp"] = p["is_fp_pre"]
    p["base"] = np.log(p["v_pre"]) if log else p["v_pre"]
    p["int_y"] = (p["d_post"] - p["d_pre"]).dt.days / 365.25
    return p

print("\n--- Observation interval and first-session health (>= 2 valid sessions) ---")
rp = pairs_for("predictpd_riskscore_final_2024_{}", RS, log=True)
f2, n2 = rp[rp["is_fp"]], rp[~rp["is_fp"]]
add("n (>=2 valid risk sessions)", len(f2), len(n2), None, ".0f")
add("Observation interval (y)", f2["int_y"].mean(), n2["int_y"].mean(),
    stats.ttest_ind(f2["int_y"], n2["int_y"], equal_var=False)[1])
add("First-session log risk", f2["base"].mean(), n2["base"].mean(),
    stats.ttest_ind(f2["base"], n2["base"], equal_var=False)[1])
for lbl, tmpl in [("HADS anxiety", "hads_anxiety_score_final_{}"),
                  ("HADS depression", "hads_depression_score_final_{}")]:
    hp = pairs_for(tmpl, HS)
    hf, hn = hp[hp["is_fp"]], hp[~hp["is_fp"]]
    add(f"First-session {lbl}", hf["base"].mean(), hn["base"].mean(),
        stats.ttest_ind(hf["base"], hn["base"], equal_var=False)[1])

pd.DataFrame(rows).to_csv(PROC / "preforum_table1_person_level.csv", index=False)
print(f"\nSaved {PROC / 'preforum_table1_person_level.csv'}")
