"""Risk score recomputed without the depression component using the predictpd R package (MA6). Needs RSCRIPT and PREDICTPD_R_LIB (see README)."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from paths import COHORT_CSV as RAW
from paths import PROC

RSCRIPT = os.environ.get("RSCRIPT", "Rscript")
R_LIB = os.environ.get("PREDICTPD_R_LIB")
if not R_LIB:
    raise SystemExit("Set PREDICTPD_R_LIB to the R/ directory of the predictpd package "
                     "(see README, section R dependency).")

SESSIONS = ["4", "5", "7", "8"]

# Step A: extract inputs
need = {"HashFinal", "gender", "fx_firstdegree", "hx_diabetes", "hx_diabetes_age",
        "pesticide_exposure_ever"}
for s in SESSIONS:
    need |= {
        f"age_session_{s}", f"smoking_status_{s}", f"drink_coffee_{s}",
        f"use_laxatives_more_than_1x_per_week_{s}",
        f"maintain_erection_untreated_poor_{s}",
        f"hads_depression_level_moderatesevere_{s}",
        f"rbdsq_summary_score_highrisk_{s}",
        f"headinjury_loss_consiousness_{s}",
        f"rx_nsaids_use_{s}", f"rx_calcium_channel_bocker_use_{s}",
        f"rx_beta_blocker_use_{s}", f"drink_alcohol_{s}",
        f"predictpd_riskscore_2024_{s}", f"predictpd_riskscore_final_2024_{s}",
    }

with open(RAW, "r", encoding="latin-1") as f:
    header = f.readline().strip().split(",")
missing = sorted(need - set(header))
if missing:
    raise SystemExit(f"Missing input columns in raw data: {missing}")

df = pd.read_csv(RAW, usecols=sorted(need), encoding="latin-1", low_memory=False)
df = df.groupby("HashFinal", as_index=False).first()
df.to_csv(PROC / "riskscore_lib_inputs.csv", index=False)
print(f"Step A: extracted {df.shape} (collapsed to unique HashFinal) "
      "-> riskscore_lib_inputs.csv")

# Step B: run R
r = subprocess.run(
    [RSCRIPT, str(Path(__file__).resolve().parent / "riskscore_nodep_compute.R"),
     str(PROC / "riskscore_lib_inputs.csv"), str(PROC / "riskscore_lib_output.csv"), R_LIB],
    capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    raise SystemExit(f"Rscript failed:\n{r.stderr}")

# Step C: validate and derive final_nodep
# Gate: the package must reproduce the stored base score; the depression ratio is checked exactly.
lib = pd.read_csv(PROC / "riskscore_lib_output.csv")
m = df.merge(lib, on="HashFinal", validate="1:1")

def yesno(series):
    s = series.astype(str).str.strip().str.lower()
    return np.where(s.isin(["yes", "y", "1", "true", "1.0"]), "Yes",
                    np.where(s.isin(["no", "n", "0", "false", "0.0"]), "No", None))

rows = []
ok_all = True
for s in SESSIONS:
    stored = m[f"predictpd_riskscore_2024_{s}"]
    computed = m[f"lib_base_{s}"]
    both = stored.notna() & computed.notna()
    rel = (stored[both] - computed[both]).abs() / stored[both].abs()
    n_exact = int((rel < 1e-6).sum())
    n_close = int((rel < 1e-3).sum())
    lb, ln = m[f"lib_base_{s}"], m[f"lib_nodep_{s}"]
    lib_ratio = ln / lb
    dep = pd.Series(yesno(m[f"hads_depression_level_moderatesevere_{s}"]))
    ana_ratio = pd.Series(np.where(dep == "Yes", 1.6,
                          np.where(dep == "No", 0.87, 1.0)), index=m.index)
    rboth = lib_ratio.notna()
    ratio_agree = float((lib_ratio[rboth] - ana_ratio[rboth]).abs().max()) if rboth.any() else np.nan
    rows.append({
        "session": s,
        "n_stored": int(stored.notna().sum()),
        "n_lib_computable": int(computed.notna().sum()),
        "n_both": int(both.sum()),
        "pct_exact_1e6": round(100 * n_exact / both.sum(), 2),
        "pct_close_1e3": round(100 * n_close / both.sum(), 2),
        "max_rel_diff": float(rel.max()),
        "median_rel_diff": float(rel.median()),
        "max_ratio_disagreement": ratio_agree,
    })
    print(f"S{s}: lib computable={computed.notna().sum()}/{stored.notna().sum()} stored; "
          f"on both n={both.sum()}: exact(<1e-6)={100 * n_exact / both.sum():.1f}%, "
          f"close(<1e-3)={100 * n_close / both.sum():.1f}%, max rel diff={rel.max():.3g}; "
          f"analytic-vs-lib ratio max diff={ratio_agree:.2e}")
    if both.sum() == 0 or n_exact / both.sum() < 0.98 or ratio_agree > 1e-9:
        ok_all = False

pd.DataFrame(rows).to_csv(PROC / "riskscore_validation_report.csv", index=False)

if not ok_all:
    print("\nVALIDATION GATE NOT PASSED - final_nodep NOT generated. "
          "Inspect riskscore_validation_report.csv and per-factor differences.")
    raise SystemExit(1)

out = m[["HashFinal"]].copy()
for s in SESSIONS:
    dep = pd.Series(yesno(m[f"hads_depression_level_moderatesevere_{s}"]))
    ratio = pd.Series(np.where(dep == "Yes", 1.6,
                      np.where(dep == "No", 0.87, 1.0)), index=m.index)
    out[f"riskscore_final_nodep_{s}"] = m[f"predictpd_riskscore_final_2024_{s}"] * ratio
    scored = m[f"predictpd_riskscore_final_2024_{s}"].notna()
    changed = (ratio[scored] != 1).mean()
    print(f"S{s}: final_nodep n={int(scored.sum())}  "
          f"(depression term present for {100 * changed:.1f}% of scored rows)")

out.to_csv(PROC / "riskscore_final_nodep.csv", index=False)
print("\nVALIDATION PASSED - written riskscore_final_nodep.csv")
