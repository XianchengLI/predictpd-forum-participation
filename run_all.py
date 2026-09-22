"""Run all scripts in order, then the reproduction check. --skip-r skips the R-dependent steps."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

ORDER = [
    "build_reply_network.py",
    "preforum_table1_person_level.py",
    "preforum_baseline_regeneration.py",
    "followup_after_cutoff_sensitivity.py",
    "lme_continuous_time_sensitivity.py",
    "ancova_sensitivity.py",
    "piecewise_random_slope_check.py",
    "matching_standardised_euclidean.py",
    "riskscore_nodep_pipeline.py",
    "lme_nodep_sensitivity.py",
    "centrality_temporal_analysis.py",
    "centrality_hads_analysis.py",
    "engagement_adjusted_analysis.py",
    "engagement_pair_rule_sensitivity.py",
    "post_count_continuous_check.py",
    "check_reproduction.py",
]
R_STEPS = {"riskscore_nodep_pipeline.py", "lme_nodep_sensitivity.py"}

skip_r = "--skip-r" in sys.argv
for name in ORDER:
    if skip_r and name in R_STEPS:
        print(f"--- skipping {name}")
        continue
    print(f"\n{'=' * 78}\n--- {name}\n{'=' * 78}", flush=True)
    r = subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=SCRIPTS)
    if r.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {r.returncode}")
print("\nAll scripts completed.")
