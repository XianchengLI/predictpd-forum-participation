# PREDICT-PD forum participation study: analysis code

Code for *Need-Driven Participation in an Online Peer-Support Forum for Parkinson Disease
Risk: A Nested Observational Study of the PREDICT-PD Cohort* (Li et al., submitted to JMIR,
2026). The scripts reproduce every number in the manuscript from two raw files;
[REPRODUCTION.md](REPRODUCTION.md) lists each reported value next to the computed one.

The repository contains no data. Individual-level PREDICT-PD data are available from the
corresponding author on reasonable request, subject to the study's data-access procedures.

## Inputs

Place in `data/raw/` (or point `PD_PILOT_RAW` at a directory holding them):

- `PREDICTPD_Data_Filtered_withNHSData_v1.csv`: PREDICT-PD extract (June 2025). Columns used:
  `HashFinal`, `gender`, and per session *s* = 0..8 `date_session_s`, `age_session_s`,
  `predictpd_riskscore_final_2024_s`, `hads_anxiety_score_final_s`,
  `hads_depression_score_final_s`, plus the risk-algorithm inputs listed in
  `scripts/riskscore_nodep_pipeline.py`. Latin-1; dates DD/MM/YYYY.
- `Forum_Posts1630928282.xlsx`: forum posts with `IDHash`, `ReplyingToPost`, `CreatedBy`
  (= `HashFinal`), `CreationDate`.

The risk score is the odds against PD (higher = lower risk) and is log-transformed. Dates are
parsed with `dayfirst=True`.

## Environment

Python 3.9+ and `pip install -r requirements.txt`. One script (`riskscore_nodep_pipeline.py`)
needs R and the cohort team's `predictpd` package
(<https://github.com/Wolfson-PNU-QMUL/PD_Prodromal_Risk_Scores>); set `RSCRIPT` (path to
Rscript) and `PREDICTPD_R_LIB` (the package's `R/` directory).

## Run

```
python run_all.py            # all scripts in order, then the reproduction check
python run_all.py --skip-r   # without the R-dependent depression-excluded score
```

Outputs go to `data/processed/`. About ten minutes.

| # | Script | Manuscript |
|---|---|---|
| 1 | `build_reply_network.py` | Reply network and engagement measures (Methods) |
| 2 | `preforum_table1_person_level.py` | Table 1; cohort and forum counts (Methods, Figure 1) |
| 3 | `preforum_baseline_regeneration.py` | Figure 1 counts; Tables 2-4; MA3; MA4; MA5 (random intercept); pair files for 4-7 and 10 |
| 4 | `followup_after_cutoff_sensitivity.py` | Follow-up after cutoff (Results; MA6) |
| 5 | `lme_continuous_time_sensitivity.py` | Continuous-time Model 1 (Results; MA6) |
| 6 | `ancova_sensitivity.py` | ANCOVA (MA6) |
| 7 | `piecewise_random_slope_check.py` | MA5 random-slope columns, LR tests, stability check |
| 8 | `matching_standardised_euclidean.py` | Matched analysis (MA2) |
| 9 | `riskscore_nodep_pipeline.py` + `riskscore_nodep_compute.R` | Depression-excluded score (MA6) |
| 10 | `lme_nodep_sensitivity.py` | LME on the depression-excluded score (MA6) |
| 11 | `centrality_temporal_analysis.py` | Table 5; engagement descriptives (Methods) |
| 12 | `centrality_hads_analysis.py` | Engagement vs HADS change (Results) |
| 13 | `engagement_adjusted_analysis.py` | Adjusted contrasts; correlation matrix (MA6) |
| 14 | `engagement_pair_rule_sensitivity.py` | Session-pair rule (MA6) |
| 15 | `post_count_continuous_check.py` | Post count as ordered variable (MA6) |
| 16 | `check_reproduction.py` | Writes `REPRODUCTION.md` |

`paths.py` and `engagement_common.py` are shared modules.

## Licence

MIT (see `LICENSE`).
