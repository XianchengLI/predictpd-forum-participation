"""Paths, constants and readers shared by all scripts. Raw data: data/raw/ or $PD_PILOT_RAW."""

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PD_PILOT_RAW", REPO / "data" / "raw"))
PROC = Path(os.environ.get("PD_PILOT_PROC", REPO / "data" / "processed"))
PROC.mkdir(parents=True, exist_ok=True)

COHORT_CSV = RAW / "PREDICTPD_Data_Filtered_withNHSData_v1.csv"  # PREDICT-PD extract, June 2025
FORUM_XLSX = RAW / "Forum_Posts1630928282.xlsx"  # forum posts, April-August 2021

# Forum cutoff assigned to non-forum participants: the median first-post date among FP.
NP_CUTOFF = pd.Timestamp("2021-05-08")

# Risk score column (higher = lower predicted PD risk); sessions with a score are 4, 5, 7, 8.
RISK_COL = "predictpd_riskscore_final_2024_{}"
RISK_SESSIONS = [4, 5, 7, 8]
HADS_COLS = {
    "hads_anxiety": "hads_anxiety_score_final_{}",
    "hads_depression": "hads_depression_score_final_{}",
}

def read_cohort(columns):
    """Requested columns only; dates are UK format, so dayfirst=True is required."""
    with open(COHORT_CSV, "r", encoding="latin-1") as f:
        header = f.readline().strip().split(",")
    wanted = set(columns)
    df = pd.read_csv(
        COHORT_CSV,
        usecols=[c for c in header if c in wanted],
        low_memory=False,
        encoding="latin-1",
    )
    for c in df.columns:
        if c.startswith("date_session_"):
            df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    return df

def read_forum():
    forum = pd.read_excel(FORUM_XLSX)
    forum["CreationDate"] = pd.to_datetime(forum["CreationDate"], dayfirst=True, errors="coerce")
    return forum

def first_post_dates(forum=None):
    forum = read_forum() if forum is None else forum
    return forum.groupby("CreatedBy")["CreationDate"].min()
