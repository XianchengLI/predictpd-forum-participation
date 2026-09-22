"""Sample construction and dichotomisation for the engagement analyses (Table 5, MA6)."""

import numpy as np
import pandas as pd

from paths import HADS_COLS, PROC, RISK_COL, RISK_SESSIONS, first_post_dates, read_cohort

SESSIONS = RISK_SESSIONS  # 4, 5, 7, 8

METRICS = {
    "post_count": ("ge2", "1 post", "2+ posts"),
    "in_degree": ("nonzero", "No replies", "Has replies"),
    "betweenness": ("nonzero", "Non-bridge", "Bridge"),
    "closeness": ("gt_median", "Low", "High"),
    "clustering": ("nonzero", "Zero", "Non-zero"),
}
WINDOWS = {"No window": None, "1Y window": 365}

def high_flag(d, metric):
    rule = METRICS[metric][0]
    if rule == "ge2":
        return d[metric] >= 2
    if rule == "nonzero":
        return d[metric] > 0
    return d[metric] > d[metric].median()

def load():
    cols = {"HashFinal"}
    for s in SESSIONS:
        cols |= {RISK_COL.format(s), f"date_session_{s}", f"age_session_{s}"}
        cols |= {t.format(s) for t in HADS_COLS.values()}
    df = read_cohort(cols)
    df["cutoff"] = df["HashFinal"].map(first_post_dates())
    fp_df = df[df["cutoff"].notna()].copy()
    centrality = pd.read_csv(PROC / "forum_user_centrality_full.csv").rename(
        columns={"user_hash": "HashFinal"}
    )
    return fp_df, centrality

def _log_risk(x):
    return np.log(max(x, 0.001))

def build_pairs(fp_df, centrality, max_days, outcome="risk_score"):
    """Closest session before and after the first post; widest_change uses earliest/latest."""
    tmpl = RISK_COL if outcome == "risk_score" else HADS_COLS[outcome]
    rows = []
    for r in fp_df.to_dict("records"):
        pre_c, post_c = [], []
        for s in SESSIONS:
            date, val = r[f"date_session_{s}"], r[tmpl.format(s)]
            if pd.isna(date) or pd.isna(val):
                continue
            dd = (date - r["cutoff"]).days
            if dd < 0 and (max_days is None or abs(dd) <= max_days):
                pre_c.append((s, date, val, dd))
            elif dd > 0 and (max_days is None or dd <= max_days):
                post_c.append((s, date, val, dd))
        if not pre_c or not post_c:
            continue
        pre = max(pre_c, key=lambda x: x[3])  # closest before the first post
        post = min(post_c, key=lambda x: x[3])  # closest after the first post
        age = r.get(f"age_session_{pre[0]}")
        f = _log_risk if outcome == "risk_score" else float
        rows.append({
            "HashFinal": r["HashFinal"],
            "pre_session": pre[0],
            "post_session": post[0],
            "pre_value": f(pre[2]),
            "post_value": f(post[2]),
            "interval_y": (post[1] - pre[1]).days / 365.25,
            "age_at_cutoff": (
                float(age) + (r["cutoff"] - pre[1]).days / 365.25 if pd.notna(age) else np.nan
            ),
            "n_pre_candidates": len(pre_c),
            "n_post_candidates": len(post_c),
            "widest_change": f(max(post_c, key=lambda x: x[3])[2])
            - f(min(pre_c, key=lambda x: x[3])[2]),
        })
    pairs = pd.DataFrame(rows)
    pairs["change"] = pairs["post_value"] - pairs["pre_value"]
    return pairs.merge(centrality, on="HashFinal", how="inner")
