"""
UNICORN Challenge Statistical Analysis 
==========================================================

Core methodology is preserved (Paired permutation test + Holm-Bonferroni across 4 teams that submitted to all 20 tasks (ALL Leaderboard).
Normalized scores were computed during the evaluation step and stored per task across 5 adaptor runs; the statistical analysis script uses these normalized task-level scores as input.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def holm_bonferroni_adjust(pvals: List[float]) -> List[float]:
    """
    Holm–Bonferroni adjusted p-values (FWER control).

    Input: list of raw p-values.
    Output: adjusted p-values in the original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    prev = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * pvals[idx]
        val = max(val, prev)
        prev = val
        adjusted[idx] = min(val, 1.0)

    return adjusted.tolist()


# ============================================================================
#  Additional Effect Size Metrics
# ============================================================================


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired data: mean(a-b) / SD(a-b)

    Interpretation:
    - Small effect: |d| ≈ 0.2
    - Medium effect: |d| ≈ 0.5
    - Large effect: |d| ≈ 0.8
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


def paired_within_task_swap_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    B: int = 200_000,
    alternative: str = "two-sided",
    seed: int = 0,
) -> Dict[str, float]:
    """
    Paired permutation test implemented as "swap A and B within each task".

    For each task i, with probability 0.5 we swap (a_i, b_i) -> (b_i, a_i).
    This creates the null distribution under exchangeability within each paired task.

    Returns: {"N", "T_o", "p_value"}
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1D arrays of the same shape.")

    d = a - b
    N = d.size
    T_o = float(d.mean())

    rng = np.random.default_rng(seed)

    # swap mask: True means swap within that task
    swap = rng.random(size=(B, N)) < 0.5

    # If we swap in task i, the difference becomes (b_i - a_i) = -d_i; else it stays d_i
    d_perm = np.where(swap, -d, d)  # shape (B, N) via broadcasting
    T = d_perm.mean(axis=1)

    if alternative == "two-sided":
        p = (1.0 + np.sum(np.abs(T) >= abs(T_o))) / (1.0 + B)
    else:
        raise ValueError("alternative must be 'two-sided' for this evaluation.")

    return {"N": int(N), "T_o": T_o, "p_value": float(p)}


def analyze_four_teams(
    df: pd.DataFrame,
    team_col: str = "team",
    task_col: str = "task",
    score_col: str = "score_norm",
    teams: Optional[List[str]] = None,
    B_perm: int = 200_000,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:

    if teams is None:
        teams = sorted(df[team_col].unique().tolist())

    if len(teams) != 4:
        raise ValueError(f"Expected exactly 4 teams; got {len(teams)}: {teams}")

    wide = df[df[team_col].isin(teams)].pivot(
        index=task_col, columns=team_col, values=score_col
    )

    if wide.isna().any().any():
        missing = wide.isna().sum().sum()
        raise ValueError(
            f"Found missing task scores in the 4-team subset (count={missing}). "
            f"Paired analysis requires all 4 teams have completed all tasks."
        )

    wide = wide.sort_index()
    N_tasks = wide.shape[0]

    if verbose:
        print(f"Analysis Configuration:")
        print(f"  Teams: {teams}")
        print(f"  Tasks: {N_tasks}")
        print(f"  Permutations: {B_perm:,}")
        print()

    if N_tasks != 20:
        warnings.warn(f"Expected 20 tasks, got {N_tasks}. Proceeding anyway.")

    comparisons = list(itertools.combinations(teams, 2))
    rows = []

    raw_pvals_perm = []

    rng = np.random.default_rng(seed)

    for A, B in comparisons:
        a = wide[A].to_numpy()
        b = wide[B].to_numpy()

        # run permutation test
        res_perm = paired_within_task_swap_permutation_test(
            a,
            b,
            B=B_perm,
            alternative="two-sided",
            seed=int(rng.integers(0, 1_000_000)),
        )

        # effect sizes
        diff_mean = res_perm["T_o"]
        d = cohens_d(a, b)

        rows.append(
            {
                "team_A": A,
                "team_B": B,
                "N_tasks": res_perm["N"],
                "mean_diff_A_minus_B": diff_mean,
                "cohens_d": d,
                # permutation
                "p_raw": res_perm["p_value"],
            }
        )

        raw_pvals_perm.append(res_perm["p_value"])

    # Holm adjustment to correct for multiple comparisons (per method)
    p_adj_perm = holm_bonferroni_adjust(raw_pvals_perm)

    for r, adjp in zip(rows, p_adj_perm):
        r["p_holm"] = adjp
        r["significant_holm"] = adjp < 0.05

    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)

    if verbose:
        print("=" * 80)
        print("PAIRWISE COMPARISON RESULTS (Permutation Test + Holm-Bonferroni)")
        print("=" * 80)
        print()
        cols = [
            "team_A",
            "team_B",
            "mean_diff_A_minus_B",
            "cohens_d",
            "p_raw",
            "p_holm",
            "significant_holm",
        ]
        print(out[cols].to_string(index=False))
        print()
        print(
            f"Significant comparisons (Permutation Holm): {out['significant_holm'].sum()}/6"
        )
        print()

    return out


def load_results(file_path: str) -> pd.DataFrame:
    """
    Load results from an Excel file with one sheet per task.

    Each sheet must contain columns:
        - team
        - normalized score

    Returns a DataFrame with columns:
        team | task | score_norm
    """

    xls = pd.ExcelFile(file_path)  # <-- important

    all_data = []

    for sheet_name in xls.sheet_names:

        # Skip non-task sheets
        if sheet_name.lower() in ["overview", "worst-score"]:
            continue

        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)

        # Keep only needed columns
        df_sheet = df_sheet[["team", "normalized score"]].copy()

        # Rename
        df_sheet = df_sheet.rename(columns={"normalized score": "score_norm"})

        # Add task name
        df_sheet["task"] = sheet_name

        all_data.append(df_sheet)

    if not all_data:
        raise ValueError("No task sheets found in Excel file.")

    df = pd.concat(all_data, ignore_index=True)

    return df


# ============================================================================
# EXAMPLE USAGE WITH REAL DATA FORMAT
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UNICORN CHALLENGE STATISTICAL ANALYSIS - ENHANCED VERSION")
    print("=" * 80)
    print()
    ##Normalized task scores per team (sorted by tasks) - for reference only, not used in analysis
    mevis_scores = [
        0.679,
        0.714,
        0.3939999999999999,
        0.664,
        0.151,
        0.34,
        0.0,
        0.33,
        0.6712292002147074,
        0.482,
        0.756,
        0.517,
        0.5760000000000001,
        0.3580000000000001,
        0.281,
        0.74,
        0.30578512396694196,
        0.9271012006861062,
        0.117,
        0.117,
    ]
    aihmi_scores = [
        0.632,
        0.31000000000000005,
        0.3899999999999999,
        0.35,
        0.222,
        -0.005333333333333338,
        0.0,
        0.006,
        0.5370370370370371,
        0.639,
        0.759,
        0.623,
        0.706,
        0.812,
        0.312,
        0.732,
        0.5909090909090909,
        0.721269296740995,
        0.48,
        0.119,
    ]
    baseline_scores = [
        0.767,
        0.3340000000000001,
        0.17599999999999993,
        0.462,
        0.243,
        0.005333333333333338,
        0.0,
        0.0,
        0.35721953837895865,
        0.037,
        0.008,
        0.643,
        0.728,
        0.8220000000000001,
        0.286,
        0.738,
        0.5950413223140497,
        0.7427101200686104,
        0.48,
        0.143,
    ]
    kaiko_scores = [
        0.155,
        0.10000000000000009,
        -0.252,
        0.302,
        0.199,
        -0.01466666666666668,
        0.0,
        0.0,
        0.4739667203435319,
        0.0,
        0.003,
        0.69,
        0.706,
        0.46399999999999997,
        0.458,
        -0.052000000000000046,
        0.6735537190082647,
        0.9528301886792452,
        0.456,
        0.148,
    ]
    # Shape dataframe 
    # Expected format:
    # team,task,score_norm
    # MEVIS,task_01,0.52
    # MEVIS,task_02,0.41
    rng = np.random.default_rng(42)

    teams = ["MEVIS", "AIMHI", "Baseline", "kaiko"]
    tasks = [f"task_{i:02d}" for i in range(1, 21)]

    # Build a long-form df identical to what load_results returns
    df = pd.DataFrame(
        {
            "task": tasks * 4,
            "team": sum([[t] * len(tasks) for t in teams], []),
            "score_norm": mevis_scores + aihmi_scores + baseline_scores + kaiko_scores,
        }
    )
    print("Overall UNICORN Scores (for reference):")
    print(df.groupby("team")["score_norm"].mean().sort_values(ascending=False))
    print()

    # Run enhanced analysis
    results = analyze_four_teams(
        df,
        team_col="team",
        task_col="task",
        score_col="score_norm",
        teams=["MEVIS", "AIMHI", "Baseline", "kaiko"],
        B_perm=200_000,  # We use 200,000 permutations to ensure that Monte Carlo error in the estimated p-values is below about ±0.0005 near the 0.05 significance threshold
        seed=42,
        verbose=True,
    )
  
    # Save results
    results.to_csv(
        "/Volumes/temporary/judith/code/unicorn_eval/statistics/results/unicorn_results.csv",
        index=False,
    )

    print("ANALYSIS COMPLETED: Output saved to unicorn_results.csv")
