"""
Baseline Randomwalk: avaliação meia-pirâmide / SoftEd assimétrico - (janela [t-K, t], pico em t-K).
Resultados em results/02_model_baseline_randomwalk_meio_softed/
Análise em notebooks/
"""

# =========================
# 0. IMPORTS
# =========================
import pandas as pd
import numpy as np
import json
import time
import os
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

# =========================
# 1. CONFIG
# =========================
BASE_NAME = "events_wide_minute.parquet"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / BASE_NAME

RESULTS_PATH = BASE_DIR / "results" / "02_model_baseline_randomwalk"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_BEST_FILE = RESULTS_PATH / "results_best.parquet"
RESULTS_BEST_BY_SIDE_FILE = RESULTS_PATH / "results_best_by_side.parquet"
RESULTS_RAW_FILE = RESULTS_PATH / "results_raw.parquet"

TARGET_ALARMS = 12000  # calibrado para ~4000 alarmes após filtro best-per-team/task (1/3 das features é mantida)
RANDOM_SEED = 42       # reprodutibilidade
N_RUNS = 10            # repetições para estimar variância do baseline aleatório

K = 10
N_WORKERS = 7
FEATURES = ["passe", "passe_certo", "passe_errado"]
TASKS = [
    ("attack", "open_play", "gol_open_play"),
    ("defense", "open_play", "gol_open_play"),
]

RUN_DATE = datetime.today().date().isoformat()
TEAMS = None
FORCE_RERUN = True
TEST_LABEL = "baseline aleatorio meio_softed"

# =========================
# 2. FUNCTIONS
# =========================
def count_total_series_length(df, teams):
    """Conta total de pontos em todas as séries válidas para calibrar p."""
    total = 0
    for team in teams:
        df_team = df[(df["home_team"] == team) | (df["away_team"] == team)]
        for match_id in df_team["match_id"].unique():
            df_match = df_team[df_team["match_id"] == match_id]
            for side in ["casa", "fora"]:
                if side == "casa" and df_match["home_team"].iloc[0] != team:
                    continue
                if side == "fora" and df_match["away_team"].iloc[0] != team:
                    continue
                for _, _, _ in TASKS:
                    for feature in FEATURES:
                        feature_col = f"{feature}_{side}"
                        if df_match[feature_col].sum() == 0:
                            continue
                        total += len(df_match)
    return total


def random_alarms(n_minutes, p, rng):
    """Dispara True aleatoriamente com probabilidade p por minuto."""
    return rng.random(n_minutes) < p


def har_eval_soft_half_python(drift_series, label_series, k):
    """
    Meia-pirâmide: janela [t-K, t], pico em t-K.
    score(alarme) = 1 - (alarme - (t-K)) / K  se alarme ∈ [t-K, t], senão 0.
    FP = 1 - score para cada alarme (complementar ao TP), garantindo TP+FP+FN+TN = total de minutos.
    Não usa R/harbinger — avaliação puramente em Python.
    """
    import numpy as np
    goal_pos  = np.where(np.array(label_series, dtype=bool))[0]
    alarm_pos = np.where(np.array(drift_series, dtype=bool))[0]

    TP, FP, FN = 0.0, 0.0, 0.0

    for a in alarm_pos:
        best = 0.0
        for t in goal_pos:
            if (t - k) <= a <= t:
                best = max(best, 1.0 - (a - (t - k)) / k)
        TP += best
        FP += (1.0 - best)

    FN = len(goal_pos) - TP

    TN = (len(drift_series) - len(goal_pos)) - FP
    return TP, FP, FN, TN


def append_parquet(df_new, path):
    if df_new.empty:
        return
    if path.exists():
        df_old = pd.read_parquet(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    first_cols = []
    for col in ["run_id", "run_timestamp", "test_label", "run_number", "alarm_prob", "side"]:
        if col in df_new.columns:
            first_cols.append(col)
    if first_cols:
        rest = [c for c in df_new.columns if c not in first_cols]
        df_new = df_new[first_cols + rest]
    df_new.to_parquet(path, index=False)


def _agg_mean(df_agg, group_cols):
    """Agrega por média entre runs (baseline estocástico)."""
    df_agg["precision"] = df_agg["TP_sum"] / (df_agg["TP_sum"] + df_agg["FP_sum"])
    df_agg["recall"] = df_agg["TP_sum"] / (df_agg["TP_sum"] + df_agg["FN_sum"])
    df_agg["f1"] = 2 * df_agg["precision"] * df_agg["recall"] / (df_agg["precision"] + df_agg["recall"])
    df_agg = df_agg.fillna(0)
    return df_agg.groupby(group_cols, as_index=False).agg(
        TP_sum=("TP_sum", "mean"),
        FP_sum=("FP_sum", "mean"),
        FN_sum=("FN_sum", "mean"),
        TN_sum=("TN_sum", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        f1_std=("f1", "std"),
    )


def _worker_run_team(args):
    team, df_team, k, alarm_prob, run_number, seed = args
    res = run_team(df_team, team, k, alarm_prob, run_number, seed)
    res["params"] = res["params"].astype(str)

    df_agg = (
        res.groupby(["team", "task", "goal_type", "feature", "detector", "params", "run_number"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_agg["precision"] = df_agg["TP_sum"] / (df_agg["TP_sum"] + df_agg["FP_sum"])
    df_agg["recall"] = df_agg["TP_sum"] / (df_agg["TP_sum"] + df_agg["FN_sum"])
    df_agg["f1"] = 2 * df_agg["precision"] * df_agg["recall"] / (df_agg["precision"] + df_agg["recall"])
    df_agg = df_agg.fillna(0)
    df_agg["alarm_prob"] = alarm_prob

    df_agg_side = (
        res.groupby(["team", "task", "goal_type", "side", "feature", "detector", "params", "run_number"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_agg_side["precision"] = df_agg_side["TP_sum"] / (df_agg_side["TP_sum"] + df_agg_side["FP_sum"])
    df_agg_side["recall"] = df_agg_side["TP_sum"] / (df_agg_side["TP_sum"] + df_agg_side["FN_sum"])
    df_agg_side["f1"] = 2 * df_agg_side["precision"] * df_agg_side["recall"] / (df_agg_side["precision"] + df_agg_side["recall"])
    df_agg_side = df_agg_side.fillna(0)
    df_agg_side["alarm_prob"] = alarm_prob

    return df_agg, df_agg_side, res


def run_team(df_team, team, k, alarm_prob, run_number, seed):
    # Seed determinístico por (run_number, team) para reprodutibilidade
    team_seed = seed + run_number * 10000 + hash(team) % 10000
    rng = np.random.default_rng(team_seed)

    results = []
    for match_id in df_team["match_id"].unique():
        df_match = df_team[df_team["match_id"] == match_id]
        for side in ["casa", "fora"]:
            if side == "casa" and df_match["home_team"].iloc[0] != team:
                continue
            if side == "fora" and df_match["away_team"].iloc[0] != team:
                continue
            opponent = "fora" if side == "casa" else "casa"
            for task, goal_type, goal_base in TASKS:
                label_col = f"{goal_base}_{side if task == 'attack' else opponent}"
                df_ctx = df_match.copy()
                goal_series = df_ctx[label_col].astype(bool)
                for feature in FEATURES:
                    feature_col = f"{feature}_{side}"
                    if df_ctx[feature_col].sum() == 0:
                        continue
                    # Série de alarmes aleatórios (mesma série para todos os períodos)
                    drift_series = pd.Series(index=df_ctx.index, dtype=bool)
                    for period in sorted(df_ctx["period"].unique()):
                        mask = df_ctx["period"] == period
                        n = mask.sum()
                        alarms = random_alarms(n, alarm_prob, rng)
                        for i, idx in enumerate(df_ctx.loc[mask].index):
                            drift_series.loc[idx] = alarms[i]
                    drift_series = drift_series.astype(bool)
                    TP, FP, FN, TN = har_eval_soft_half_python(drift_series, goal_series, k)
                    results.append({
                        "run_date": RUN_DATE,
                        "match_id": match_id,
                        "run_number": run_number,
                        "team": team,
                        "task": task,
                        "goal_type": goal_type,
                        "side": side,
                        "feature": feature,
                        "detector": "RandomWalk",
                        "params": json.dumps({"alarm_prob": round(alarm_prob, 6), "seed": seed}),
                        "alarm_prob": alarm_prob,
                        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                    })
    return pd.DataFrame(results)


# =========================
# 3. MAIN
# =========================
if __name__ == "__main__":
    print("=== 02_model: Baseline Random Walk ===")
    print(f"Target alarmes: {TARGET_ALARMS} | Seed: {RANDOM_SEED} | Runs: {N_RUNS}")

    start = time.time()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    run_id = f"v8_{run_timestamp}"

    df = pd.read_parquet(DATA_PATH)
    teams = (
        pd.unique(df[["home_team", "away_team"]].values.ravel()).tolist()
        if TEAMS is None else TEAMS
    )

    # Calibrar p com base no total de pontos
    print("Calibrando probabilidade de alarme...")
    total_points = count_total_series_length(df, teams)
    alarm_prob = TARGET_ALARMS / total_points
    print(f"Total pontos: {total_points:,} | p = {alarm_prob:.5f} (~1 alarme a cada {1/alarm_prob:.0f} min por série)")

    pending_teams = teams if FORCE_RERUN else (
        [t for t in teams if t not in set(pd.read_parquet(RESULTS_BEST_FILE)["team"].unique())]
        if RESULTS_BEST_FILE.exists() else teams
    )

    print(f"Times: {len(pending_teams)} | Runs: {N_RUNS} | Workers: {N_WORKERS}")

    for run_number in range(N_RUNS):
        print(f"\n--- Run {run_number + 1}/{N_RUNS} ---")
        args_list = [
            (team, df[(df["home_team"] == team) | (df["away_team"] == team)],
             K, alarm_prob, run_number, RANDOM_SEED)
            for team in pending_teams
        ]
        with Pool(processes=N_WORKERS) as pool:
            for df_agg, df_agg_side, df_raw in pool.imap_unordered(_worker_run_team, args_list, chunksize=1):
                for dfp, path in [(df_agg, RESULTS_BEST_FILE), (df_agg_side, RESULTS_BEST_BY_SIDE_FILE)]:
                    dfp.insert(0, "test_label", TEST_LABEL)
                    dfp.insert(0, "run_timestamp", run_timestamp)
                    dfp.insert(0, "run_id", run_id)
                    append_parquet(dfp, path)
                df_raw.insert(0, "test_label", TEST_LABEL)
                df_raw.insert(0, "run_timestamp", run_timestamp)
                append_parquet(df_raw, RESULTS_RAW_FILE)
                print(f"  ✓ {df_agg['team'].iloc[0]} (run={run_number})")

    print(f"\nTempo total: {(time.time() - start)/60:.1f} min")
    print(f"Salvo em: {RESULTS_BEST_FILE}")
    print(f"Alarmes esperados por run: ~{TARGET_ALARMS:,}")
