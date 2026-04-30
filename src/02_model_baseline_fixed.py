"""
Baseline Fixed: avaliação meia-pirâmide / SoftEd assimétrico - (janela [t-K, t], pico em t-K).
Resultados em results/02_model_baseline_fixed/
Análise em notebooks/
"""

# =========================
# 0. IMPORTS
# =========================
import pandas as pd
import json
import time
import os
from pathlib import Path
from itertools import product
from datetime import datetime
from multiprocessing import Pool

# =========================
# 1. CONFIG
# =========================
BASE_NAME = "events_wide_minute.parquet"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / BASE_NAME

RESULTS_PATH = BASE_DIR / "results" / "02_model_baseline_fixed"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_BEST_FILE = RESULTS_PATH / "results_best.parquet"
RESULTS_BEST_BY_SIDE_FILE = RESULTS_PATH / "results_best_by_side.parquet"
RESULTS_RAW_FILE = RESULTS_PATH / "results_raw.parquet"

# Grid de intervalos fixos (minutos entre alarmes)
INTERVAL_OPTIONS = [25]  # adicionando apenas o intervalo de 25 min

K = 10
N_WORKERS = 4
FEATURES = ["passe", "passe_certo", "passe_errado"]
TASKS = [
    ("attack", "open_play", "gol_open_play"),
    ("defense", "open_play", "gol_open_play"),
]

RUN_DATE = datetime.today().date().isoformat()
TEAMS = None
FORCE_RERUN = True
TEST_LABEL = "baseline fixo meio_softed"

# ==============================
# 2. FUNCTIONS
# ==============================
def fixed_interval_alarms(n_minutes, interval_minutes):
    """
    Dispara True a cada 'interval_minutes' passos, independente do sinal.
    Começa a contar do minuto 0 (primeiro alarme em interval_minutes - 1).
    """
    return [(i > 0 and i % interval_minutes == 0) for i in range(n_minutes)]


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
    for col in ["run_id", "run_timestamp", "test_label", "interval_minutes", "side"]:
        if col in df_new.columns:
            first_cols.append(col)
    if first_cols:
        rest = [c for c in df_new.columns if c not in first_cols]
        df_new = df_new[first_cols + rest]
    df_new.to_parquet(path, index=False)


def _agg_best(df_agg, group_cols):
    df_agg["precision"] = df_agg["TP_sum"] / (df_agg["TP_sum"] + df_agg["FP_sum"])
    df_agg["recall"] = df_agg["TP_sum"] / (df_agg["TP_sum"] + df_agg["FN_sum"])
    df_agg["f1"] = 2 * df_agg["precision"] * df_agg["recall"] / (df_agg["precision"] + df_agg["recall"])
    df_agg = df_agg.fillna(0)
    best_idx = df_agg.groupby(group_cols)["f1"].idxmax()
    return df_agg.loc[best_idx].copy()


def _worker_run_team(args):
    team, df_team, k, interval_minutes = args
    res = run_team(df_team, team, k, interval_minutes)
    res["params"] = res["params"].astype(str)
    df_agg = (
        res.groupby(["team", "task", "goal_type", "feature", "detector", "params", "interval_minutes"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_best = _agg_best(df_agg, ["team", "task", "goal_type"])
    df_agg_side = (
        res.groupby(["team", "task", "goal_type", "side", "feature", "detector", "params", "interval_minutes"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_best_side = _agg_best(df_agg_side, ["team", "task", "goal_type", "side"])
    return df_best, df_best_side, res


def run_team(df_team, team, k, interval_minutes):
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
                    drift_series = pd.Series(index=df_ctx.index, dtype=bool)
                    for period in sorted(df_ctx["period"].unique()):
                        mask = df_ctx["period"] == period
                        n = mask.sum()
                        alarms = fixed_interval_alarms(n, interval_minutes)
                        for i, idx in enumerate(df_ctx.loc[mask].index):
                            drift_series.loc[idx] = alarms[i]
                    drift_series = drift_series.astype(bool)
                    TP, FP, FN, TN = har_eval_soft_half_python(drift_series, goal_series, k)
                    results.append({
                        "run_date": RUN_DATE,
                        "match_id": match_id,
                        "team": team,
                        "task": task,
                        "goal_type": goal_type,
                        "side": side,
                        "feature": feature,
                        "detector": "FixedInterval",
                        "params": json.dumps({"interval_minutes": interval_minutes}),
                        "interval_minutes": interval_minutes,
                        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                    })
    return pd.DataFrame(results)

# ==============================
# 3. MAIN
# ==============================
if __name__ == "__main__":
    print("=== 02_model: Baseline alarmes fixos ===")
    print(f"Intervalos testados: {INTERVAL_OPTIONS} minutos")
    print(f"Resultados: {RESULTS_BEST_FILE}")

    start = time.time()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    run_id = f"v7_{run_timestamp}"
    df = pd.read_parquet(DATA_PATH)
    teams = (
        pd.unique(df[["home_team", "away_team"]].values.ravel()).tolist()
        if TEAMS is None else TEAMS
    )

    pending_teams = teams if FORCE_RERUN else (
        [t for t in teams if t not in set(pd.read_parquet(RESULTS_BEST_FILE)["team"].unique())]
        if RESULTS_BEST_FILE.exists() else teams
    )

    print(f"Times: {len(pending_teams)} | Intervalos: {len(INTERVAL_OPTIONS)} | Workers: {N_WORKERS}")

    for interval in INTERVAL_OPTIONS:
        print(f"\n--- Intervalo fixo = {interval} min ---")
        args_list = [
            (team, df[(df["home_team"] == team) | (df["away_team"] == team)], K, interval)
            for team in pending_teams
        ]
        with Pool(processes=N_WORKERS) as pool:
            for df_best, df_best_side, df_raw in pool.imap_unordered(_worker_run_team, args_list, chunksize=1):
                for dfp, path in [(df_best, RESULTS_BEST_FILE), (df_best_side, RESULTS_BEST_BY_SIDE_FILE)]:
                    dfp.insert(0, "test_label", TEST_LABEL)
                    dfp.insert(0, "run_timestamp", run_timestamp)
                    dfp.insert(0, "run_id", run_id)
                    append_parquet(dfp, path)
                df_raw.insert(0, "test_label", TEST_LABEL)
                df_raw.insert(0, "run_timestamp", run_timestamp)
                append_parquet(df_raw, RESULTS_RAW_FILE)
                print(f"  ✓ {df_best['team'].iloc[0]} (intervalo={interval} min)")

    print(f"\nTempo total: {(time.time() - start)/60:.1f} min")
    print(f"Salvo em: {RESULTS_BEST_FILE}")
