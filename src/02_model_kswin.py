"""
Kswin: avaliação meia-pirâmide / SoftEd assimétrico - (janela [t-K, t], pico em t-K).
Resultados em results/02_model_kswin/
Análise em notebooks/meio_softed/
"""

# =========================
# 0. IMPORTS
# =========================
import pandas as pd
import json
import time
import hashlib
import os
from pathlib import Path
from itertools import product
from datetime import datetime
from multiprocessing import Pool

from river import drift

# =========================
# 1. CONFIG 
# =========================
BASE_NAME = "events_wide_minute.parquet"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / BASE_NAME

RESULTS_PATH = BASE_DIR / "results" / "02_model_kswin"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_BEST_FILE = RESULTS_PATH / "results_best.parquet"
RESULTS_BEST_BY_SIDE_FILE = RESULTS_PATH / "results_best_by_side.parquet"
RESULTS_RAW_FILE = RESULTS_PATH / "results_raw.parquet"

WINDOW_MA_OPTIONS = [3, 5, 10, 12, 15]

K = 10
N_WORKERS = 7
FEATURES = ["passe", "passe_certo", "passe_errado"]
TASKS = [
    ("attack", "open_play", "gol_open_play"),
    ("defense", "open_play", "gol_open_play"),
]

RUN_DATE = datetime.today().date().isoformat()
TEAMS = None
EXPERIMENT_TAG = "todos_times"
FORCE_RERUN = True
TEST_LABEL = "KSWIN meio_softed"
# True = 1 combo KSWIN (bem mais rápido); False = grid completo (21 combos, lento)
FAST_GRID = False

# ==============================
# 2. FUNCTIONS
# ==============================
def generate_experiment_id(base, task, feature, detector, params, k, tag):
    raw = f"{base}|{task}|{feature}|{detector}|{params}|k={k}|minute_in_period|{tag}"
    return hashlib.md5(raw.encode()).hexdigest()


def compute_ma_per_period(series_per_period, window_ma=10):
    import numpy as np
    arr = np.asarray(series_per_period, dtype=float)
    return pd.Series(arr).rolling(window=window_ma, min_periods=1).mean().values


def drift_detection_on_ma(ma_values, params, window_ma=10):
    """
    KSWIN na série já suavizada (MA).
    Só alimenta o detector a partir do índice window_ma (warmup implícito).
    """
    detector = drift.KSWIN(**params)
    out_drift = []
    for i, v in enumerate(ma_values):
        if i >= window_ma:
            detector.update(float(v))
            raw_drift = detector.drift_detected
            if raw_drift:
                detector = drift.KSWIN(**params)
        else:
            raw_drift = False
        out_drift.append(raw_drift)
    return out_drift


def har_eval_soft_half_python(drift_series, label_series, k):
    """
    Meia-pirâmide: janela [t-K, t], pico em t-K.
    Associação ótima alarme↔gol via Hungarian algorithm (scipy).
    FP = 1 - score, FN = m - TP, TN = (t - m) - FP.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    goal_pos  = np.where(np.array(label_series, dtype=bool))[0]
    alarm_pos = np.where(np.array(drift_series, dtype=bool))[0]

    if len(alarm_pos) == 0 or len(goal_pos) == 0:
        FP = float(len(alarm_pos))
        FN = float(len(goal_pos))
        TN = float(len(drift_series) - len(goal_pos)) - FP
        return 0.0, FP, FN, TN

    def score(a, t):
        if (t - k) <= a <= t:
            return 1.0 - (a - (t - k)) / k
        return 0.0

    segs = [[int(t) - k, int(t)] for t in goal_pos]
    merged = [segs[0][:]]
    for seg in segs[1:]:
        if seg[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append(seg[:])

    S_d = np.zeros(len(alarm_pos))
    for inf, sup in merged:
        D_idx = np.where((alarm_pos >= inf) & (alarm_pos <= sup))[0]
        E_idx = np.where((goal_pos  >= inf) & (goal_pos  <= sup))[0]
        if len(D_idx) == 0:
            continue
        D_mini = alarm_pos[D_idx]
        E_mini = goal_pos[E_idx]
        if len(D_mini) == 1 and len(E_mini) == 1:
            S_d[D_idx[0]] = score(D_mini[0], E_mini[0])
        elif len(E_mini) >= 1:
            Mu = np.array([[score(a, t) for t in E_mini] for a in D_mini])
            row_ind, col_ind = linear_sum_assignment(-Mu)
            for r, c in zip(row_ind, col_ind):
                S_d[D_idx[r]] = Mu[r, c]

    TP = float(np.sum(S_d))
    FP = float(np.sum(1.0 - S_d))
    FN = float(len(goal_pos)) - TP
    TN = float(len(drift_series) - len(goal_pos)) - FP
    return TP, FP, FN, TN


def get_detector_products(fast_grid=False):
    """
    KSWIN(alpha, window_size, stat_size, ...). River faz
    sample(range(window_size - stat_size), stat_size) → precisa
    window_size - stat_size >= stat_size ⇒ window_size >= 2 * stat_size.

    Se fast_grid=True, usa menos combinações para rodar mais rápido (tuning inicial).
    """
    if fast_grid:
        alphas = [0.005]
        window_sizes = [40]
        stat_sizes = [15]
    else:
        alphas = [0.001, 0.005, 0.01]
        window_sizes = [25, 40, 60]
        stat_sizes = [10, 15, 20]
    products = []
    for alpha in alphas:
        for window_size in window_sizes:
            for stat_size in stat_sizes:
                if stat_size >= window_size:
                    continue
                if window_size < 2 * stat_size:
                    continue
                products.append({
                    "detector": "KSWIN",
                    "params": {
                        "alpha": alpha,
                        "window_size": window_size,
                        "stat_size": stat_size,
                    },
                })
    return products


def append_parquet(df_new, path):
    if df_new.empty:
        return
    if path.exists():
        df_old = pd.read_parquet(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    first_cols = []
    for col in ["run_id", "run_timestamp", "fast_grid", "test_label", "window_ma", "side"]:
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
    team, df_team, detector_products, k, window_ma = args
    res = run_team(df_team, team, detector_products, k, window_ma)
    res["params"] = res["params"].astype(str)
    df_agg = (
        res.groupby(["team", "task", "goal_type", "feature", "detector", "params", "window_ma"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_best = _agg_best(df_agg, ["team", "task", "goal_type"])
    df_agg_side = (
        res.groupby(["team", "task", "goal_type", "side", "feature", "detector", "params", "window_ma"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_best_side = _agg_best(df_agg_side, ["team", "task", "goal_type", "side"])
    return df_best, df_best_side, res


def run_team(df_team, team, detector_products, k, window_ma):
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
                label_col = f"{goal_base}_{side if task=='attack' else opponent}"
                df_ctx = df_match.copy()
                goal_series = df_ctx[label_col].astype(bool)
                for feature in FEATURES:
                    feature_col = f"{feature}_{side}"
                    if df_ctx[feature_col].sum() == 0:
                        continue
                    ma_series = pd.Series(index=df_ctx.index, dtype=float)
                    for period in sorted(df_ctx["period"].unique()):
                        mask = df_ctx["period"] == period
                        s = df_ctx.loc[mask, feature_col]
                        ma_vals = compute_ma_per_period(s.tolist(), window_ma)
                        for i, idx in enumerate(s.index):
                            ma_series.loc[idx] = ma_vals[i]
                    for det in detector_products:
                        drift_series = pd.Series(index=df_ctx.index, dtype=object)
                        for period in sorted(df_ctx["period"].unique()):
                            mask = df_ctx["period"] == period
                            ma_vals = ma_series.loc[mask].tolist()
                            drifts = drift_detection_on_ma(ma_vals, det["params"], window_ma)
                            for i, idx in enumerate(df_ctx.loc[mask].index):
                                drift_series.loc[idx] = drifts[i]
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
                            "detector": det["detector"],
                            "params": json.dumps(det["params"]),
                            "window_ma": window_ma,
                            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                        })
    return pd.DataFrame(results)

# ==============================
# 3. MAIN
# ==============================
if __name__ == "__main__":
    print("=== 02_model: KSWIN | janela MA | Hungarian | sem cooldown ===")
    print(f"Drift (MA): {WINDOW_MA_OPTIONS}")
    print(f"Resultados: {RESULTS_BEST_FILE}")
    print(f"Por lado: {RESULTS_BEST_BY_SIDE_FILE}")
    detector_products = get_detector_products(fast_grid=FAST_GRID)
    print(f"Grid KSWIN: {len(detector_products)} produtos de parâmetros" + (" (FAST_GRID)" if FAST_GRID else ""))

    start = time.time()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    run_id = f"v6_{run_timestamp}_fast{FAST_GRID}"
    df = pd.read_parquet(DATA_PATH)
    teams = (
        pd.unique(df[["home_team", "away_team"]].values.ravel()).tolist()
        if TEAMS is None else TEAMS
    )

    if FORCE_RERUN:
        pending_teams = teams
    elif RESULTS_BEST_FILE.exists():
        teams_done = set(pd.read_parquet(RESULTS_BEST_FILE)["team"].unique())
        pending_teams = [t for t in teams if t not in teams_done]
    else:
        pending_teams = teams

    print(f"Times: {len(pending_teams)} (workers={N_WORKERS})")

    if not pending_teams:
        print("Nada a processar.")
    else:
        for window_ma in WINDOW_MA_OPTIONS:
            print(f"\n--- MA={window_ma} ---")
            args_list = [
                (team, df[(df["home_team"] == team) | (df["away_team"] == team)], detector_products, K, window_ma)
                for team in pending_teams
            ]
            with Pool(processes=N_WORKERS) as pool:
                for df_best, df_best_side, df_raw in pool.imap_unordered(_worker_run_team, args_list, chunksize=1):
                    for dfp, path in [(df_best, RESULTS_BEST_FILE), (df_best_side, RESULTS_BEST_BY_SIDE_FILE)]:
                        dfp.insert(0, "test_label", TEST_LABEL)
                        dfp.insert(0, "fast_grid", FAST_GRID)
                        dfp.insert(0, "run_timestamp", run_timestamp)
                        dfp.insert(0, "run_id", run_id)
                        append_parquet(dfp, path)
                    df_raw.insert(0, "test_label", TEST_LABEL)
                    df_raw.insert(0, "run_timestamp", run_timestamp)
                    append_parquet(df_raw, RESULTS_RAW_FILE)
                    print(f"  ✓ {df_best['team'].iloc[0]} (MA={window_ma})")

    print(f"\nTempo total: {(time.time() - start)/60:.1f} min")
    print(f"Salvo em: {RESULTS_BEST_FILE}")
