"""
Adwin: avaliação meia-pirâmide / SoftEd assimétrico - (janela [t-K, t], pico em t-K).
Resultados em results/02_model_adwin/
Análise em notebooks/
"""

# =========================
# 0. IMPORTS
# =========================
import pandas as pd
import numpy as np
import json
import time
import hashlib
import os
from pathlib import Path
from itertools import product
from datetime import datetime
from multiprocessing import Pool

from river import drift
from scipy.optimize import linear_sum_assignment

# =========================
# 1. CONFIG
# =========================
BASE_NAME = "events_wide_minute.parquet"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / BASE_NAME

RESULTS_PATH = BASE_DIR / "results" / "02_model_adwin"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_BEST_FILE = RESULTS_PATH / "results_best.parquet"
RESULTS_BEST_BY_SIDE_FILE = RESULTS_PATH / "results_best_by_side.parquet"
RESULTS_RAW_FILE = RESULTS_PATH / "results_raw.parquet"

# Configurações do Warmup e do Cooldown
WINDOW_MA_OPTIONS = [3, 5, 10, 12]
COOLDOWN_OPTIONS = [3, 5]

K = 5

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
TEST_LABEL = "ADWIN - TP/FP/TN/FN e window_ma corrigidos"
# True = grid reduzido (1 delta) para tuning rápido; False = grid completo
FAST_GRID = False

# ==============================
# 2. FUNCTIONS
# ==============================
def generate_experiment_id(base, task, feature, detector, params, k, tag):
    raw = f"{base}|{task}|{feature}|{detector}|{params}|k={k}|minute_in_period|{tag}"
    return hashlib.md5(raw.encode()).hexdigest()


def compute_ma_per_period(series_per_period, window_ma=10):
    arr = np.asarray(series_per_period, dtype=float)
    return pd.Series(arr).rolling(window=window_ma, min_periods=1).mean().values


def drift_detection_on_ma(ma_values, params, cooldown_minutes=10, window_ma=10, label_values=None):
    """
    ADWIN na série já suavizada (MA) + cooldown + warmup.
    Só alimenta o detector a partir do índice window_ma (evita pico quando a MA completa), ou seja, window_ma funciona como warmup.
    Se label_values for fornecido, o cooldown é zerado no minuto de cada gol para permitir
    que o detector volte a alarmar imediatamente após o evento.
    """
    detector = drift.ADWIN(**params)
    out_drift = []
    cooldown_remaining = 0
    ma_start = window_ma

    for i, v in enumerate(ma_values):
        if label_values is not None and label_values[i]:
            cooldown_remaining = 0
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1
        if i >= ma_start:
            detector.update(v)
            raw_drift = detector.drift_detected
            if raw_drift and cooldown_remaining == 0:
                out_drift.append(True)
                cooldown_remaining = cooldown_minutes
            else:
                out_drift.append(False)
        else:
            out_drift.append(False)
    return out_drift


def har_eval_soft_half_python(drift_series, label_series, k, window_ma=0):
    """
    Meia-pirâmide: janela [t-K, t], pico em t-K.
    score(alarme) = 1 - (alarme - (t-K)) / K  se alarme ∈ [t-K, t], senão 0.
    Matching via Hungarian (n>1, m>1) ou max (n>1, m=1) para evitar que dois alarmes
    cubram o mesmo gol — equivalente ao har_eval_soft_assimetrico do R/harbinger.
    Os primeiros window_ma minutos de cada período são excluídos da análise (warmup da MA).
    """

    drift_arr = np.array(drift_series, dtype=float)
    label_arr = np.array(label_series, dtype=float)
    drift_arr = np.where(np.isnan(drift_arr), 0.0, drift_arr).astype(bool)
    label_arr = np.where(np.isnan(label_arr), 0.0, label_arr).astype(bool)

    goal_pos  = np.where(label_arr[window_ma:])[0] + window_ma
    alarm_pos = np.where(drift_arr[window_ma:])[0] + window_ma
    effective_n = len(drift_series) - window_ma

    if len(alarm_pos) == 0 or len(goal_pos) == 0:
        TP, FP = 0.0, float(len(alarm_pos))
        FN = float(len(goal_pos))
        TN = float(effective_n - len(goal_pos)) - FP
        return TP, FP, FN, TN

    def _score(a, t):
        return 1.0 - (a - (t - k)) / k if (t - k) <= a <= t else 0.0

    segs = sorted((int(t) - k, int(t)) for t in goal_pos)
    merged, lo, hi = [], segs[0][0], segs[0][1]
    for s, e in segs[1:]:
        if s <= hi:
            hi = max(hi, e)
        else:
            merged.append((lo, hi))
            lo, hi = s, e
    merged.append((lo, hi))

    S_d = np.zeros(len(alarm_pos))

    for seg_lo, seg_hi in merged:
        ai = np.where((alarm_pos >= seg_lo) & (alarm_pos <= seg_hi))[0]
        gi = np.where((goal_pos  >= seg_lo) & (goal_pos  <= seg_hi))[0]
        if len(ai) == 0 or len(gi) == 0:
            continue

        D, E = alarm_pos[ai], goal_pos[gi]
        M = np.array([[_score(d, t) for t in E] for d in D])

        if len(D) == 1:
            S_d[ai[0]] = M[0].max()
        elif len(E) == 1:
            best = int(M[:, 0].argmax())
            S_d[ai[best]] = M[best, 0]
        else:
            rows, cols = linear_sum_assignment(-M)
            for r, c in zip(rows, cols):
                S_d[ai[r]] = M[r, c]

    TP = float(S_d.sum())
    FP = float((1.0 - S_d).sum())
    FN = float(len(goal_pos)) - TP
    TN = float(effective_n - len(goal_pos)) - FP
    return TP, FP, FN, TN


def get_detector_products(fast_grid=False):
    """
    ADWIN(delta). O parâmetro delta controla a sensibilidade:
    valores menores → menos alarmes (mais conservador);
    valores maiores → mais sensível (mais alertas).

    Se fast_grid=True, usa apenas 1 delta para tuning rápido.
    """
    if fast_grid:
        deltas = [0.002]
    else:
        deltas = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
    return [
        {"detector": "ADWIN", "params": {"delta": delta}}
        for delta in deltas
    ]


def append_parquet(df_new, path):
    if df_new.empty:
        return
    if path.exists():
        df_old = pd.read_parquet(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    first_cols = []
    for col in ["run_id", "run_timestamp", "fast_grid", "test_label", "window_ma", "cooldown_minutes", "side"]:
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
    # fillna(0) trata os casos
    # TP_sum + FP_sum = 0 no precision
    # TP_sum + FN_sum = 0 no recall
    # precision + recall = 0 no f1
    df_agg = df_agg.fillna(0)
    best_idx = df_agg.groupby(group_cols)["f1"].idxmax()
    return df_agg.loc[best_idx].copy()


def _worker_run_team(args):
    team, df_team, detector_products, k, window_ma, cooldown_minutes = args
    res = run_team(df_team, team, detector_products, k, window_ma, cooldown_minutes)
    res["params"] = res["params"].astype(str)
    df_agg = (
        res.groupby(["team", "task", "goal_type", "feature", "detector", "params", "window_ma", "cooldown_minutes"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_best = _agg_best(df_agg, ["team", "task", "goal_type"])
    df_agg_side = (
        res.groupby(["team", "task", "goal_type", "side", "feature", "detector", "params", "window_ma", "cooldown_minutes"], as_index=False)
        .agg(TP_sum=("TP", "sum"), FP_sum=("FP", "sum"), FN_sum=("FN", "sum"), TN_sum=("TN", "sum"))
    )
    df_best_side = _agg_best(df_agg_side, ["team", "task", "goal_type", "side"])
    return df_best, df_best_side, res


def run_team(df_team, team, detector_products, k, window_ma, cooldown_minutes):
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
                            drifts = drift_detection_on_ma(
                                ma_vals, det["params"], cooldown_minutes, window_ma,
                                label_values=goal_series.loc[mask].tolist()
                            )
                            for i, idx in enumerate(df_ctx.loc[mask].index):
                                drift_series.loc[idx] = drifts[i]
                        drift_series = drift_series.astype(bool)
                        TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
                        for period in sorted(df_ctx["period"].unique()):
                            mask = df_ctx["period"] == period
                            tp, fp, fn, tn = har_eval_soft_half_python(
                                drift_series[mask], goal_series[mask], k, window_ma
                            )
                            TP += tp; FP += fp; FN += fn; TN += tn
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
                            "cooldown_minutes": cooldown_minutes,
                            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                        })
    return pd.DataFrame(results)

# ==============================
# 3. MAIN
# ==============================
if __name__ == "__main__":
    n_combos = len(WINDOW_MA_OPTIONS) * len(COOLDOWN_OPTIONS)
    print("=== 02_model: ADWIN | janela MA × cooldown (igual v4/v5) ===")
    print(f"Drift (MA): {WINDOW_MA_OPTIONS} | Cooldown: {COOLDOWN_OPTIONS} | Combinações: {n_combos}")
    print(f"Resultados: {RESULTS_BEST_FILE}")
    print(f"Por lado: {RESULTS_BEST_BY_SIDE_FILE}")
    detector_products = get_detector_products(fast_grid=FAST_GRID)
    print(f"Grid ADWIN: {len(detector_products)} produtos de parâmetros" + (" (FAST_GRID)" if FAST_GRID else ""))

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

    total_work = n_combos * len(pending_teams)
    print(f"Times: {len(pending_teams)} | Total: {total_work} (workers={N_WORKERS})")

    if not pending_teams:
        print("Nada a processar.")
    else:
        for window_ma, cooldown in product(WINDOW_MA_OPTIONS, COOLDOWN_OPTIONS):
            print(f"\n--- MA={window_ma} | Cooldown={cooldown} ---")
            args_list = [
                (team, df[(df["home_team"] == team) | (df["away_team"] == team)], detector_products, K, window_ma, cooldown)
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
                    print(f"  ✓ {df_best['team'].iloc[0]} (MA={window_ma}, cd={cooldown})")

    print(f"\nTempo total: {(time.time() - start)/60:.1f} min")
    print(f"Salvo em: {RESULTS_BEST_FILE}")