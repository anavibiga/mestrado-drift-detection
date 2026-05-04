"""
Baseline Aleatório (Bernoulli): cada minuto emite alarme com probabilidade ALARM_PROB = 0.5,
sem qualquer hipótese sobre os dados. Avaliação meia-pirâmide / SoftED assimétrico.
Resultados em results/02_model_baseline_random/
"""

# =========================
# 0. IMPORTS
# =========================
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

# =========================
# 1. CONFIG
# =========================
BASE_NAME = "events_wide_minute.parquet"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / BASE_NAME

RESULTS_PATH = BASE_DIR / "results" / "02_model_baseline_random"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_BEST_FILE = RESULTS_PATH / "results_best.parquet"
RESULTS_BEST_BY_SIDE_FILE = RESULTS_PATH / "results_best_by_side.parquet"
RESULTS_RAW_FILE = RESULTS_PATH / "results_raw.parquet"

ALARM_PROB = 0.5   # Bernoulli p=0.5: coin flip por minuto, sem hipótese sobre os dados
RANDOM_SEED = 42   # reprodutibilidade
N_RUNS = 10        # repetições para estimar variância do baseline aleatório

K = 5
N_WORKERS = 7
FEATURES = ["passe", "passe_certo", "passe_errado"]
TASKS = [
    ("attack", "open_play", "gol_open_play"),
    ("defense", "open_play", "gol_open_play"),
]

RUN_DATE = datetime.today().date().isoformat()
TEAMS = None
FORCE_RERUN = True
TEST_LABEL = "baseline aleatorio Bernoulli p=0.5"

# =========================
# 2. FUNCTIONS
# =========================
def random_alarms(n_minutes, p, rng):
    """Dispara True aleatoriamente com probabilidade p por minuto."""
    return rng.random(n_minutes) < p


def har_eval_soft_half_python(drift_series, label_series, k):
    """
    Meia-pirâmide: janela [t-K, t], pico em t-K.
    score(alarme) = 1 - (alarme - (t-K)) / K  se alarme ∈ [t-K, t], senão 0.
    Matching via Hungarian (n>1, m>1) ou max (n>1, m=1) para evitar que dois alarmes
    cubram o mesmo gol — equivalente ao har_eval_soft_assimetrico do R/harbinger.
    """
    from scipy.optimize import linear_sum_assignment

    drift_arr = np.array(drift_series, dtype=float)
    label_arr = np.array(label_series, dtype=float)
    drift_arr = np.where(np.isnan(drift_arr), 0.0, drift_arr).astype(bool)
    label_arr = np.where(np.isnan(label_arr), 0.0, label_arr).astype(bool)

    goal_pos  = np.where(label_arr)[0]
    alarm_pos = np.where(drift_arr)[0]
    effective_n = len(drift_series)

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
                    drift_series = pd.Series(index=df_ctx.index, dtype=bool)
                    for period in sorted(df_ctx["period"].unique()):
                        mask = df_ctx["period"] == period
                        n = mask.sum()
                        alarms = random_alarms(n, alarm_prob, rng)
                        for i, idx in enumerate(df_ctx.loc[mask].index):
                            drift_series.loc[idx] = alarms[i]
                    drift_series = drift_series.astype(bool)
                    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
                    for period in sorted(df_ctx["period"].unique()):
                        mask = df_ctx["period"] == period
                        tp, fp, fn, tn = har_eval_soft_half_python(
                            drift_series[mask], goal_series[mask], k
                        )
                        TP += tp; FP += fp; FN += fn; TN += tn
                    results.append({
                        "run_date": RUN_DATE,
                        "match_id": match_id,
                        "run_number": run_number,
                        "team": team,
                        "task": task,
                        "goal_type": goal_type,
                        "side": side,
                        "feature": feature,
                        "detector": "RandomBernoulli",
                        "params": json.dumps({"alarm_prob": alarm_prob, "seed": seed}),
                        "alarm_prob": alarm_prob,
                        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                    })
    return pd.DataFrame(results)


# =========================
# 3. MAIN
# =========================
if __name__ == "__main__":
    print("=== 02_model: Baseline Aleatório (Bernoulli p=0.5) ===")
    print(f"p = {ALARM_PROB} | Seed: {RANDOM_SEED} | Runs: {N_RUNS}")
    print(f"Resultados: {RESULTS_BEST_FILE}")

    start = time.time()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    run_id = f"baseline_random_{run_timestamp}"

    df = pd.read_parquet(DATA_PATH)
    teams = (
        pd.unique(df[["home_team", "away_team"]].values.ravel()).tolist()
        if TEAMS is None else TEAMS
    )

    pending_teams = teams if FORCE_RERUN else (
        [t for t in teams if t not in set(pd.read_parquet(RESULTS_BEST_FILE)["team"].unique())]
        if RESULTS_BEST_FILE.exists() else teams
    )

    print(f"Times: {len(pending_teams)} | Runs: {N_RUNS} | Workers: {N_WORKERS}")

    for run_number in range(N_RUNS):
        print(f"\n--- Run {run_number + 1}/{N_RUNS} ---")
        args_list = [
            (team, df[(df["home_team"] == team) | (df["away_team"] == team)],
             K, ALARM_PROB, run_number, RANDOM_SEED)
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
                print(f"  ✓ {df_agg['team'].iloc[0]} (run={run_number + 1})")

    print(f"\nTempo total: {(time.time() - start)/60:.1f} min")
    print(f"Salvo em: {RESULTS_BEST_FILE}")
