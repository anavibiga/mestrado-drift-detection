"""
Criação de features para o modelo (FORMATO WIDE)

Objetivo:
- 1 linha = 1 minuto da partida
- Eventos separados casa / fora na MESMA linha
- Cada tempo da partida é uma série temporal distinta (0-45+)
- Série temporal limpa para drift detection

Colunas finais:
match_id
period
minute
home_team
away_team
passe_casa / passe_fora
passe_certo_casa / passe_certo_fora
passe_errado_casa / passe_errado_fora
gol_casa / gol_fora
gol_open_play_casa / gol_open_play_fora
"""

# ==============================
# 0. IMPORTS
# ==============================
import pandas as pd
from pathlib import Path
import time
from datetime import timedelta

start = time.time()

# ==============================
# 1. CONFIG
# ==============================
competition_id = 11
season_id = 27
# min_goal_minute = 0

BASE_DIR = Path(__file__).resolve().parent.parent
base_path = BASE_DIR / "data" / "open-data" / "data"

matches_path = base_path / "matches" / str(competition_id) / f"{season_id}.json"
events_path = base_path / "events"

output_path = BASE_DIR / "data" / "processed"
output_path.mkdir(parents=True, exist_ok=True)

# ==============================
# 2. LOAD MATCHES
# ==============================
df_matches = pd.read_json(matches_path)

# normalizando nomes
df_matches["home_team"] = df_matches["home_team"].apply(
    lambda x: x["home_team_name"]
)
df_matches["away_team"] = df_matches["away_team"].apply(
    lambda x: x["away_team_name"]
)

# ==============================
# 3. LOAD EVENTS
# ==============================
dfs = []

for match_id in df_matches["match_id"]:
    ev = pd.read_json(events_path / f"{match_id}.json")
    ev["match_id"] = match_id
    dfs.append(ev)

df_events = pd.concat(dfs, ignore_index=True)

# ==============================
# 4. NORMALIZAÇÃO
# ==============================
events = df_events.copy()

events["team"] = events["team"].apply(
    lambda x: x["name"] if isinstance(x, dict) else x
)

events["type"] = events["type"].apply(
    lambda x: x["name"] if isinstance(x, dict) else x
)

events["shot_outcome"] = events["shot"].apply(
    lambda x: x.get("outcome", {}).get("name") if isinstance(x, dict) else None
)

events["shot_type"] = events["shot"].apply(
    lambda x: x.get("type", {}).get("name") if isinstance(x, dict) else None
)

events["pass_outcome"] = events["pass"].apply(
    lambda x: x.get("outcome", {}).get("name") if isinstance(x, dict) else None
)

# ==============================
# 5. MERGE CASA/FORA
# ==============================
events = events.merge(
    df_matches[["match_id", "home_team", "away_team"]],
    on="match_id",
    how="left"
)

# Manter apenas eventos atrelados a um time
events = events[
    (events["team"] == events["home_team"]) |
    (events["team"] == events["away_team"])
].copy()

# Classificar os eventos em casa x fora
events["side"] = (events["team"] == events["home_team"]).map(
    {True: "casa", False: "fora"}
)

# ==============================
# 6. MINUTO DENTRO DO PERÍODO
# ==============================
events["minute_in_period"] = events["minute"]

# 2º tempo começa do zero
events.loc[
    events["period"] == 2,
    "minute_in_period"
] = events.loc[events["period"] == 2, "minute"] - 45

# ==============================
# 7. FEATURES
# ==============================
events = events.assign(

    # gols
    gol=((events["type"] == "Shot") &
         (events["shot_outcome"] == "Goal")).astype(int),

    # gols open play = excluindo falta, pênalti e escanteio
    gol_open_play=((events["type"] == "Shot") &
                   (events["shot_outcome"] == "Goal") &
                   (~events["shot_type"].isin(["Free Kick", "Penalty", "Corner"]))).astype(int),

    # passes
    passe=(events["type"] == "Pass").astype(int),

    # passe errado sempre tem o porquê, então o pass_outcome nunca é NA
    passe_errado=((events["type"] == "Pass") &
                  (events["pass_outcome"].notna())).astype(int),

    # passe certo não tem definição de pass_outcome, então sempre é NA
    passe_certo=((events["type"] == "Pass") &
                 (events["pass_outcome"].isna())).astype(int),
)

# ==============================
# 8. AGREGAÇÃO POR MINUTO
# ==============================
agg = (
    events
    .groupby(
        ["match_id", "period", "minute_in_period", "side"],
        as_index=False
    )
    .agg({
        "passe": "sum",
        "passe_certo": "sum",
        "passe_errado": "sum",
        "gol": "sum",
        "gol_open_play": "sum"
    })
)

# ==============================
# 9. PIVOT CASA/FORA
# ==============================
wide = agg.pivot_table(
    index=["match_id", "period", "minute_in_period"],
    columns="side",
    values=[
        "passe",
        "passe_certo",
        "passe_errado",
        "gol",
        "gol_open_play"
    ],
    fill_value=0
)

wide.columns = [f"{c[0]}_{c[1]}" for c in wide.columns]
wide = wide.reset_index()

# ==============================
# 10. GARANTIR TODOS OS MINUTOS
# ==============================

max_minutes = (
    wide
    .groupby(["match_id", "period"])["minute_in_period"]
    .max()
    .reset_index()
)

full_index = pd.concat([
    pd.DataFrame({
        "match_id": row.match_id,
        "period": row.period,
        "minute_in_period": range(int(row.minute_in_period) + 1)
    })
    for row in max_minutes.itertuples(index=False)
], ignore_index=True)

wide = (
    full_index
    .merge(wide, how="left")
    .fillna(0)
)

wide["minute_in_period"] = wide["minute_in_period"].astype(int)

# ==============================
# 11. ADICIONAR TIMES
# ==============================
wide = wide.merge(
    df_matches[["match_id", "home_team", "away_team"]],
    on="match_id",
    how="left"
)

# ==============================
# 12. OUTPUT
# ==============================
wide.to_parquet(
    output_path / "events_wide_minute.parquet",
    index=False
)

print("Base criada com sucesso:", wide.shape)

elapsed = time.time() - start
print("Tempo total:", str(timedelta(seconds=int(elapsed))))
