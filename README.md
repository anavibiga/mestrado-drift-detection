# Detecção de Drift em Dados de Futebol

Repositório do trabalho de mestrado (PPCIC - CEFET/RJ) que investiga o uso de algoritmos de **detecção de concept drift** para identificar mudanças no padrão de eventos de partidas de futebol próximas a gols.

## Visão geral

O problema é formulado como detecção de drift em séries temporais: dado o volume de eventos por minuto (passes, chutes, pressões etc.) de cada time ao longo de uma partida, os detectores devem sinalizar uma mudança de comportamento (drift) que corresponda à ocorrência de um gol.

**Competição:** La Liga — temporada 2015/16 (StatsBomb Open Data)

## Estrutura do repositório

```
.
├── data/
│   ├── open-data/          # StatsBomb Open Data (ver nota abaixo)
│   └── processed/          # Dados processados (formato wide por minuto)
├── src/
│   ├── 01_feature_engineering.py       # Gera events_wide_minute.parquet
│   ├── 02_model_adwin.py               # Detector ADWIN
│   ├── 02_model_kswin.py               # Detector KSWIN
│   ├── 02_model_page_hinkley.py        # Detector Page-Hinkley
│   ├── 02_model_baseline_fixed.py      # Baseline: limiar fixo
│   └── 02_model_baseline_randomwalk.py # Baseline: random walk
├── notebooks/
│   ├── 01_data_sanity_check.ipynb      # Validação dos dados brutos
│   ├── 02_eda_laliga.ipynb             # Análise exploratória
│   └── 03_analise_resultados.ipynb     # Avaliação dos detectores
├── results/                            # Resultados por modelo (.parquet)
└── figures/                            # Figuras geradas para a dissertação
```

### Sobre os dados (`data/`)

Os dados brutos são provenientes do [repositório público do StatsBomb](https://github.com/statsbomb/open-data). Não foram utilizados todos os arquivos disponíveis — apenas os referentes à **La Liga, temporada 2015/2016**. A pasta `data/open-data/` foi omitida deste repositório por ter aproximadamente **9 GB**; para reproduzir os experimentos, clone o repositório do StatsBomb e coloque os arquivos no caminho esperado.

## Pipeline

1. **Feature engineering** (`src/01_feature_engineering.py`)
   Converte os eventos brutos do StatsBomb para uma tabela wide com 1 linha por minuto por partida, separando eventos de casa e fora. Saída: `data/processed/events_wide_minute.parquet`.

2. **Execução dos modelos** (`src/02_model_*.py`)
   Cada script roda um detector sobre a série temporal de eventos, usando uma janela deslizante com avaliação assimétrica (meia-pirâmide / SoftEd). Resultados salvos em `results/<modelo>/`.

3. **Análise** (`notebooks/03_analise_resultados.ipynb`)
   Compara os detectores via F1, MCC, curva precision-recall e análise de alarmes falsos.

## Detectores avaliados

| Detector | Biblioteca |
|---|---|
| ADWIN | `river` |
| KSWIN | `river` |
| Page-Hinkley | `river` |
| Baseline fixo | — |
| Baseline random walk | — |

## Features

- `passe` — total de passes por minuto
- `passe_certo` — passes certos por minuto
- `passe_errado` — passes errados por minuto

## Dependências principais

- Python 3.10+
- `pandas`, `numpy`, `scipy`
- `river` (detectores de drift)
- `matplotlib` (figuras)
- `pyarrow` / `fastparquet` (leitura/escrita de parquet)

## Dados

Os dados brutos são do [StatsBomb Open Data](https://github.com/statsbomb/open-data) e estão sujeitos à licença StatsBomb. Não devem ser redistribuídos sem autorização.
