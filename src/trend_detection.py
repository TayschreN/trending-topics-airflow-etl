import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def load_historical_baseline(
    gold_path: str, janela_dias: int, data_atual: str
) -> pd.DataFrame:
    gold_file = os.path.join(gold_path, "daily_trending_topics.parquet")
    if not os.path.exists(gold_file):
        logger.info("Arquivo gold ainda não existe — histórico vazio")
        return pd.DataFrame(columns=["termo", "media_frequencia", "std_frequencia"])

    df_gold = pd.read_parquet(gold_file)
    df_gold = df_gold[df_gold["data"] < data_atual]
    df_gold = df_gold[df_gold["data"] >= data_atual - pd.Timedelta(days=janela_dias)]

    if df_gold.empty:
        logger.info("Nenhum dado histórico disponível na janela")
        return pd.DataFrame(columns=["termo", "media_frequencia", "std_frequencia"])

    baseline = (
        df_gold.groupby("termo")["frequencia"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "media_frequencia", "std": "std_frequencia"})
    )
    return baseline


def compute_emergence_score(
    df_hoje: pd.DataFrame, df_baseline: pd.DataFrame, threshold_zscore: float
) -> pd.DataFrame:
    df = df_hoje.copy()

    if df_baseline.empty:
        df["z_score"] = df["frequencia"]
        df["emergente"] = None
        return df

    df = df.merge(
        df_baseline[["termo", "media_frequencia", "std_frequencia"]],
        on="termo",
        how="left",
    )

    mask_no_history = df["media_frequencia"].isna()
    mask_with_history = ~mask_no_history

    df.loc[mask_with_history, "z_score"] = (
        df.loc[mask_with_history, "frequencia"]
        - df.loc[mask_with_history, "media_frequencia"]
    ) / (df.loc[mask_with_history, "std_frequencia"] + 1e-6)

    df.loc[mask_no_history, "z_score"] = df.loc[mask_no_history, "frequencia"]
    df.loc[mask_with_history, "emergente"] = (
        df.loc[mask_with_history, "z_score"] >= threshold_zscore
    )
    df.loc[mask_no_history, "emergente"] = None

    df = df.drop(columns=["media_frequencia", "std_frequencia"], errors="ignore")
    return df
