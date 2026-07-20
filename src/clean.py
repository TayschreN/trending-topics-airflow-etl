import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


def clean_and_normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["titulo"] = df["titulo"].fillna("").astype(str).str.lower().str.strip()
    df["resumo"] = df["resumo"].fillna("").astype(str).str.lower().str.strip()

    df["texto_completo"] = df["titulo"] + " " + df["resumo"]

    df = df[df["texto_completo"].str.strip() != ""].reset_index(drop=True)

    logger.info(f"Registros após limpeza: {len(df)} (removidos {len(df_raw) - len(df)})")
    return df
