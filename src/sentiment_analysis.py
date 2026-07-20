import pandas as pd
import logging
from pysentimiento import create_analyzer

logger = logging.getLogger(__name__)

_analyzer_cache = None


def get_sentiment_analyzer():
    global _analyzer_cache
    if _analyzer_cache is None:
        logger.info("Carregando analisador de sentimento pysentimiento...")
        _analyzer_cache = create_analyzer(task="sentiment", lang="pt")
    return _analyzer_cache


def compute_sentiment(
    df: pd.DataFrame, analyzer, texto_col: str = "texto_completo"
) -> pd.DataFrame:
    df = df.copy()
    textos = df[texto_col].fillna("").tolist()

    resultados = analyzer.predict(textos)

    df["sentimento_label"] = [r.output for r in resultados]
    df["sentimento_score"] = [r.probas[r.output] for r in resultados]

    logger.info(f"Sentimento calculado para {len(df)} registros")
    return df
