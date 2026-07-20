import feedparser
import pandas as pd
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


def extract_rss_feeds(config: dict, data_execucao: str) -> pd.DataFrame:
    todas_noticias = []

    for fonte in config["fontes_rss"]:
        nome = fonte["nome"]
        url = fonte["url"]
        logger.info(f"Extraindo feed: {nome} - {url}")

        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                logger.warning(f"Feed {nome} retornou 0 entradas")
                continue

            for entry in feed.entries:
                titulo = entry.get("title", "")
                resumo_raw = entry.get("summary", "")
                resumo = re.sub(r"<[^>]+>", "", resumo_raw) if resumo_raw else ""
                data_pub = entry.get("published", "")
                link = entry.get("link", "")

                todas_noticias.append({
                    "titulo": titulo,
                    "resumo": resumo,
                    "data_publicacao": data_pub,
                    "link": link,
                    "fonte": nome,
                    "data_coleta": data_execucao,
                })

            logger.info(f"  {len(feed.entries)} notícias extraídas de {nome}")

        except Exception as e:
            logger.error(f"Erro ao processar feed {nome}: {e}")
            continue

    if not todas_noticias:
        raise RuntimeError("Todas as fontes RSS falharam — nenhuma notícia foi extraída")

    df = pd.DataFrame(todas_noticias)
    caminho = f"data/bronze/news_raw_{data_execucao}.parquet"
    df.to_parquet(caminho, index=False)
    logger.info(f"Salvo em {caminho} — {len(df)} registros")
    return df
