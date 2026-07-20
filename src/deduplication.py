import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["duplicata_de"] = None

    titulos = df["titulo"].fillna("").tolist()
    if len(titulos) < 2:
        return df

    vectorizer = TfidfVectorizer().fit_transform(titulos)
    sim_matrix = cosine_similarity(vectorizer)

    n = len(df)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pub_i = df.iloc[i].get("data_publicacao", "")
                pub_j = df.iloc[j].get("data_publicacao", "")
                if pub_i and pub_j:
                    if pub_j < pub_i:
                        keep[i] = False
                        df.at[i, "duplicata_de"] = df.at[j, "link"]
                    else:
                        keep[j] = False
                        df.at[j, "duplicata_de"] = df.at[i, "link"]
                else:
                    keep[j] = False
                    df.at[j, "duplicata_de"] = df.at[i, "link"]

    df["duplicata"] = ~pd.Series(keep)
    df_sem_duplicatas = df[keep].reset_index(drop=True)

    logger.info(
        f"Removidas {n - len(df_sem_duplicatas)} duplicatas (threshold={threshold})"
    )
    return df_sem_duplicatas
