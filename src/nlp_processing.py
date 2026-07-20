import nltk
import re
import pandas as pd
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def tokenize_and_clean(texto: str, stopwords_customizadas: list) -> list[str]:
    stopwords = set(nltk.corpus.stopwords.words("portuguese") + stopwords_customizadas)
    tokens = nltk.word_tokenize(texto.lower(), language="portuguese")
    tokens = [
        t
        for t in tokens
        if re.match(r"^[a-zà-ú]+$", t) and t not in stopwords and len(t) > 2
    ]
    return tokens


def compute_word_frequency(
    lista_tokens_por_documento: list[list[str]],
) -> pd.DataFrame:
    counter = Counter()
    for doc_tokens in lista_tokens_por_documento:
        counter.update(set(doc_tokens))
    df_freq = pd.DataFrame(counter.most_common(), columns=["termo", "frequencia"])
    return df_freq


def compute_tfidf(corpus: list[str], corpus_minimo: int) -> pd.DataFrame:
    if len(corpus) < corpus_minimo:
        logger.warning(
            f"Corpus muito pequeno ({len(corpus)} docs, mínimo={corpus_minimo}) — "
            "pulando TF-IDF"
        )
        return pd.DataFrame(columns=["termo", "tfidf_score"])

    vectorizer = TfidfVectorizer(max_features=200, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1

    df_tfidf = pd.DataFrame({"termo": feature_names, "tfidf_score": mean_scores})
    df_tfidf = df_tfidf.sort_values("tfidf_score", ascending=False).reset_index(
        drop=True
    )
    return df_tfidf
