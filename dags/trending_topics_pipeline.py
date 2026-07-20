import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import logging

from src.extract import extract_rss_feeds
from src.clean import clean_and_normalize
from src.deduplication import remove_duplicates
from src.sentiment_analysis import get_sentiment_analyzer, compute_sentiment
from src.nlp_processing import tokenize_and_clean, compute_word_frequency, compute_tfidf
from src.trend_detection import load_historical_baseline, compute_emergence_score
from src.aggregate import aggregate_daily_results

import yaml

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
SILVER_PATH = os.path.join(BASE_DIR, "data", "silver")
GOLD_PATH = os.path.join(BASE_DIR, "data", "gold")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def _silver_path(task_id: str, data_exec: str) -> str:
    os.makedirs(SILVER_PATH, exist_ok=True)
    return os.path.join(SILVER_PATH, f"{task_id}_{data_exec}.parquet")


def _extract(**context):
    data_exec = context["ds"]
    df = extract_rss_feeds(config, data_exec)
    path = _silver_path("extract", data_exec)
    df.to_parquet(path, index=False)
    return path


def _validate(**context):
    path = context["ti"].xcom_pull(task_ids="extract_rss_feeds")
    if not path or not os.path.exists(path):
        raise AirflowException("Nenhum dado extraído — falha na task anterior")
    logger.info("Validação OK: dados recebidos em %s", path)
    return path


def _clean(**context):
    path = context["ti"].xcom_pull(task_ids="validate_raw_data")
    import pandas as pd
    df = pd.read_parquet(path)
    df = clean_and_normalize(df)
    out_path = _silver_path("clean", context["ds"])
    df.to_parquet(out_path, index=False)
    return out_path


def _deduplicate(**context):
    path = context["ti"].xcom_pull(task_ids="clean_and_normalize_text")
    import pandas as pd
    df = pd.read_parquet(path)
    threshold = config["threshold_similaridade_dedup"]
    df = remove_duplicates(df, threshold)
    out_path = _silver_path("dedup", context["ds"])
    df.to_parquet(out_path, index=False)
    return out_path


def _sentiment(**context):
    path = context["ti"].xcom_pull(task_ids="remove_duplicates")
    import pandas as pd
    df = pd.read_parquet(path)
    analyzer = get_sentiment_analyzer()
    df = compute_sentiment(df, analyzer)
    out_path = _silver_path("sentiment", context["ds"])
    df.to_parquet(out_path, index=False)
    return out_path


def _tokenize_and_process(**context):
    path = context["ti"].xcom_pull(task_ids="compute_sentiment_scores")
    import pandas as pd
    df = pd.read_parquet(path)
    textos = df["texto_completo"].fillna("").tolist()
    stopwords_custom = config.get("stopwords_customizadas", [])
    lista_tokens = [tokenize_and_clean(t, stopwords_custom) for t in textos]
    df_tokens = pd.DataFrame({"texto_completo": textos, "tokens": lista_tokens})
    out_path = _silver_path("tokens", context["ds"])
    df_tokens.to_parquet(out_path, index=False)
    return out_path


def _frequency_and_tfidf(**context):
    path = context["ti"].xcom_pull(task_ids="tokenize_and_process")
    import pandas as pd
    df_tokens = pd.read_parquet(path)
    lista_tokens = df_tokens["tokens"].tolist()
    corpus = df_tokens["texto_completo"].tolist()
    df_freq = compute_word_frequency(lista_tokens)
    corpus_minimo = config["corpus_minimo_tfidf"]
    df_tfidf = compute_tfidf(corpus, corpus_minimo)
    freq_path = _silver_path("freq", context["ds"])
    tfidf_path = _silver_path("tfidf", context["ds"])
    df_freq.to_parquet(freq_path, index=False)
    df_tfidf.to_parquet(tfidf_path, index=False)
    import json
    return json.dumps({"freq_path": freq_path, "tfidf_path": tfidf_path})


def _detect_emerging(**context):
    import json
    data_exec = context["ds"]
    json_data = context["ti"].xcom_pull(task_ids="compute_frequency_and_tfidf")
    parsed = json.loads(json_data)
    import pandas as pd
    df_freq = pd.read_parquet(parsed["freq_path"])

    gold_path = GOLD_PATH
    janela = config["janela_historica_dias"]
    baseline = load_historical_baseline(gold_path, janela, data_exec)
    threshold = config["threshold_zscore_emergente"]
    df_emerg = compute_emergence_score(df_freq, baseline, threshold)
    emerg_path = _silver_path("emerging", data_exec)
    df_emerg.to_parquet(emerg_path, index=False)
    result = {
        "emerg_path": emerg_path,
        "tfidf_path": parsed["tfidf_path"],
    }
    import json
    return json.dumps(result)


def _aggregate(**context):
    import json
    import pandas as pd
    data_exec = context["ds"]

    sentiment_path = context["ti"].xcom_pull(task_ids="compute_sentiment_scores")
    df_noticias = pd.read_parquet(sentiment_path)

    json_data = context["ti"].xcom_pull(task_ids="detect_emerging_topics")
    parsed = json.loads(json_data)
    df_emerg = pd.read_parquet(parsed["emerg_path"])
    df_tfidf = pd.read_parquet(parsed["tfidf_path"])

    tokens_path = context["ti"].xcom_pull(task_ids="tokenize_and_process")
    df_tokens = pd.read_parquet(tokens_path)
    df_freq = compute_word_frequency(df_tokens["tokens"].tolist())

    result = aggregate_daily_results(
        df_noticias, df_freq, df_tfidf, df_emerg, data_exec, GOLD_PATH
    )
    logger.info(f"Agregação concluída: {result}")
    return json.dumps(result)


def _refresh_dashboard(**context):
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    gold_file = os.path.join(GOLD_PATH, "daily_trending_topics.parquet")
    if not os.path.exists(gold_file):
        logger.warning("Gold file não encontrado — pulando wordcloud")
        return

    df = pd.read_parquet(gold_file)
    latest = df["data"].max()
    df_hoje = df[df["data"] == latest]

    if df_hoje.empty:
        logger.warning("Sem dados para gerar wordcloud")
        return

    freq_dict = dict(zip(df_hoje["termo"], df_hoje["frequencia"]))
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100)
    wc.generate_from_frequencies(freq_dict)

    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard",
        "assets",
    )
    os.makedirs(assets_dir, exist_ok=True)
    output_path = os.path.join(assets_dir, "wordcloud_hoje.png")
    wc.to_file(output_path)
    logger.info(f"Wordcloud salva em {output_path}")

    plt.close("all")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": None,
}

with DAG(
    dag_id="trending_topics_pipeline",
    default_args=default_args,
    description="Pipeline ETL: coleta RSS → NLP → sentimento → tópicos emergentes",
    schedule_interval="@daily",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["nlp", "sentiment", "trending-topics"],
) as dag:

    t_extract = PythonOperator(
        task_id="extract_rss_feeds",
        python_callable=_extract,
    )

    t_validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=_validate,
    )

    t_clean = PythonOperator(
        task_id="clean_and_normalize_text",
        python_callable=_clean,
    )

    t_dedup = PythonOperator(
        task_id="remove_duplicates",
        python_callable=_deduplicate,
    )

    t_sentiment = PythonOperator(
        task_id="compute_sentiment_scores",
        python_callable=_sentiment,
    )

    t_tokenize = PythonOperator(
        task_id="tokenize_and_process",
        python_callable=_tokenize_and_process,
    )

    t_freq = PythonOperator(
        task_id="compute_frequency_and_tfidf",
        python_callable=_frequency_and_tfidf,
    )

    t_emerging = PythonOperator(
        task_id="detect_emerging_topics",
        python_callable=_detect_emerging,
    )

    t_aggregate = PythonOperator(
        task_id="aggregate_and_save_gold",
        python_callable=_aggregate,
    )

    t_refresh = PythonOperator(
        task_id="refresh_dashboard_data",
        python_callable=_refresh_dashboard,
    )

    t_extract >> t_validate >> t_clean >> t_dedup >> t_sentiment >> t_tokenize >> t_freq >> t_emerging >> t_aggregate >> t_refresh
