import sys
import os
import pytest
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.aggregate import aggregate_daily_results, _categorizar_termo, _sentimento_pontuado


def test_categorizar_termo_politica():
    assert _categorizar_termo("presidente") == "política"
    assert _categorizar_termo("governo federal") == "política"


def test_categorizar_termo_economia():
    assert _categorizar_termo("inflação") == "economia"
    assert _categorizar_termo("mercado financeiro") == "economia"


def test_categorizar_termo_outros():
    assert _categorizar_termo("receita de bolo") == "outros"
    assert _categorizar_termo("clima hoje") == "outros"


def test_sentimento_pontuado():
    assert _sentimento_pontuado("POS", 0.9) == 0.9
    assert _sentimento_pontuado("NEG", 0.8) == -0.8
    assert _sentimento_pontuado("NEU", 0.7) == 0.0


def test_aggregate_daily_results_creates_gold_files():
    df_noticias = pd.DataFrame({
        "texto_completo": ["governo anuncia pacote econômico", "time vence campeonato de futebol"],
        "sentimento_label": ["POS", "POS"],
        "sentimento_score": [0.9, 0.8],
    })
    df_freq = pd.DataFrame({
        "termo": ["governo", "pacote", "econômico", "time", "campeonato", "futebol"],
        "frequencia": [1, 1, 1, 1, 1, 1],
    })
    df_tfidf = pd.DataFrame({
        "termo": ["governo", "pacote", "econômico", "time", "campeonato", "futebol"],
        "tfidf_score": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    })
    df_emergencia = pd.DataFrame({
        "termo": ["governo", "pacote", "econômico", "time", "campeonato", "futebol"],
        "frequencia": [1, 1, 1, 1, 1, 1],
        "z_score": [3.0, 2.5, 1.0, 0.5, 0.0, 0.0],
        "emergente": [True, True, False, False, False, False],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        result = aggregate_daily_results(
            df_noticias, df_freq, df_tfidf, df_emergencia, "2026-07-20", tmpdir
        )
        assert "gold1_path" in result
        assert "gold2_path" in result
        assert os.path.exists(result["gold1_path"])
        assert os.path.exists(result["gold2_path"])

        df_gold1 = pd.read_parquet(result["gold1_path"])
        assert "termo" in df_gold1.columns
        assert "tema" in df_gold1.columns
        assert "data" in df_gold1.columns
        assert df_gold1["data"].iloc[0] == "2026-07-20"


def test_aggregate_incremental_appends():
    df_noticias = pd.DataFrame({
        "texto_completo": ["governo anuncia pacote"],
        "sentimento_label": ["POS"],
        "sentimento_score": [0.9],
    })
    df_freq = pd.DataFrame({"termo": ["governo", "pacote"], "frequencia": [1, 1]})
    df_tfidf = pd.DataFrame({"termo": ["governo", "pacote"], "tfidf_score": [0.5, 0.4]})
    df_emergencia = pd.DataFrame({
        "termo": ["governo", "pacote"],
        "frequencia": [1, 1],
        "z_score": [3.0, 2.5],
        "emergente": [True, True],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        result1 = aggregate_daily_results(
            df_noticias, df_freq, df_tfidf, df_emergencia, "2026-07-20", tmpdir
        )
        result2 = aggregate_daily_results(
            df_noticias, df_freq, df_tfidf, df_emergencia, "2026-07-21", tmpdir
        )
        df_gold = pd.read_parquet(result2["gold1_path"])
        assert len(df_gold["data"].unique()) == 2
