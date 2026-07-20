import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nlp_processing import tokenize_and_clean, compute_word_frequency, compute_tfidf


def test_tokenize_and_clean():
    texto = "O presidente disse que a economia vai bem segundo analistas"
    tokens = tokenize_and_clean(texto, ["segundo", "disse"])
    assert "presidente" in tokens
    assert "economia" in tokens
    assert "disse" not in tokens
    assert "segundo" not in tokens
    assert "o" not in tokens
    assert "que" not in tokens
    for t in tokens:
        assert t.isalpha()


def test_tokenize_and_clean_removes_numbers():
    texto = "notícia 123 sobre futebol"
    tokens = tokenize_and_clean(texto, [])
    assert "notícia" in tokens
    assert "futebol" in tokens
    assert "123" not in tokens


def test_compute_word_frequency():
    lista = [["gato", "cachorro"], ["gato", "pássaro"], ["cachorro", "peixe"]]
    df = compute_word_frequency(lista)
    freq = dict(zip(df["termo"], df["frequencia"]))
    assert freq["gato"] == 2
    assert freq["cachorro"] == 2
    assert freq["pássaro"] == 1


def test_compute_tfidf_empty_below_minimum():
    corpus = ["apenas um documento"]
    df = compute_tfidf(corpus, corpus_minimo=5)
    assert df.empty
    assert list(df.columns) == ["termo", "tfidf_score"]


def test_compute_tfidf_returns_results():
    corpus = [
        "o gato subiu no telhado hoje",
        "o cachorro correu no parque hoje",
        "o gato e o cachorro são amigos hoje",
    ]
    df = compute_tfidf(corpus, corpus_minimo=2)
    assert not df.empty
    assert "termo" in df.columns
    assert "tfidf_score" in df.columns
    assert len(df) <= 200
