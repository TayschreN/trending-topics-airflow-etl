import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.deduplication import remove_duplicates


def test_no_duplicates_returns_all():
    df = pd.DataFrame({
        "titulo": ["notícia única A", "notícia única B", "notícia única C"],
        "data_publicacao": ["2026-07-19", "2026-07-20", "2026-07-21"],
    })
    result = remove_duplicates(df, threshold=0.9)
    assert len(result) == 3


def test_removes_duplicates_above_threshold():
    df = pd.DataFrame({
        "titulo": [
            "Governo anuncia novo pacote econômico hoje",
            "Governo anuncia novo pacote econômico hoje",
            "Futebol: time vence campeonato",
        ],
        "data_publicacao": ["2026-07-19", "2026-07-20", "2026-07-21"],
    })
    result = remove_duplicates(df, threshold=0.75)
    assert len(result) == 2


def test_keeps_oldest_when_duplicate():
    df = pd.DataFrame({
        "titulo": [
            "Notícia sobre economia cresce",
            "Notícia sobre economia cresce",
        ],
        "data_publicacao": ["2026-07-19", "2026-07-20"],
    })
    result = remove_duplicates(df, threshold=0.75)
    assert len(result) == 1
    assert result["data_publicacao"].iloc[0] == "2026-07-19"


def test_single_row_returns_same():
    df = pd.DataFrame({
        "titulo": ["notícia única"],
        "data_publicacao": ["2026-07-20"],
    })
    result = remove_duplicates(df, threshold=0.75)
    assert len(result) == 1
