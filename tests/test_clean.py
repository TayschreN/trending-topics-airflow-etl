import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.clean import clean_and_normalize


def test_clean_lowercases_and_strips():
    df = pd.DataFrame({
        "titulo": ["  Notícia IMPORTANTE  "],
        "resumo": ["  Resumo DETALHADO  "],
    })
    result = clean_and_normalize(df)
    assert result["titulo"].iloc[0] == "notícia importante"
    assert result["resumo"].iloc[0] == "resumo detalhado"


def test_clean_creates_texto_completo():
    df = pd.DataFrame({
        "titulo": ["Título A"],
        "resumo": ["Resumo B"],
    })
    result = clean_and_normalize(df)
    assert "texto_completo" in result.columns
    assert "título a resumo b" in result["texto_completo"].iloc[0]


def test_clean_removes_empty_rows():
    df = pd.DataFrame({
        "titulo": ["válido", "", "  "],
        "resumo": ["notícia", "", "   "],
    })
    result = clean_and_normalize(df)
    assert len(result) == 1


def test_clean_handles_nulls():
    df = pd.DataFrame({
        "titulo": [None, "válido"],
        "resumo": ["notícia", None],
    })
    result = clean_and_normalize(df)
    assert len(result) == 2
    assert result["texto_completo"].iloc[0] == " notícia"
