import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sentiment_analysis import compute_sentiment


class MockResult:
    def __init__(self, output, probas_dict):
        self.output = output
        self.probas = probas_dict


class MockAnalyzer:
    def predict(self, textos):
        mapping = {
            "ótima notícia vitória histórica": MockResult("POS", {"POS": 0.95, "NEG": 0.02, "NEU": 0.03}),
            "tragédia deixa vítimas": MockResult("NEG", {"NEG": 0.92, "POS": 0.03, "NEU": 0.05}),
            "o congresso aprovou a lei": MockResult("NEU", {"NEU": 0.80, "POS": 0.10, "NEG": 0.10}),
        }
        results = []
        for t in textos:
            t_lower = t.lower().strip()
            if t_lower in mapping:
                results.append(mapping[t_lower])
            else:
                results.append(MockResult("NEU", {"NEU": 0.6, "POS": 0.2, "NEG": 0.2}))
        return results


def test_compute_sentiment_positive():
    analyzer = MockAnalyzer()
    df = pd.DataFrame({"texto_completo": ["ótima notícia vitória histórica"]})
    df = compute_sentiment(df, analyzer)
    assert df["sentimento_label"].iloc[0] == "POS"
    assert df["sentimento_score"].iloc[0] >= 0.9


def test_compute_sentiment_negative():
    analyzer = MockAnalyzer()
    df = pd.DataFrame({"texto_completo": ["tragédia deixa vítimas"]})
    df = compute_sentiment(df, analyzer)
    assert df["sentimento_label"].iloc[0] == "NEG"
    assert df["sentimento_score"].iloc[0] >= 0.9


def test_compute_sentiment_neutral():
    analyzer = MockAnalyzer()
    df = pd.DataFrame({"texto_completo": ["o congresso aprovou a lei"]})
    df = compute_sentiment(df, analyzer)
    assert df["sentimento_label"].iloc[0] == "NEU"
    assert df["sentimento_score"].iloc[0] >= 0.7


def test_compute_sentiment_adds_columns():
    analyzer = MockAnalyzer()
    df = pd.DataFrame({"texto_completo": ["ótima notícia vitória histórica"]})
    df_result = compute_sentiment(df, analyzer)
    assert "sentimento_label" in df_result.columns
    assert "sentimento_score" in df_result.columns
