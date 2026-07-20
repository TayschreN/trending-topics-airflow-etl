import sys
import os
import pytest
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trend_detection import compute_emergence_score, load_historical_baseline


def test_compute_emergence_score_no_history():
    df_hoje = pd.DataFrame({"termo": ["crise", "futebol"], "frequencia": [10, 5]})
    df_baseline = pd.DataFrame(columns=["termo", "media_frequencia", "std_frequencia"])
    result = compute_emergence_score(df_hoje, df_baseline, threshold_zscore=2.0)
    assert "z_score" in result.columns
    assert "emergente" in result.columns
    assert result["emergente"].isna().all()
    assert result["z_score"].iloc[0] == 10


def test_compute_emergence_score_with_history():
    df_hoje = pd.DataFrame({"termo": ["crise", "futebol"], "frequencia": [50, 5]})
    df_baseline = pd.DataFrame({
        "termo": ["crise", "futebol"],
        "media_frequencia": [10.0, 10.0],
        "std_frequencia": [5.0, 5.0],
    })
    result = compute_emergence_score(df_hoje, df_baseline, threshold_zscore=2.0)
    crise_z = result[result["termo"] == "crise"]["z_score"].iloc[0]
    futebol_z = result[result["termo"] == "futebol"]["z_score"].iloc[0]
    assert crise_z > 2.0
    assert futebol_z < 2.0
    assert result[result["termo"] == "crise"]["emergente"].iloc[0] == True
    assert result[result["termo"] == "futebol"]["emergente"].iloc[0] == False


def test_emergente_none_when_no_history():
    df_hoje = pd.DataFrame({"termo": ["novo"], "frequencia": [100]})
    df_baseline = pd.DataFrame(columns=["termo", "media_frequencia", "std_frequencia"])
    result = compute_emergence_score(df_hoje, df_baseline, threshold_zscore=2.0)
    assert result["emergente"].iloc[0] is None


def test_load_historical_baseline_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_historical_baseline(tmpdir, 14, "2026-07-20")
        assert result.empty
        assert list(result.columns) == ["termo", "media_frequencia", "std_frequencia"]
