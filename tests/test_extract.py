import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.extract import extract_rss_feeds


class MockEntry(dict):
    def __init__(self, title, summary, published, link):
        super().__init__()
        self["title"] = title
        self["summary"] = summary
        self["published"] = published
        self["link"] = link
        self.__dict__ = self


class MockFeed:
    def __init__(self, entries, status=200):
        self.entries = entries
        self.status = status


@patch("src.extract.feedparser.parse")
def test_extract_rss_feeds(mock_parse):
    mock_feeds = [
        MockFeed(entries=[
            MockEntry(
                title="Notícia Teste 1",
                summary="Resumo da notícia teste",
                published="2026-07-19",
                link="http://exemplo.com/1",
            ),
            MockEntry(
                title="Notícia Teste 2",
                summary="<p>Outro resumo</p>",
                published="2026-07-20",
                link="http://exemplo.com/2",
            ),
        ])
    ] * 3

    mock_parse.side_effect = mock_feeds

    config = {
        "fontes_rss": [
            {"nome": "fonte_a", "url": "http://teste.com/a"},
            {"nome": "fonte_b", "url": "http://teste.com/b"},
            {"nome": "fonte_c", "url": "http://teste.com/c"},
        ]
    }

    df = extract_rss_feeds(config, "2026-07-20")

    assert len(df) == 6
    assert list(df.columns) == [
        "titulo", "resumo", "data_publicacao", "link", "fonte", "data_coleta"
    ]
    assert df["data_coleta"].iloc[0] == "2026-07-20"
    assert df["fonte"].nunique() == 3
    assert "<p>" not in df["resumo"].iloc[3]


@patch("src.extract.feedparser.parse")
def test_extract_all_feeds_fail(mock_parse):
    mock_parse.return_value = MockFeed(entries=[])
    config = {
        "fontes_rss": [
            {"nome": "fonte_a", "url": "http://teste.com/a"},
        ]
    }
    with pytest.raises(RuntimeError, match="Todas as fontes"):
        extract_rss_feeds(config, "2026-07-20")
