import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from airflow.models import DagBag


def test_dag_imports_without_errors():
    dagbag = DagBag(
        dag_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dags"),
        include_examples=False,
    )
    assert len(dagbag.import_errors) == 0, (
        f"Erros de importação na DAG: {dagbag.import_errors}"
    )


def test_dag_contains_expected_tasks():
    dagbag = DagBag(
        dag_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dags"),
        include_examples=False,
    )
    dag = dagbag.get_dag(dag_id="trending_topics_pipeline")
    assert dag is not None
    task_ids = [t.task_id for t in dag.tasks]
    expected = [
        "extract_rss_feeds",
        "validate_raw_data",
        "clean_and_normalize_text",
        "remove_duplicates",
        "compute_sentiment_scores",
        "tokenize_and_process",
        "compute_frequency_and_tfidf",
        "detect_emerging_topics",
        "aggregate_and_save_gold",
        "refresh_dashboard_data",
    ]
    for task_id in expected:
        assert task_id in task_ids, f"Task {task_id} não encontrada na DAG"


def test_dag_structure_linear():
    dagbag = DagBag(
        dag_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dags"),
        include_examples=False,
    )
    dag = dagbag.get_dag(dag_id="trending_topics_pipeline")
    assert dag is not None
    assert dag.schedule_interval == "@daily"
    assert dag.catchup == False
