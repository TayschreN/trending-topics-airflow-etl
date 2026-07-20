# Instruções para Agentes de IA

## Projeto: trending-topics-airflow-etl

Pipeline ETL de tópicos emergentes com Airflow, NLP e dashboard Dash.

## Stack
- **Orquestração**: Apache Airflow 2.10
- **NLP**: NLTK (tokenização), scikit-learn (TF-IDF), pysentimiento BERT (sentimento)
- **Dashboard**: Dash + Plotly (tema escuro)
- **Dados**: RSS (G1, Folha, CNN Brasil) → Parquet intermediário → Gold tables
- **Infra**: Docker, CI/CD (GitHub Actions), pre-commit, pytest

## Estrutura
```
src/
  extract.py          — coleta RSS
  clean.py            — limpeza de HTML/normalização
  deduplication.py    — dedup por similaridade
  sentiment_analysis.py — pysentimiento BERT (pt)
  nlp_processing.py   — tokenização, TF-IDF, frequência
  trend_detection.py  — z-score vs baseline histórica
  aggregate.py        — merge final + gold tables
dashboard/
  app.py              — Dash app (10 gráficos, 6 KPIs, insights)
  assets/style.css    — tema escuro
config/config.yaml    — RSS feeds, stopwords, parâmetros
dags/trending_topics_pipeline.py — DAG Airflow (11 tasks)
data/gold/           — Parquet gold tables
tests/               — pytest
