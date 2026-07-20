FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY --chown=airflow:root . /opt/airflow/

ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow

RUN mkdir -p /opt/airflow/data/bronze /opt/airflow/data/silver /opt/airflow/data/gold /opt/airflow/logs /opt/airflow/dashboard/assets
