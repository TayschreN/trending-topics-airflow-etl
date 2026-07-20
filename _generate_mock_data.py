import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gold")
os.makedirs(GOLD_PATH, exist_ok=True)

random.seed(42)
np.random.seed(42)

termos_politica = ["governo", "presidente", "eleição", "congresso", "senado", "ministro", "partido", "voto", "câmara", "político", "reforma", "constituição", "oposição", "base aliada", "STF", "PL", "PT", "União Brasil", "Poder", "estado"]
termos_economia = ["economia", "inflação", "PIB", "dólar", "bolsa", "mercado", "juros", "orçamento", "bancos", "investimento", "câmbio", "fiscal", "monetário", "comércio", "crise", "crescimento", "dívida", "importação", "exportação", "emprego"]
termos_esporte = ["futebol", "copa", "olímpiadas", "campeonato", "jogador", "time", "medalha", "técnico", "seleção", "brasileirão", "libertadores", "final", "gol", "artilheiro", "estádio", "Neymar", "Vini Jr", "Endrick", "basquete", "vôlei"]
termos_tecnologia = ["tecnologia", "inteligência artificial", "internet", "software", "dados", "digital", "aplicativo", "startup", "inovação", "computador", "IA", "machine learning", "blockchain", "nuvem", "cyber", "robô", "automação", "metaverso", "5G", "openAI"]
termos_saude = ["saúde", "vacina", "hospital", "doença", "tratamento", "médico", "covid", "SUS", "medicamento", "cirurgia", "dengue", "câncer", "coração", "emergência", "psicológico", "nutrição", "epidemia", "clínico", "paciente", "remédio"]

todos_termos = termos_politica + termos_economia + termos_esporte + termos_tecnologia + termos_saude
base_datas = [datetime(2026, 7, 6) + timedelta(days=i) for i in range(14)]

rows_topics = []
rows_sentiment = []

for data in base_datas:
    data_str = data.strftime("%Y-%m-%d")
    num_termos_dia = random.randint(30, 50)
    termos_escolhidos = random.sample(todos_termos, min(num_termos_dia, len(todos_termos)))

    sentimento_tema = {}

    for termo in termos_escolhidos:
        freq_base = random.randint(1, 15)
        if termo in termos_politica:
            freq = freq_base + random.randint(0, 5)
        elif termo in termos_esporte:
            freq = freq_base + random.randint(0, 8)
        else:
            freq = freq_base

        tfidf = round(random.uniform(0.01, 0.35), 4)
        z_score = round(random.uniform(-1.5, 3.5), 2)

        if termo in termos_politica:
            tema = "política"
        elif termo in termos_economia:
            tema = "economia"
        elif termo in termos_esporte:
            tema = "esporte"
        elif termo in termos_tecnologia:
            tema = "tecnologia"
        elif termo in termos_saude:
            tema = "saúde"
        else:
            tema = "outros"

        sentimento = random.choice(["POS", "NEG", "NEU"])
        sentimento_score_map = {"POS": random.uniform(0.6, 0.99), "NEG": random.uniform(0.6, 0.99), "NEU": random.uniform(0.5, 0.8)}
        sent_score = sentimento_score_map[sentimento]

        sentimento_medio = round(random.uniform(-0.5, 0.5), 3)

        emergente = None
        if data >= base_datas[13]:
            if z_score >= 2.0:
                emergente = True
            else:
                emergente = False

        rows_topics.append({
            "data": data_str,
            "termo": termo,
            "frequencia": freq,
            "tfidf_score": tfidf,
            "z_score": z_score,
            "emergente": emergente,
            "tema": tema,
            "sentimento_medio_termo": sentimento_medio,
            "sentimento_predominante_termo": sentimento,
        })

        if tema not in sentimento_tema:
            sentimento_tema[tema] = {"scores": [], "labels": []}
        sentimento_tema[tema]["scores"].append(sentimento_medio)
        sentimento_tema[tema]["labels"].append(sentimento)

    for tema, vals in sentimento_tema.items():
        scores = vals["scores"]
        labels = vals["labels"]
        media = round(sum(scores) / len(scores), 3) if scores else 0
        total = len(labels)
        pos = labels.count("POS")
        neg = labels.count("NEG")
        neu = labels.count("NEU")
        rows_sentiment.append({
            "data": data_str,
            "tema": tema,
            "sentimento_medio": media,
            "pct_positivo": round(pos / total, 3) if total else 0,
            "pct_negativo": round(neg / total, 3) if total else 0,
            "pct_neutro": round(neu / total, 3) if total else 0,
            "qtd_noticias": total,
        })

df_topics = pd.DataFrame(rows_topics)
df_topics.to_parquet(os.path.join(GOLD_PATH, "daily_trending_topics.parquet"), index=False)
print(f"Gold 1: {len(df_topics)} linhas - {df_topics['data'].nunique()} dias, {df_topics['tema'].nunique()} temas")

df_sentiment = pd.DataFrame(rows_sentiment)
df_sentiment.to_parquet(os.path.join(GOLD_PATH, "daily_overall_sentiment.parquet"), index=False)
print(f"Gold 2: {len(df_sentiment)} linhas")

print("Dados mock gerados com sucesso!")
