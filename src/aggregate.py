import pandas as pd
import logging
import os
from collections import Counter

logger = logging.getLogger(__name__)

TEMA_KEYWORDS = {
    "política": [
        "governo", "presidente", "eleição", "voto", "congresso",
        "senado", "câmara", "ministro", "partido", "político",
    ],
    "economia": [
        "economia", "inflação", "pib", "dólar", "bolsa",
        "mercado", "juros", "imposto", "orçamento", "bancos",
    ],
    "esporte": [
        "futebol", "olímpiadas", "campeonato", "jogador",
        "time", "título", "medalha", "copa", "esporte",
    ],
    "saúde": [
        "saúde", "hospital", "vacina", "doença", "tratamento",
        "médico", "covid", "sus", "medicamento", "cirurgia",
    ],
    "tecnologia": [
        "tecnologia", "internet", "aplicativo", "software",
        "inteligência artificial", "dados", "digital", "startup",
        "inovação", "computador",
    ],
}


def _categorizar_termo(termo: str) -> str:
    for tema, keywords in TEMA_KEYWORDS.items():
        if any(kw in termo.lower() for kw in keywords):
            return tema
    return "outros"


def _sentimento_pontuado(label: str, score: float) -> float:
    if label == "POS":
        return score
    elif label == "NEG":
        return -score
    return 0.0


def aggregate_daily_results(
    df_noticias: pd.DataFrame,
    df_freq: pd.DataFrame,
    df_tfidf: pd.DataFrame,
    df_emergencia: pd.DataFrame,
    data_execucao: str,
    gold_path: str,
) -> dict:
    df_merged = df_emergencia.merge(
        df_tfidf[["termo", "tfidf_score"]], on="termo", how="left"
    )
    df_merged["tfidf_score"] = df_merged["tfidf_score"].fillna(0.0)
    df_merged["tema"] = df_merged["termo"].apply(_categorizar_termo)

    sentimento_por_termo = {}
    fontes_por_termo = {}
    for _, row in df_noticias.iterrows():
        texto = str(row.get("texto_completo", "")).lower()
        label = row.get("sentimento_label", "NEU")
        score = row.get("sentimento_score", 0.0)
        fonte = row.get("fonte", "desconhecida")
        for termo in df_merged["termo"]:
            if termo in texto:
                if termo not in sentimento_por_termo:
                    sentimento_por_termo[termo] = {"scores": [], "labels": []}
                    fontes_por_termo[termo] = []
                sentimento_por_termo[termo]["scores"].append(
                    _sentimento_pontuado(label, score)
                )
                sentimento_por_termo[termo]["labels"].append(label)
                fontes_por_termo[termo].append(fonte)

    df_merged["sentimento_medio_termo"] = 0.0
    df_merged["sentimento_predominante_termo"] = "NEU"
    df_merged["fontes"] = ""
    for i, termo in enumerate(df_merged["termo"]):
        if termo in sentimento_por_termo:
            data = sentimento_por_termo[termo]
            if data["scores"]:
                df_merged.at[i, "sentimento_medio_termo"] = sum(data["scores"]) / len(
                    data["scores"]
                )
            if data["labels"]:
                df_merged.at[i, "sentimento_predominante_termo"] = max(
                    set(data["labels"]), key=data["labels"].count
                )
        if termo in fontes_por_termo:
            df_merged.at[i, "fontes"] = ",".join(sorted(set(fontes_por_termo[termo])))

    df_merged["data"] = data_execucao
    df_gold_1 = df_merged[
        [
            "data",
            "termo",
            "frequencia",
            "tfidf_score",
            "z_score",
            "emergente",
            "tema",
            "sentimento_medio_termo",
            "sentimento_predominante_termo",
            "fontes",
        ]
    ]

    gold_file_1 = os.path.join(gold_path, "daily_trending_topics.parquet")
    if os.path.exists(gold_file_1):
        df_existing = pd.read_parquet(gold_file_1)
        df_existing = df_existing[df_existing["data"] != data_execucao]
        df_gold_1 = pd.concat([df_existing, df_gold_1], ignore_index=True)
    df_gold_1.to_parquet(gold_file_1, index=False)
    logger.info(f"Gold 1 salvo: {gold_file_1} — {len(df_gold_1)} linhas")

    df_noticias["tema_noticia"] = df_noticias.get("texto_completo", "").apply(
        _categorizar_termo
    )
    overall = (
        df_noticias.groupby("tema_noticia")
        .agg(
            sentimento_medio=("sentimento_score", "mean"),
            qtd_noticias=("sentimento_score", "count"),
        )
        .reset_index()
    )
    overall["data"] = data_execucao
    overall["tema"] = overall["tema_noticia"]

    def _compute_pcts(group):
        total = len(group)
        pos = (group["sentimento_label"] == "POS").sum()
        neg = (group["sentimento_label"] == "NEG").sum()
        neu = (group["sentimento_label"] == "NEU").sum()
        return pd.Series(
            {
                "pct_positivo": pos / total if total else 0,
                "pct_negativo": neg / total if total else 0,
                "pct_neutro": neu / total if total else 0,
            }
        )

    pcts = df_noticias.groupby("tema_noticia").apply(_compute_pcts).reset_index()
    overall = overall.merge(
        pcts, left_on="tema_noticia", right_on="tema_noticia", how="left"
    )

    gold_file_2 = os.path.join(gold_path, "daily_overall_sentiment.parquet")
    if os.path.exists(gold_file_2):
        df_existing2 = pd.read_parquet(gold_file_2)
        df_existing2 = df_existing2[df_existing2["data"] != data_execucao]
        overall = pd.concat([df_existing2, overall], ignore_index=True)
    overall.to_parquet(gold_file_2, index=False)
    logger.info(f"Gold 2 salvo: {gold_file_2} — {len(overall)} linhas")

    return {
        "gold1_path": gold_file_1,
        "gold2_path": gold_file_2,
        "total_noticias": len(df_noticias),
        "total_termos": len(df_merged),
    }
