import os
import json
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import pandas as pd

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gold")

app = Dash(__name__, title="Radar de Tópicos Emergentes")

app.layout = html.Div([
    html.H1("Radar de Tópicos Emergentes e Sentimento"),

    html.Div([
        dcc.Dropdown(id="filtro-tema", placeholder="Filtrar por tema", multi=True),
        dcc.DatePickerRange(id="filtro-periodo"),
    ], style={"display": "flex", "gap": "16px", "margin": "16px 0"}),

    html.Div(id="kpis-do-dia", style={"display": "flex", "gap": "32px", "margin": "16px 0"}),

    dcc.Graph(id="grafico-sentimento-tempo"),

    html.Div([
        dcc.Graph(id="ranking-frequencia", style={"width": "50%"}),
        dcc.Graph(id="ranking-tfidf", style={"width": "50%"}),
    ], style={"display": "flex"}),

    html.Img(id="wordcloud-do-dia", style={"width": "100%", "maxWidth": "800px", "margin": "16px 0"}),

    html.H3("Tópicos emergentes hoje"),
    dash_table.DataTable(
        id="tabela-emergentes",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold"},
    ),
])


def _load_gold1():
    path = os.path.join(GOLD_PATH, "daily_trending_topics.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_gold2():
    path = os.path.join(GOLD_PATH, "daily_overall_sentiment.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


@app.callback(
    [
        Output("filtro-tema", "options"),
        Output("filtro-periodo", "min_date_allowed"),
        Output("filtro-periodo", "max_date_allowed"),
        Output("filtro-periodo", "start_date"),
        Output("filtro-periodo", "end_date"),
    ],
    Input("filtro-tema", "id"),
)
def _init_filters(_):
    df1 = _load_gold1()
    temas = sorted(df1["tema"].unique()) if not df1.empty else []
    dates = df1["data"].unique() if not df1.empty else []
    if len(dates) > 0:
        min_d, max_d = min(dates), max(dates)
    else:
        min_d, max_d = None, None
    return (
        [{"label": t, "value": t} for t in temas],
        min_d,
        max_d,
        min_d,
        max_d,
    )


@app.callback(
    Output("kpis-do-dia", "children"),
    [Input("filtro-tema", "value"), Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")],
)
def _update_kpis(temas, start, end):
    df1 = _load_gold1()
    if df1.empty:
        return []

    if start:
        df1 = df1[df1["data"] >= start]
    if end:
        df1 = df1[df1["data"] <= end]
    if temas:
        df1 = df1[df1["tema"].isin(temas)]

    latest = df1["data"].max() if not df1.empty else None
    df_hoje = df1[df1["data"] == latest] if latest else pd.DataFrame()

    if df_hoje.empty:
        return [html.Div("Sem dados")]

    total_noticias = len(df_hoje)
    sent_medio = df_hoje["sentimento_medio_termo"].mean()
    qtd_emergentes = df_hoje["emergente"].dropna().sum()

    return [
        html.Div(f"Notícias: {total_noticias}"),
        html.Div(f"Sentimento médio: {sent_medio:.2f}"),
        html.Div(f"Emergentes: {int(qtd_emergentes)}"),
    ]


@app.callback(
    Output("grafico-sentimento-tempo", "figure"),
    [Input("filtro-tema", "value"), Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")],
)
def _update_sentiment_chart(temas, start, end):
    df2 = _load_gold2()
    if df2.empty:
        return px.line(title="Sem dados")

    if start:
        df2 = df2[df2["data"] >= start]
    if end:
        df2 = df2[df2["data"] <= end]
    if temas:
        df2 = df2[df2["tema"].isin(temas)]

    fig = px.line(
        df2,
        x="data",
        y="sentimento_medio",
        color="tema",
        title="Sentimento médio por tema ao longo do tempo",
        markers=True,
    )
    return fig


@app.callback(
    Output("ranking-frequencia", "figure"),
    [Input("filtro-tema", "value"), Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")],
)
def _update_rank_freq(temas, start, end):
    df1 = _load_gold1()
    if df1.empty:
        return px.bar(title="Sem dados")

    if start:
        df1 = df1[df1["data"] >= start]
    if end:
        df1 = df1[df1["data"] <= end]
    if temas:
        df1 = df1[df1["tema"].isin(temas)]

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest].nlargest(10, "frequencia")

    fig = px.bar(
        df_hoje,
        x="frequencia",
        y="termo",
        orientation="h",
        title="Top 10 por frequência",
        color="tema",
    )
    return fig


@app.callback(
    Output("ranking-tfidf", "figure"),
    [Input("filtro-tema", "value"), Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")],
)
def _update_rank_tfidf(temas, start, end):
    df1 = _load_gold1()
    if df1.empty:
        return px.bar(title="Sem dados")

    if start:
        df1 = df1[df1["data"] >= start]
    if end:
        df1 = df1[df1["data"] <= end]
    if temas:
        df1 = df1[df1["tema"].isin(temas)]

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest].nlargest(10, "tfidf_score")

    fig = px.bar(
        df_hoje,
        x="tfidf_score",
        y="termo",
        orientation="h",
        title="Top 10 por TF-IDF",
        color="tema",
    )
    return fig


@app.callback(
    Output("tabela-emergentes", "data"),
    Output("tabela-emergentes", "columns"),
    [Input("filtro-tema", "value"), Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")],
)
def _update_emerging_table(temas, start, end):
    df1 = _load_gold1()
    if df1.empty:
        return [], []

    if start:
        df1 = df1[df1["data"] >= start]
    if end:
        df1 = df1[df1["data"] <= end]
    if temas:
        df1 = df1[df1["tema"].isin(temas)]

    latest = df1["data"].max()
    df_emerg = df1[(df1["data"] == latest) & (df1["emergente"] == True)]

    if df_emerg.empty:
        return [], []

    cols = [
        {"name": "Termo", "id": "termo"},
        {"name": "Frequência", "id": "frequencia"},
        {"name": "Z-Score", "id": "z_score"},
        {"name": "Tema", "id": "tema"},
        {"name": "Sentimento", "id": "sentimento_predominante_termo"},
    ]
    data = df_emerg.to_dict("records")
    return data, cols


if __name__ == "__main__":
    app.run(debug=True)
