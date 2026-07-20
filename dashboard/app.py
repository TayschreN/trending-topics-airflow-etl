import os
from collections import Counter
from datetime import datetime, timedelta
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gold")

TEMA_CORES = {
    "política": "#ef4444",
    "economia": "#f59e0b",
    "esporte": "#10b981",
    "saúde": "#3b82f6",
    "tecnologia": "#8b5cf6",
    "outros": "#6b7280",
}

app = Dash(__name__, title="Radar de Tópicos Emergentes")

app.layout = html.Div([
    html.Div([
        html.H1("Radar de Tópicos Emergentes"),
        html.Div("Análise de sentimento e detecção de tendências em notícias brasileiras", className="subtitle"),
    ], className="app-header"),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id="filtro-tema",
                placeholder="Filtrar por tema",
                multi=True,
            ),
        ], className="filter-item"),
        html.Div([
            dcc.DatePickerRange(
                id="filtro-periodo",
                display_format="DD/MM/YYYY",
                start_date_placeholder_text="Data inicial",
                end_date_placeholder_text="Data final",
            ),
        ], className="filter-item"),
        html.Div([
            dcc.Dropdown(
                id="filtro-fonte",
                placeholder="Filtrar por fonte",
                multi=True,
            ),
        ], className="filter-item"),
        html.Div([
            dcc.Dropdown(
                id="filtro-sentimento",
                placeholder="Sentimento",
                multi=True,
                options=[
                    {"label": "Positivo", "value": "POS"},
                    {"label": "Negativo", "value": "NEG"},
                    {"label": "Neutro", "value": "NEU"},
                ],
            ),
        ], className="filter-item"),
    ], className="controls-bar"),

    html.Div(id="kpi-grid", className="kpi-grid"),

    html.Div(id="insights-bar", className="insights-bar"),

    html.Div([
        html.Div(id="grafico-sentimento-tempo", className="chart-card"),
        html.Div(id="grafico-distribuicao-sentimento", className="chart-card"),
    ], className="chart-grid-2"),

    html.Div([
        html.Div(id="ranking-frequencia", className="chart-card"),
        html.Div(id="ranking-tfidf", className="chart-card"),
    ], className="chart-grid-2"),

    html.Div([
        html.Div(id="grafico-tendencia-emergentes", className="chart-card"),
        html.Div(id="grafico-distribuicao-temas", className="chart-card"),
        html.Div(id="grafico-fontes", className="chart-card"),
    ], className="chart-grid-3"),

    html.Div([
        html.Div(id="grafico-dispersao", className="chart-card"),
        html.Div(id="grafico-sentimento-geral", className="chart-card"),
    ], className="chart-grid-2"),

    html.Div(id="wordcloud-container", className="wordcloud-container"),

    html.Div([
        html.H3("Tópicos emergentes"),
        dash_table.DataTable(
            id="tabela-emergentes",
            page_size=10,
            style_table={"overflowX": "auto", "border": "none"},
            style_cell={
                "textAlign": "left",
                "padding": "10px 12px",
                "backgroundColor": "#1a1d2e",
                "color": "#c9d1d9",
                "fontFamily": "Inter, sans-serif",
                "fontSize": "13px",
                "border": "none",
                "borderBottom": "1px solid #2d2f4a",
            },
            style_header={
                "fontWeight": "600",
                "color": "#8b949e",
                "backgroundColor": "#1a1d2e",
                "borderBottom": "2px solid #2d2f4a",
                "fontSize": "12px",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{emergente} = true"},
                    "backgroundColor": "rgba(16, 185, 129, 0.1)",
                    "color": "#10b981",
                },
                {
                    "if": {"filter_query": "{sentimento_predominante_termo} = POS"},
                    "color": "#10b981",
                },
                {
                    "if": {"filter_query": "{sentimento_predominante_termo} = NEG"},
                    "color": "#ef4444",
                },
            ],
        ),
    ], className="emerging-table"),

    dcc.Interval(id="auto-refresh", interval=300000, n_intervals=0),
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


def _theme_color(tema):
    cores = {"política": "#ef4444", "economia": "#f59e0b", "esporte": "#10b981", "saúde": "#3b82f6", "tecnologia": "#8b5cf6"}
    return cores.get(tema, "#6b7280")


def _apply_filters(df1, df2, temas, fontes, sentimento, start, end):
    if df1.empty:
        return df1, df2
    if start:
        df1 = df1[df1["data"] >= start]
    if end:
        df1 = df1[df1["data"] <= end]
    if temas:
        df1 = df1[df1["tema"].isin(temas)]
    if fontes and "fontes" in df1.columns:
        mask = df1["fontes"].apply(lambda x: any(f in str(x).split(",") for f in fontes))
        df1 = df1[mask]
    if sentimento:
        df1 = df1[df1["sentimento_predominante_termo"].isin(sentimento)]

    if not df2.empty:
        if start:
            df2 = df2[df2["data"] >= start]
        if end:
            df2 = df2[df2["data"] <= end]
        if temas:
            df2 = df2[df2["tema"].isin(temas)]

    return df1, df2


def _make_empty_fig(msg="Sem dados no período"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color="#8b949e"),
    )
    fig.update_layout(
        paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#c9d1d9"),
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig





def _fig_layout(title="", height=320):
    return dict(
        title=dict(text=title, font=dict(size=14, color="#e1e4e8"), x=0, xanchor="left"),
        paper_bgcolor="#1a1d2e",
        plot_bgcolor="#1a1d2e",
        font=dict(color="#c9d1d9", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        hovermode="closest",
        legend=dict(font=dict(color="#c9d1d9"), orientation="h", y=1.1, x=0),
        xaxis=dict(gridcolor="#2d2f4a", zerolinecolor="#2d2f4a"),
        yaxis=dict(gridcolor="#2d2f4a", zerolinecolor="#2d2f4a"),
    )


@app.callback(
    [
        Output("filtro-tema", "options"),
        Output("filtro-periodo", "min_date_allowed"),
        Output("filtro-periodo", "max_date_allowed"),
        Output("filtro-periodo", "start_date"),
        Output("filtro-periodo", "end_date"),
        Output("filtro-fonte", "options"),
    ],
    Input("filtro-tema", "id"),
)
def _init_filters(_):
    df1 = _load_gold1()
    temas = sorted(df1["tema"].unique()) if not df1.empty else []
    dates = df1["data"].unique() if not df1.empty else []
    if len(dates) > 0:
        dates_sorted = sorted(dates)
        min_d, max_d = dates_sorted[0], dates_sorted[-1]
    else:
        min_d, max_d = None, None

    fontes = set()
    if not df1.empty and "fontes" in df1.columns:
        for f in df1["fontes"].dropna():
            for src in str(f).split(","):
                src = src.strip()
                if src:
                    fontes.add(src)
    fontes_opts = [{"label": f.capitalize(), "value": f} for f in sorted(fontes)]

    return (
        [{"label": t.capitalize(), "value": t} for t in temas],
        min_d, max_d, min_d, max_d,
        fontes_opts,
    )


@app.callback(
    Output("kpi-grid", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_kpis(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return [html.Div("Sem dados", className="kpi-card")]

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest] if latest else pd.DataFrame()

    if df_hoje.empty:
        return [html.Div("Sem dados", className="kpi-card")]

    total_noticias = len(df_hoje)
    sent_medio = df_hoje["sentimento_medio_termo"].mean()
    qtd_emergentes = int(df_hoje["emergente"].dropna().sum())
    total_termos = len(df_hoje)
    temas_ativos = df_hoje["tema"].nunique()

    sent_icon = "😊" if sent_medio > 0.1 else ("😡" if sent_medio < -0.1 else "😐")
    qtd_hoje = int(df_hoje["emergente"].dropna().sum())
    total_noticias_periodo = len(df1)
    total_termos_hoje = len(df_hoje)
    temas_ativos = df1["tema"].nunique()
    dias_com_dados = df1["data"].nunique()

    return [
        html.Div([
            html.Div("📰", className="kpi-icon"),
            html.Div(str(total_noticias_periodo), className="kpi-value"),
            html.Div("Notícias no período", className="kpi-label"),
        ], className="kpi-card"),
        html.Div([
            html.Div(sent_icon, className="kpi-icon"),
            html.Div(f"{sent_medio:.2f}" if not pd.isna(sent_medio) else "N/A", className="kpi-value"),
            html.Div("Sentimento médio hoje", className="kpi-label"),
        ], className="kpi-card"),
        html.Div([
            html.Div("🔥", className="kpi-icon"),
            html.Div(str(qtd_hoje), className="kpi-value"),
            html.Div("Tópicos emergentes", className="kpi-label"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📊", className="kpi-icon"),
            html.Div(str(temas_ativos), className="kpi-value"),
            html.Div("Temas ativos", className="kpi-label"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📝", className="kpi-icon"),
            html.Div(str(total_termos_hoje), className="kpi-value"),
            html.Div("Termos hoje", className="kpi-label"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📅", className="kpi-icon"),
            html.Div(str(dias_com_dados), className="kpi-value"),
            html.Div("Dias com dados", className="kpi-label"),
        ], className="kpi-card"),
    ]


@app.callback(
    Output("insights-bar", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_insights(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return []

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest]

    insights = []

    top_termo = df_hoje.loc[df_hoje["frequencia"].idxmax(), "termo"] if not df_hoje.empty else "N/A"
    insights.append(html.Div([
        html.Strong("🔥 Tópico mais frequente: "), top_termo
    ], className="insight-card"))

    if not df_hoje.empty:
        pos_pct = (df_hoje["sentimento_predominante_termo"] == "POS").mean() * 100
        neg_pct = (df_hoje["sentimento_predominante_termo"] == "NEG").mean() * 100
        insights.append(html.Div([
            html.Strong("😊 Sentimento: "),
            f"{pos_pct:.0f}% positivo, {neg_pct:.0f}% negativo"
        ], className="insight-card"))

    if not df2.empty:
        latest_sent = df2[df2["data"] == df2["data"].max()]
        if not latest_sent.empty:
            best_tema = latest_sent.loc[latest_sent["sentimento_medio"].idxmax(), "tema"]
            worst_tema = latest_sent.loc[latest_sent["sentimento_medio"].idxmin(), "tema"]
            insights.append(html.Div([
                html.Strong("😊 Tema mais positivo: "), best_tema,
                html.Br(),
                html.Strong("😡 Tema mais negativo: "), worst_tema,
            ], className="insight-card"))

    if not df1.empty:
        total_emerg = df1["emergente"].dropna().sum()
        pct_emerg = (total_emerg / len(df1)) * 100
        insights.append(html.Div([
            html.Strong("🔥 Tópicos emergentes: "),
            f"{int(total_emerg)} ({pct_emerg:.1f}% do total)"
        ], className="insight-card"))

    if not df1.empty and "fontes" in df1.columns:
        todas_fontes = set()
        for f in df1["fontes"].dropna():
            for src in str(f).split(","):
                src = src.strip()
                if src:
                    todas_fontes.add(src)
        insights.append(html.Div([
            html.Strong("📰 Fontes ativas: "),
            ", ".join(sorted(todas_fontes))
        ], className="insight-card"))

    return insights


@app.callback(
    Output("grafico-sentimento-tempo", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_sentiment_chart(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df2.empty:
        return dcc.Graph(figure=_make_empty_fig())

    fig = px.line(
        df2, x="data", y="sentimento_medio", color="tema",
        title="Sentimento médio por tema ao longo do tempo",
        markers=True, color_discrete_map=TEMA_CORES,
    )
    fig.update_layout(**_fig_layout("Sentimento médio por tema ao longo do tempo"))
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig.add_hline(y=0, line_dash="dash", line_color="#4a4d6a", opacity=0.5)
    return dcc.Graph(figure=fig)


@app.callback(
    Output("grafico-distribuicao-sentimento", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_sentiment_dist(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest]

    if df_hoje.empty:
        return dcc.Graph(figure=_make_empty_fig())

    counts = df_hoje["sentimento_predominante_termo"].value_counts()
    labels = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutro"}
    colors_map = {"POS": "#10b981", "NEG": "#ef4444", "NEU": "#6b7280"}

    fig = go.Figure(data=[go.Pie(
        labels=[labels.get(k, k) for k in counts.index],
        values=counts.values,
        marker=dict(colors=[colors_map.get(k, "#6b7280") for k in counts.index]),
        textinfo="label+percent",
        textfont=dict(color="#ffffff", size=12),
        hole=0.5,
        hovertemplate="%{label}: %{value} termos (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        **_fig_layout("Distribuição de sentimento"),
        showlegend=False,
    )
    return dcc.Graph(figure=fig)


@app.callback(
    Output("ranking-frequencia", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_rank_freq(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest].nlargest(10, "frequencia")

    fig = px.bar(
        df_hoje, x="frequencia", y="termo", orientation="h",
        title="Top 10 por frequência",
        color="tema", color_discrete_map=TEMA_CORES,
        text="frequencia",
    )
    fig.update_layout(**_fig_layout("Top 10 termos por frequência"))
    fig.update_traces(textposition="outside", textfont=dict(color="#c9d1d9"))
    fig.update_yaxes(autorange="reversed")
    return dcc.Graph(figure=fig)


@app.callback(
    Output("ranking-tfidf", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_rank_tfidf(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest].nlargest(10, "tfidf_score")

    fig = px.bar(
        df_hoje, x="tfidf_score", y="termo", orientation="h",
        title="Top 10 por TF-IDF",
        color="tema", color_discrete_map=TEMA_CORES,
        text="tfidf_score",
    )
    fig.update_layout(**_fig_layout("Top 10 termos por TF-IDF"))
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", textfont=dict(color="#c9d1d9"))
    fig.update_yaxes(autorange="reversed")
    return dcc.Graph(figure=fig)


@app.callback(
    Output("grafico-tendencia-emergentes", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_emerging_trend(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    trend = df1.groupby("data").agg(
        total_termos=("termo", "count"),
        emergentes=("emergente", lambda x: x.dropna().sum()),
    ).reset_index().sort_values("data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["data"], y=trend["emergentes"],
        mode="lines+markers", name="Emergentes",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=6, color="#10b981"),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)",
    ))
    fig.update_layout(**_fig_layout("Evolução de tópicos emergentes"))
    return dcc.Graph(figure=fig)


@app.callback(
    Output("grafico-distribuicao-temas", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_theme_dist(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest]
    dist = df_hoje["tema"].value_counts().reset_index()
    dist.columns = ["tema", "contagem"]

    fig = px.bar(
        dist, x="tema", y="contagem", color="tema",
        color_discrete_map=TEMA_CORES,
        text="contagem",
    )
    fig.update_layout(
        **_fig_layout("Distribuição por tema"),
        showlegend=False,
    )
    fig.update_xaxes(title="", tickangle=-30)
    fig.update_yaxes(title="Quantidade de termos")
    fig.update_traces(textposition="outside", textfont=dict(color="#c9d1d9"))
    return dcc.Graph(figure=fig)


@app.callback(
    Output("grafico-fontes", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_source_chart(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest]

    fonte_counts = Counter()
    for f in df_hoje["fontes"].dropna():
        for src in str(f).split(","):
            src = src.strip()
            if src:
                fonte_counts[src] += 1

    if not fonte_counts:
        return dcc.Graph(figure=_make_empty_fig())

    df_fontes = pd.DataFrame(fonte_counts.most_common(), columns=["fonte", "contagem"])

    fig = px.bar(
        df_fontes, x="fonte", y="contagem", color="fonte",
        text="contagem", color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
    )
    fig.update_layout(
        **_fig_layout("Distribuição por fonte"),
        showlegend=False,
    )
    fig.update_xaxes(title="", tickangle=-20)
    fig.update_yaxes(title="Termos")
    fig.update_traces(textposition="outside", textfont=dict(color="#c9d1d9"))
    return dcc.Graph(figure=fig)


@app.callback(
    Output("grafico-dispersao", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_scatter(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest].copy()
    df_hoje["tamanho"] = df_hoje["frequencia"] * 3

    fig = px.scatter(
        df_hoje, x="frequencia", y="z_score", color="tema",
        size="tamanho", hover_name="termo",
        color_discrete_map=TEMA_CORES,
        title="Frequência vs Z-Score",
        labels={"frequencia": "Frequência", "z_score": "Z-Score"},
    )
    fig.update_layout(**_fig_layout("Frequência vs Z-Score (termos)"))
    fig.add_hline(y=2, line_dash="dash", line_color="#10b981", opacity=0.5,
                  annotation_text="Emergente (z=2)", annotation_font=dict(color="#10b981", size=10))
    return dcc.Graph(figure=fig)


@app.callback(
    Output("grafico-sentimento-geral", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_sentiment_overall(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df2.empty:
        return dcc.Graph(figure=_make_empty_fig())

    latest = df2["data"].max()
    df_hoje = df2[df2["data"] == latest]

    fig = go.Figure()
    for _, row in df_hoje.iterrows():
        tema = row["tema"]
        fig.add_trace(go.Bar(
            name=tema,
            x=["Positivo", "Negativo", "Neutro"],
            y=[row["pct_positivo"] * 100, row["pct_negativo"] * 100, row["pct_neutro"] * 100],
            marker_color=_theme_color(tema),
            text=[f"{row['pct_positivo']*100:.0f}%", f"{row['pct_negativo']*100:.0f}%", f"{row['pct_neutro']*100:.0f}%"],
            textposition="inside",
            textfont=dict(color="white", size=10),
        ))
    fig.update_layout(
        **_fig_layout("Distribuição de sentimento por tema"),
        barmode="group",
        showlegend=True,
    )
    fig.update_yaxes(title="Percentual", tickformat=".0%")
    return dcc.Graph(figure=fig)


@app.callback(
    Output("wordcloud-container", "children"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_wordcloud(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return html.Div("Sem dados para wordcloud", style={"color": "#8b949e", "textAlign": "center"})

    latest = df1["data"].max()
    df_hoje = df1[df1["data"] == latest]

    if df_hoje.empty:
        return html.Div("Sem dados para wordcloud", style={"color": "#8b949e", "textAlign": "center"})

    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    import base64

    freq_dict = dict(zip(df_hoje["termo"], df_hoje["frequencia"]))
    wc = WordCloud(
        width=800, height=300, background_color="#1a1d2e",
        max_words=80, colormap="viridis",
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(freq_dict)

    img = io.BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode()

    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={"maxWidth": "100%", "height": "auto", "borderRadius": "8px"},
    )


@app.callback(
    Output("tabela-emergentes", "data"),
    Output("tabela-emergentes", "columns"),
    [Input("filtro-tema", "value"), Input("filtro-fonte", "value"),
     Input("filtro-sentimento", "value"), Input("filtro-periodo", "start_date"),
     Input("filtro-periodo", "end_date")],
)
def _update_emerging_table(temas, fontes, sentimento, start, end):
    df1, df2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sentimento, start, end)
    if df1.empty:
        return [], []

    latest = df1["data"].max()
    df_emerg = df1[(df1["data"] == latest) & (df1["emergente"] == True)]

    if df_emerg.empty:
        return [], []

    cols = [
        {"name": "Termo", "id": "termo"},
        {"name": "Frequência", "id": "frequencia"},
        {"name": "TF-IDF", "id": "tfidf_score"},
        {"name": "Z-Score", "id": "z_score"},
        {"name": "Tema", "id": "tema"},
        {"name": "Sentimento", "id": "sentimento_predominante_termo"},
        {"name": "Fontes", "id": "fontes"},
    ]
    data = df_emerg.to_dict("records")
    for row in data:
        row["tfidf_score"] = round(row.get("tfidf_score", 0), 4)
        row["z_score"] = round(row.get("z_score", 0), 2)
    return data, cols


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
