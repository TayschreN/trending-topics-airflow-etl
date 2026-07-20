import os
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gold")

TEMA_CORES = {
    "política": "#ef4444", "economia": "#f59e0b", "esporte": "#10b981",
    "saúde": "#3b82f6", "tecnologia": "#8b5cf6", "outros": "#6b7280",
}

app = Dash(__name__, title="Radar de Tópicos Emergentes")

app.layout = html.Div([
    html.Div([
        html.H1("Radar de Tópicos Emergentes"),
        html.Div("Análise de sentimento e detecção de tendências em notícias brasileiras", className="subtitle"),
    ], className="app-header"),

    html.Div([
        html.Div([dcc.Dropdown(id="filtro-tema", placeholder="Filtrar por tema", multi=True)], className="filter-item"),
        html.Div([dcc.DatePickerRange(id="filtro-periodo", display_format="DD/MM/YYYY",
            start_date_placeholder_text="Data inicial", end_date_placeholder_text="Data final")], className="filter-item"),
        html.Div([dcc.Dropdown(id="filtro-fonte", placeholder="Filtrar por fonte", multi=True)], className="filter-item"),
        html.Div([dcc.Dropdown(id="filtro-sentimento", placeholder="Sentimento", multi=True,
            options=[{"label": "Positivo", "value": "POS"}, {"label": "Negativo", "value": "NEG"}, {"label": "Neutro", "value": "NEU"}])], className="filter-item"),
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
        html.H3("Tópicos do dia"),
        html.Div(id="resumo-termos", className="insights-bar"),
        dash_table.DataTable(
            id="tabela-termos", page_size=10,
            style_table={"overflowX": "auto", "border": "none"},
            style_cell={"textAlign": "left", "padding": "10px 12px", "backgroundColor": "#1a1d2e",
                "color": "#c9d1d9", "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                "border": "none", "borderBottom": "1px solid #2d2f4a"},
            style_header={"fontWeight": "600", "color": "#8b949e", "backgroundColor": "#1a1d2e",
                "borderBottom": "2px solid #2d2f4a", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.5px"},
            style_data_conditional=[
                {"if": {"filter_query": "{emergente} = true"}, "backgroundColor": "rgba(16, 185, 129, 0.1)", "color": "#10b981"},
                {"if": {"filter_query": "{sentimento_predominante_termo} = POS"}, "color": "#10b981"},
                {"if": {"filter_query": "{sentimento_predominante_termo} = NEG"}, "color": "#ef4444"},
            ],
        ),
    ], className="emerging-table"),
])


def _load_gold1():
    p = os.path.join(GOLD_PATH, "daily_trending_topics.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

def _load_gold2():
    p = os.path.join(GOLD_PATH, "daily_overall_sentiment.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

def _apply_filters(df1, df2, temas, fontes, sentimento, start, end):
    if not df1.empty:
        if start: df1 = df1[df1["data"] >= start]
        if end: df1 = df1[df1["data"] <= end]
        if temas: df1 = df1[df1["tema"].isin(temas)]
        if fontes and "fontes" in df1.columns:
            m = df1["fontes"].apply(lambda x: any(f in str(x).split(",") for f in fontes))
            df1 = df1[m]
        if sentimento: df1 = df1[df1["sentimento_predominante_termo"].isin(sentimento)]
    if not df2.empty:
        if start: df2 = df2[df2["data"] >= start]
        if end: df2 = df2[df2["data"] <= end]
        if temas: df2 = df2[df2["tema"].isin(temas)]
    return df1, df2

def _empty_fig(msg="Sem dados"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#8b949e"))
    fig.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e", font=dict(color="#c9d1d9"),
        margin=dict(l=20, r=20, t=40, b=20), height=300, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def _fig(title="", h=320):
    return dict(title=dict(text=title, font=dict(size=14, color="#e1e4e8"), x=0, xanchor="left"),
        paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e", font=dict(color="#c9d1d9", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20), height=h, hovermode="closest",
        legend=dict(font=dict(color="#c9d1d9"), orientation="h", y=1.1, x=0),
        xaxis=dict(gridcolor="#2d2f4a", zerolinecolor="#2d2f4a"),
        yaxis=dict(gridcolor="#2d2f4a", zerolinecolor="#2d2f4a"))


@app.callback(
    [Output("filtro-tema", "options"), Output("filtro-periodo", "min_date_allowed"),
     Output("filtro-periodo", "max_date_allowed"), Output("filtro-periodo", "start_date"),
     Output("filtro-periodo", "end_date"), Output("filtro-fonte", "options")],
    Input("filtro-tema", "id"),
)
def _init(_):
    d1 = _load_gold1()
    temas = sorted(d1["tema"].unique()) if not d1.empty else []
    dates = d1["data"].unique() if not d1.empty else []
    md, xd = (None, None) if not dates else (min(dates), max(dates))
    fontes = set()
    if not d1.empty and "fontes" in d1.columns:
        for f in d1["fontes"].dropna():
            for s in str(f).split(","):
                if s.strip(): fontes.add(s.strip())
    return ([{"label": t.capitalize(), "value": t} for t in temas], md, xd, md, xd,
            [{"label": f.capitalize(), "value": f} for f in sorted(fontes)])


@app.callback(Output("kpi-grid", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _kpi(temas, fontes, sent, start, end):
    d1, d2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return [html.Div("Sem dados", className="kpi-card")]
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    if hoje.empty: return [html.Div("Sem dados", className="kpi-card")]
    sm = hoje["sentimento_medio_termo"].mean()
    icon = "😊" if (not pd.isna(sm) and sm > 0.1) else ("😡" if (not pd.isna(sm) and sm < -0.1) else "😐")
    n_emerg = int(hoje["emergente"].dropna().sum())
    return [
        html.Div([html.Div("📰", className="kpi-icon"), html.Div(str(len(d1)), className="kpi-value"), html.Div("Notícias no período", className="kpi-label")], className="kpi-card"),
        html.Div([html.Div(icon, className="kpi-icon"), html.Div(f"{sm:.2f}" if not pd.isna(sm) else "N/A", className="kpi-value"), html.Div("Sentimento médio hoje", className="kpi-label")], className="kpi-card"),
        html.Div([html.Div("🔥", className="kpi-icon"), html.Div(str(n_emerg), className="kpi-value"), html.Div("Tópicos emergentes", className="kpi-label")], className="kpi-card"),
        html.Div([html.Div("📊", className="kpi-icon"), html.Div(str(d1["tema"].nunique()), className="kpi-value"), html.Div("Temas ativos", className="kpi-label")], className="kpi-card"),
        html.Div([html.Div("📝", className="kpi-icon"), html.Div(str(len(hoje)), className="kpi-value"), html.Div("Termos hoje", className="kpi-label")], className="kpi-card"),
        html.Div([html.Div("📅", className="kpi-icon"), html.Div(str(d1["data"].nunique()), className="kpi-value"), html.Div("Dias com dados", className="kpi-label")], className="kpi-card"),
    ]


@app.callback(Output("insights-bar", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _insights(temas, fontes, sent, start, end):
    d1, d2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return []
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    ins = []
    if not hoje.empty and "fontes" in hoje.columns:
        fs = set()
        for f in hoje["fontes"].dropna():
            for s in str(f).split(","):
                if s.strip(): fs.add(s.strip())
        ins.append(html.Div([html.Strong("📰 Fontes ativas: "), ", ".join(sorted(fs))], className="insight-card"))

    if not hoje.empty:
        ppos = int((hoje["sentimento_predominante_termo"] == "POS").sum())
        pneg = int((hoje["sentimento_predominante_termo"] == "NEG").sum())
        pneu = int((hoje["sentimento_predominante_termo"] == "NEU").sum())
        tot = len(hoje)
        ins.append(html.Div([
            html.Strong("😊 Sentimento hoje: "),
            f"POS {ppos} ({ppos/tot*100:.0f}%) | NEG {pneg} ({pneg/tot*100:.0f}%) | NEU {pneu} ({pneu/tot*100:.0f}%)"
        ], className="insight-card"))

    if not hoje.empty:
        by_tema = hoje.groupby("tema")["frequencia"].sum().sort_values(ascending=False)
        top_tema = by_tema.index[0]
        top_pct = by_tema.iloc[0] / by_tema.sum() * 100
        ins.append(html.Div([
            html.Strong("📊 Tema dominante: "), f"{top_tema.capitalize()} ({top_pct:.0f}% dos termos)"
        ], className="insight-card"))

    if not d2.empty:
        ult2 = d2[d2["data"] == d2["data"].max()]
        if not ult2.empty:
            best = ult2.loc[ult2["sentimento_medio"].idxmax(), "tema"]
            worst = ult2.loc[ult2["sentimento_medio"].idxmin(), "tema"]
            ins.append(html.Div([
                html.Strong("😊 Mais positivo: "), best.capitalize(),
                html.Span("  |  "),
                html.Strong("😡 Mais negativo: "), worst.capitalize()
            ], className="insight-card"))

    if not hoje.empty:
        n_emerg = int(hoje["emergente"].dropna().sum())
        n_total = len(hoje)
        if n_emerg > 0:
            emerg_terms = hoje[hoje["emergente"] == True].nlargest(3, "z_score")["termo"].tolist()
            ins.append(html.Div([
                html.Strong("🔥 Emergentes: "), f"{n_emerg} de {n_total} termos",
                html.Span("  |  "), html.Strong("Top: "), ", ".join(emerg_terms[:3])
            ], className="insight-card"))

    if len(d1["data"].unique()) >= 2:
        dias = sorted(d1["data"].unique())
        hoje_t = d1[d1["data"] == dias[-1]]
        ontem_t = d1[d1["data"] == dias[-2]]
        if not hoje_t.empty and not ontem_t.empty:
            hoje_cnt = len(hoje_t)
            ontem_cnt = len(ontem_t)
            diff = hoje_cnt - ontem_cnt
            arrow = "📈" if diff > 0 else ("📉" if diff < 0 else "➡️")
            ins.append(html.Div([
                html.Strong(f"{arrow} Volume vs ontem: "),
                f"{hoje_cnt} termos (vs {ontem_cnt}, {'+' if diff > 0 else ''}{diff})"
            ], className="insight-card"))

    return ins


@app.callback(Output("grafico-sentimento-tempo", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _sent_tempo(temas, fontes, sent, start, end):
    _, d2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d2.empty: return dcc.Graph(figure=_empty_fig())
    fig = px.line(d2, x="data", y="sentimento_medio", color="tema", markers=True, color_discrete_map=TEMA_CORES)
    fig.update_layout(**_fig("Sentimento médio por tema ao longo do tempo"), hovermode="x unified")
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig.add_hline(y=0, line_dash="dash", line_color="#4a4d6a", opacity=0.5)
    return dcc.Graph(figure=fig)


@app.callback(Output("grafico-distribuicao-sentimento", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _sent_dist(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    if hoje.empty: return dcc.Graph(figure=_empty_fig())
    rot = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutro"}
    cm = {"POS": "#10b981", "NEG": "#ef4444", "NEU": "#6b7280"}
    cnt = hoje["sentimento_predominante_termo"].value_counts()
    npos = int(cnt.get("POS", 0)); nneg = int(cnt.get("NEG", 0)); nneu = int(cnt.get("NEU", 0))
    fig = go.Figure(data=[go.Pie(
        labels=[rot.get(k, k) for k in cnt.index],
        values=cnt.values, marker=dict(colors=[cm.get(k, "#6b7280") for k in cnt.index]),
        textinfo="label+percent", textfont=dict(color="#ffffff", size=12), hole=0.5,
        hovertemplate="%{label}: %{value} termos (%{percent})<extra></extra>",
    )])
    fig.add_annotation(x=0.5, y=0.55, text=f"{npos+nneg+nneu}", showarrow=False, font=dict(size=24, color="#ffffff"))
    fig.add_annotation(x=0.5, y=0.43, text="termos", showarrow=False, font=dict(size=11, color="#8b949e"))
    fig.update_layout(**_fig("Distribuição de sentimento"), showlegend=False, margin=dict(l=20, r=20, t=40, b=80))
    return dcc.Graph(figure=fig)


@app.callback(Output("ranking-frequencia", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _rank_freq(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult].nlargest(10, "frequencia")
    if hoje.empty: return dcc.Graph(figure=_empty_fig())
    hoje["label"] = hoje["termo"] + " [" + hoje["tema"].str.capitalize() + "]"
    fig = px.bar(hoje, x="frequencia", y="label", orientation="h", color="tema",
        color_discrete_map=TEMA_CORES, text="frequencia", hover_data={"termo": True, "tema": True, "frequencia": True})
    fig.update_layout(**_fig("Top 10 termos por frequência"), yaxis=dict(title=""))
    fig.update_traces(textposition="outside", textfont=dict(color="#c9d1d9"))
    fig.update_yaxes(autorange="reversed")
    return dcc.Graph(figure=fig)


@app.callback(Output("ranking-tfidf", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _rank_tfidf(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult].nlargest(10, "tfidf_score")
    if hoje.empty: return dcc.Graph(figure=_empty_fig())
    hoje["label"] = hoje["termo"] + " [" + hoje["tema"].str.capitalize() + "]"
    fig = px.bar(hoje, x="tfidf_score", y="label", orientation="h", color="tema",
        color_discrete_map=TEMA_CORES, text="tfidf_score", hover_data={"termo": True, "tema": True, "tfidf_score": True})
    fig.update_layout(**_fig("Top 10 por TF-IDF (termos distintivos do dia)"), yaxis=dict(title=""))
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", textfont=dict(color="#c9d1d9"))
    fig.update_yaxes(autorange="reversed")
    return dcc.Graph(figure=fig)


@app.callback(Output("grafico-tendencia-emergentes", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _emerg_trend(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    tr = d1.groupby("data").agg(termos=("termo", "count"), emerg=("emergente", lambda x: x.dropna().sum())).reset_index().sort_values("data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr["data"], y=tr["termos"], mode="lines+markers", name="Total termos",
        line=dict(color="#6b7280", width=2), marker=dict(size=5, color="#6b7280")))
    fig.add_trace(go.Scatter(x=tr["data"], y=tr["emerg"], mode="lines+markers", name="Emergentes",
        line=dict(color="#10b981", width=2.5), marker=dict(size=6, color="#10b981"),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)"))
    fig.update_layout(**_fig("Evolução: total de termos vs emergentes"), hovermode="x unified")
    return dcc.Graph(figure=fig)


@app.callback(Output("grafico-distribuicao-temas", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _theme_dist(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    dist = hoje.groupby("tema").agg(termos=("termo", "count"), freq=("frequencia", "sum")).reset_index().sort_values("freq", ascending=True)
    fig = px.bar(dist, x="freq", y="tema", color="tema", color_discrete_map=TEMA_CORES,
        text="freq", orientation="h", custom_data=["termos"],
        labels={"freq": "Frequência total", "tema": ""})
    fig.update_layout(**_fig("Distribuição por tema (frequência total)"), showlegend=False)
    fig.update_traces(textposition="outside", textfont=dict(color="#c9d1d9"),
        hovertemplate="%{y}<br>Frequência: %{x}<br>Termos: %{customdata[0]}<extra></extra>")
    fig.update_yaxes(autorange="reversed")
    return dcc.Graph(figure=fig)


@app.callback(Output("grafico-fontes", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _src_chart(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    fc = Counter()
    for f in hoje["fontes"].dropna():
        for s in str(f).split(","):
            if s.strip(): fc[s.strip()] += 1
    if not fc: return dcc.Graph(figure=_empty_fig())
    df = pd.DataFrame(fc.most_common(), columns=["fonte", "contagem"])
    fig = px.bar(df, x="contagem", y="fonte", color="fonte", text="contagem", orientation="h",
        color_discrete_map={"g1": "#3b82f6", "folha": "#10b981", "cnn_brasil": "#f59e0b"})
    fig.update_layout(**_fig("Termos por fonte"), showlegend=False)
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Número de termos")
    fig.update_traces(textposition="outside", textfont=dict(color="#c9d1d9"))
    return dcc.Graph(figure=fig)


@app.callback(Output("grafico-dispersao", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _scatter(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return dcc.Graph(figure=_empty_fig())
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult].copy()
    hoje["size"] = hoje["frequencia"] * 3
    fig = px.scatter(hoje, x="frequencia", y="z_score", color="tema", size="size",
        hover_name="termo", color_discrete_map=TEMA_CORES, labels={"frequencia": "Frequência", "z_score": "Z-Score"})
    fig.update_layout(**_fig("Frequência vs Z-Score por termo"))
    fig.add_hline(y=2, line_dash="dash", line_color="#10b981", opacity=0.5,
        annotation_text="Emergente (z=2)", annotation_font=dict(color="#10b981", size=10))
    fig.add_vrect(x0=5, x1=hoje["frequencia"].max() + 1, y0=2, y1=hoje["z_score"].max() + 1,
        fillcolor="rgba(16, 185, 129, 0.03)", line_width=0)
    return dcc.Graph(figure=fig)


@app.callback(Output("grafico-sentimento-geral", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _sent_overall(temas, fontes, sent, start, end):
    _, d2 = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d2.empty: return dcc.Graph(figure=_empty_fig())
    ult = d2[d2["data"] == d2["data"].max()]
    fig = go.Figure()
    for _, r in ult.iterrows():
        fig.add_trace(go.Bar(name=r["tema"].capitalize(), x=["Positivo", "Negativo", "Neutro"],
            y=[r["pct_positivo"] * 100, r["pct_negativo"] * 100, r["pct_neutro"] * 100],
            marker_color=TEMA_CORES.get(r["tema"], "#6b7280"),
            text=[f"{r['pct_positivo']*100:.0f}%" if r['pct_positivo'] > 0.05 else "",
                  f"{r['pct_negativo']*100:.0f}%" if r['pct_negativo'] > 0.05 else "",
                  f"{r['pct_neutro']*100:.0f}%" if r['pct_neutro'] > 0.05 else ""],
            textposition="inside", textfont=dict(color="white", size=10)))
    fig.update_layout(**_fig("Distribuição de sentimento por tema"), barmode="group", showlegend=True)
    fig.update_yaxes(title="Percentual", tickformat=".0%")
    return dcc.Graph(figure=fig)


@app.callback(Output("wordcloud-container", "children"),
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _wc(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return html.Div("Sem dados para wordcloud", style={"color": "#8b949e", "textAlign": "center", "padding": "40px"})
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    if hoje.empty: return html.Div("Sem dados para wordcloud", style={"color": "#8b949e", "textAlign": "center", "padding": "40px"})
    from wordcloud import WordCloud
    import matplotlib; matplotlib.use("Agg")
    import io, base64
    freq = dict(zip(hoje["termo"], hoje["frequencia"]))
    wc = WordCloud(width=800, height=300, background_color="#1a1d2e", max_words=60,
        colormap="viridis", prefer_horizontal=0.7).generate_from_frequencies(freq)
    buf = io.BytesIO(); wc.to_image().save(buf, format="PNG"); buf.seek(0)
    return html.Img(src=f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
        style={"maxWidth": "100%", "height": "auto", "borderRadius": "8px"})


@app.callback(
    [Output("resumo-termos", "children"), Output("tabela-termos", "data"), Output("tabela-termos", "columns")],
    [Input("filtro-tema", "v"), Input("filtro-fonte", "v"), Input("filtro-sentimento", "v"),
     Input("filtro-periodo", "start_date"), Input("filtro-periodo", "end_date")])
def _table(temas, fontes, sent, start, end):
    d1, _ = _apply_filters(_load_gold1(), _load_gold2(), temas, fontes, sent, start, end)
    if d1.empty: return [], [], []
    ult = d1["data"].max(); hoje = d1[d1["data"] == ult]
    if hoje.empty: return [html.Div("Sem dados hoje", style={"color": "#8b949e", "padding": "12px"})], [], []

    top_freq = hoje.nlargest(5, "frequencia")
    top_tfidf = hoje.nlargest(5, "tfidf_score") if "tfidf_score" in hoje.columns else hoje
    n_emerg = int(hoje["emergente"].dropna().sum())

    lines = []
    terms_freq = ", ".join([f"{r['termo']} ({int(r['frequencia'])})" for _, r in top_freq.iterrows()])
    if top_tfidf is not top_freq:
        terms_tfidf = ", ".join([f"{r['termo']} ({r['tfidf_score']:.3f})" for _, r in top_tfidf.iterrows()])
        lines.append(html.Div([html.Strong("🏆 Top 5 frequência: "), terms_freq], className="insight-card"))
        lines.append(html.Div([html.Strong("💡 Top 5 TF-IDF: "), terms_tfidf], className="insight-card"))
    else:
        lines.append(html.Div([html.Strong("🏆 Top 5 termos: "), terms_freq], className="insight-card"))

    if n_emerg > 0:
        top_emerg = hoje[hoje["emergente"] == True].nlargest(5, "z_score")
        terms_emerg = ", ".join([f"{r['termo']} (z={r['z_score']:.1f})" for _, r in top_emerg.iterrows()])
        lines.append(html.Div([html.Strong("🔥 Emergentes: "), terms_emerg], className="insight-card"))
    else:
        lines.append(html.Div([html.Strong("ℹ️ Nenhum tópico emergente hoje — execute por 14+ dias para acumular baseline")], className="insight-card"))

    cols = [{"name": "Termo", "id": "termo"}, {"name": "Freq", "id": "frequencia"},
            {"name": "TF-IDF", "id": "tfidf_score"}, {"name": "Z-Score", "id": "z_score"},
            {"name": "Tema", "id": "tema"}, {"name": "Sentimento", "id": "sentimento_predominante_termo"}]
    data = hoje.nlargest(50, "frequencia").to_dict("records")
    for r in data:
        r["tfidf_score"] = round(r.get("tfidf_score", 0), 4)
        r["z_score"] = round(r.get("z_score", 0), 2)
    return lines, data, cols


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
