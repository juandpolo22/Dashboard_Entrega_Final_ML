"""Funciones que construyen las figuras del dashboard con una plantilla Plotly única.

Paleta (5 colores): tinta #1f2a30, azul petróleo #1f6f78, naranja #c4561b, gris #8a949b y arena #e9e4dc.
El naranja se reserva para el modelo original y para resaltar; el azul para los demás datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

TINTA, AZUL, NARANJA, GRIS, ARENA = "#1f2a30", "#1f6f78", "#c4561b", "#8a949b", "#e9e4dc"
ESCALA = [[0, "#f4f1ec"], [0.35, "#8fb9bd"], [0.7, AZUL], [1, NARANJA]]

pio.templates["sismos"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=TINTA),
    title=dict(font=dict(size=15), x=0.01, xanchor="left"),
    paper_bgcolor="white", plot_bgcolor="white", colorway=[AZUL, NARANJA, GRIS, "#5b8c5a", "#7a5c99"],
    xaxis=dict(gridcolor="#ece8e2", zerolinecolor="#d9d3ca", linecolor="#d9d3ca", automargin=True, title_standoff=8),
    yaxis=dict(gridcolor="#ece8e2", zerolinecolor="#d9d3ca", linecolor="#d9d3ca", automargin=True, title_standoff=8),
    hoverlabel=dict(bgcolor="white", font_size=12), margin=dict(l=60, r=20, t=50, b=50),
    legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11))))
pio.templates.default = "sismos"


def vacia(texto: str = "Sin datos para los filtros elegidos") -> go.Figure:
    f = go.Figure()
    f.add_annotation(text=texto, showarrow=False, font=dict(size=14, color=GRIS))
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    return f


def histograma(x, titulo, eje_x, eje_y="frecuencia", log_y=False, color=AZUL, nbins=40) -> go.Figure:
    f = go.Figure(go.Histogram(x=x, nbinsx=nbins, marker_color=color, hovertemplate="%{x}: %{y:,}<extra></extra>"))
    f.update_layout(title=titulo, xaxis_title=eje_x, yaxis_title=eje_y, bargap=0.05)
    if log_y:
        f.update_yaxes(type="log")
    return f


def serie(x, y, titulo, eje_y, nombre=None, color=AZUL) -> go.Figure:
    f = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.6), name=nombre or eje_y))
    f.update_layout(title=titulo, xaxis_title="mes", yaxis_title=eje_y)
    return f


def barras_h(etiquetas, valores, titulo, eje_x, resaltar=None, formato=".3f") -> go.Figure:
    colores = [NARANJA if resaltar and resaltar(e) else AZUL for e in etiquetas]
    f = go.Figure(go.Bar(y=list(etiquetas), x=list(valores), orientation="h", marker_color=colores,
                         text=[f"{v:{formato}}" for v in valores], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x:" + formato + "}<extra></extra>"))
    f.update_layout(title=titulo, xaxis_title=eje_x, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40, t=50, b=50))
    return f


def _contorno():
    """Contorno de Colombia (Natural Earth vía world-atlas, dominio público); el mapa no depende de servicios externos."""
    import json
    from . import config as cfg
    ruta = cfg.DATOS / "app" / "contorno.json"
    if not ruta.exists():  # sin contorno el mapa sigue mostrando las celdas
        return go.Scatter(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False)
    c = json.load(open(ruta))
    return go.Scatter(x=c["lon"], y=c["lat"], mode="lines", line=dict(color=GRIS, width=1), hoverinfo="skip", showlegend=False)


def _ejes_mapa(f, alto=520):
    f.update_layout(height=alto, margin=dict(l=40, r=10, t=30, b=40), showlegend=False,
                    xaxis=dict(title="longitud (°)", range=[-80, -66.5], scaleanchor="y", scaleratio=1, constrain="domain", zeroline=False),
                    yaxis=dict(title="latitud (°)", range=[-4.5, 13], zeroline=False))
    return f


def mapa_celdas(lon, lat, valor, titulo, etiqueta_color, texto=None, cmax=None) -> go.Figure:
    """Celdas de 0,5° coloreadas por un valor (escala log1p), sobre el contorno de Colombia."""
    v = np.asarray(valor, float)
    tope = np.log1p(cmax if cmax else max(v.max(), 1))
    marcas = [m for m in (0, 1, 5, 20, 60, 200) if np.log1p(m) <= tope + 1e-9]
    f = go.Figure([_contorno(), go.Scatter(x=lon, y=lat, mode="markers", text=texto, hovertemplate="%{text}<extra></extra>",
                                           marker=dict(symbol="square", size=9, color=np.log1p(v), colorscale=ESCALA, cmin=0, cmax=tope,
                                                       line=dict(width=0), colorbar=dict(title=etiqueta_color, thickness=10,
                                                       tickvals=np.log1p(marcas), ticktext=[str(m) for m in marcas])))])
    f.update_layout(title=titulo)
    return _ejes_mapa(f)


def mapa_puntos(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Sismos como puntos: tamaño según la magnitud y color según la profundidad."""
    f = go.Figure([_contorno(), go.Scattergl(x=df.longitude, y=df.latitude, mode="markers",
                   marker=dict(size=3 + (df.mag - 2.5) * 2.6, color=df.depth.clip(0, 200), colorscale=ESCALA[::-1], cmin=0, cmax=200,
                               opacity=0.7, line=dict(width=0), colorbar=dict(title="prof. (km)", thickness=10)),
                   text=[f"{t:%Y-%m-%d} · M {m:.1f} · {d:.0f} km · {p}" for t, m, d, p in zip(df.time, df.mag, df.depth, df.depto)],
                   hovertemplate="%{text}<extra></extra>")])
    f.update_layout(title=titulo)
    return _ejes_mapa(f)


def correlacion(C: pd.DataFrame, titulo: str) -> go.Figure:
    f = go.Figure(go.Heatmap(z=C.values, x=C.columns, y=C.index, zmin=-1, zmax=1,
                             colorscale=[[0, NARANJA], [0.5, "white"], [1, AZUL]], text=np.round(C.values, 2),
                             texttemplate="%{text}", textfont=dict(size=9), hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    f.update_layout(title=titulo, height=620, xaxis=dict(tickangle=-45), margin=dict(l=10, r=10, t=50, b=10))
    return f


def diagnostico_regresion(y, p, tm, nombre: str = "") -> go.Figure:
    """Diagnóstico de un modelo de conteos en escala log1p (la de la métrica RMSLE), con 4 paneles:
    reales frente a predichos, residuos frente a predichos, ACF del residuo medio mensual e histograma de residuos."""
    from plotly.subplots import make_subplots
    from statsmodels.tsa.stattools import acf
    ly, lp = np.log1p(np.asarray(y, float)), np.log1p(np.clip(np.asarray(p, float), 0, None))
    r = ly - lp
    meses = np.unique(tm)
    pos = np.searchsorted(meses, tm)
    r_mes = np.bincount(pos, r) / np.bincount(pos)
    nl = min(18, len(meses) // 3)
    a = acf(r_mes, nlags=nl)[1:]
    banda = 1.96 / np.sqrt(len(meses))
    f = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.1,
                      subplot_titles=("Reales frente a predichos (log1p)", "Residuos frente a predichos",
                                      "ACF del residuo medio mensual", "Histograma de residuos (log1p)"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ly), size=min(4000, len(ly)), replace=False)
    f.add_trace(go.Scattergl(x=lp[idx], y=ly[idx] + rng.normal(0, 0.015, len(idx)), mode="markers",
                             marker=dict(size=4, color=AZUL, opacity=0.35), showlegend=False,
                             hovertemplate="predicho %{x:.2f}<br>real %{y:.2f}<extra></extra>"), row=1, col=1)
    tope = float(max(lp.max(), ly.max()))
    f.add_trace(go.Scatter(x=[0, tope], y=[0, tope], mode="lines", line=dict(color=NARANJA, dash="dash", width=1.5),
                           showlegend=False, hoverinfo="skip"), row=1, col=1)
    f.add_trace(go.Scattergl(x=lp[idx], y=r[idx], mode="markers", marker=dict(size=4, color=AZUL, opacity=0.35),
                             showlegend=False, hovertemplate="predicho %{x:.2f}<br>residuo %{y:.2f}<extra></extra>"), row=1, col=2)
    f.add_hline(y=0, line_color=NARANJA, line_dash="dash", row=1, col=2)
    f.add_trace(go.Bar(x=np.arange(1, nl + 1), y=a, marker_color=AZUL, showlegend=False,
                       hovertemplate="rezago %{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    f.add_hrect(y0=-banda, y1=banda, fillcolor=ARENA, opacity=0.9, line_width=0, layer="below", row=2, col=1)
    tope_acf = max(1.5 * banda, 1.15 * float(np.abs(a).max()))
    f.update_yaxes(range=[-tope_acf, tope_acf], row=2, col=1)        # deja ver la banda ±1,96/√T completa
    f.add_trace(go.Histogram(x=r, nbinsx=60, marker_color=AZUL, showlegend=False,
                             hovertemplate="%{x}: %{y:,}<extra></extra>"), row=2, col=2)
    f.update_xaxes(title_text="log(1 + predicho)", row=1, col=1); f.update_yaxes(title_text="log(1 + real)", row=1, col=1)
    f.update_xaxes(title_text="log(1 + predicho)", row=1, col=2); f.update_yaxes(title_text="residuo (log1p)", row=1, col=2)
    f.update_xaxes(title_text="rezago (meses)", row=2, col=1); f.update_yaxes(title_text="autocorrelación", row=2, col=1)
    f.update_xaxes(title_text="residuo = log(1+y) − log(1+ŷ)", row=2, col=2); f.update_yaxes(title_text="celdas-mes", type="log", row=2, col=2)
    f.update_layout(height=640, margin=dict(l=60, r=20, t=50, b=50), title=nombre)
    return f
