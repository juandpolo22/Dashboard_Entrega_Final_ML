"""Sección 3 · Modelos: tabla de líderes, predicción en vivo con explicación local, diagnósticos, interpretabilidad,
comparación estadística y el modelo original."""
import dash
import inspect
import textwrap
from functools import lru_cache
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from plotly.subplots import make_subplots

from src import artefactos as A
from src import data as D
from src import plots as pl
from src.datos import VARIABLES_BASE
from src import presentacion_contenido as P
from src import propuesto as PR
from src.original_model import COLUMNAS, PSEUDOCODIGO

dash.register_page(__name__, path="/modelos", name="Modelos", order=2)

CFG = {"displayModeBar": False, "responsive": True}
AB = "A+B v2"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

PSEUDO = PSEUDOCODIGO


def _t(titulo, sub, *cuerpo):
    return html.Div([html.H2(titulo), html.P(sub, className="sub"), *cuerpo], className="tarjeta")


def _g(id_, alto=380):
    return dcc.Loading(dcc.Graph(id=id_, config=CFG, style={"height": f"{alto}px"}), type="dot", color=pl.AZUL)


# ---------------------------------------------------------------- estilo común de las tablas
CABECERA = {"backgroundColor": "#1d2330", "color": "#ffffff", "fontWeight": "600", "border": "none", "padding": "10px",
            "whiteSpace": "normal", "height": "auto", "fontSize": 12.5, "textAlign": "left"}
CELDA = {"fontFamily": "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif", "fontSize": 13, "padding": "8px 10px",
         "border": "none", "borderBottom": "1px solid #ebe6de", "textAlign": "left", "whiteSpace": "normal", "height": "auto",
         "backgroundColor": "#ffffff", "color": "#1f2a30"}
MARCO = {"overflowX": "auto", "borderRadius": "10px", "border": "1px solid #e2ddd5"}
FILTRO = {"backgroundColor": "#f4f1ec", "border": "none", "borderBottom": "1px solid #e2ddd5"}


def _fmt(decimales=4):
    """Formato numérico con coma decimal; los vacíos se muestran como «—»."""
    return Format(precision=decimales, scheme=Scheme.fixed, decimal_delimiter=",", nully="—")


def _p_txt(p):
    return "—" if pd.isna(p) else ("< 0,0001" if p < 0.0001 else f"{p:.4f}".replace(".", ","))


def _dt(df, columnas, numericas=None, derecha=(), resaltar=None, extra_cond=(), **kw):
    """DataTable con el estilo del tablero: cabecera oscura, filas alternas, números a la derecha y la fila de A+B resaltada."""
    numericas = numericas or {}
    cols = []
    for c in columnas:
        d = {"name": c, "id": c}
        if c in numericas:
            d.update(type="numeric", format=_fmt(numericas[c]))
        cols.append(d)
    cond = [{"if": {"row_index": "odd"}, "backgroundColor": "#f8f6f2"}]
    if resaltar:
        col, texto = resaltar
        cond += [{"if": {"filter_query": f'{{{col}}} contains "{texto}"'}, "backgroundColor": "#fbeee6", "fontWeight": "700", "color": "#8a3a10"},
                 {"if": {"filter_query": f'{{{col}}} contains "{texto}"', "column_id": col}, "borderLeft": "4px solid #c4561b"}]
    cond += list(extra_cond)
    derecha_cols = list(numericas) + list(derecha)
    alinear = [{"if": {"column_id": c}, "textAlign": "right", "fontVariantNumeric": "tabular-nums", "whiteSpace": "nowrap"} for c in derecha_cols]
    return dash_table.DataTable(df[columnas].to_dict("records"), cols, style_table=MARCO, style_header=CABECERA, style_cell=CELDA,
                                style_cell_conditional=alinear, style_data_conditional=cond, **kw)


def _insignia(texto, tipo):
    return html.Span(texto, className=f"insignia insignia-{tipo}")


CELDA_EJEMPLO = 620  # celda que se carga al abrir: Valle del Cauca (4,75; −75,75)


def _celda_inicial():
    valores = [o["value"] for o in _celdas_opciones()]
    return CELDA_EJEMPLO if CELDA_EJEMPLO in valores else valores[0]


def _celdas_opciones():
    F = D.panel()
    F = F[F.y.isna()].sort_values("n_12m", ascending=False)
    return [{"label": f"({r.lat:.2f}, {r.lon:.2f}) · {str(r.depto).title()} · {int(r.n_12m)} sismos en 12 meses", "value": int(r.celda)}
            for r in F.itertuples()]


def _campo(e):
    paso = 1 if e["entero"] else 0.01
    return dbc.Col([
        dbc.Label(e["columna"], html_for=f"in-{e['columna']}", className="mb-0 small fw-bold", style={"wordBreak": "break-all"}),
        dbc.Input(id=f"in-{e['columna']}", type="number", min=e["min_valido"], max=e["max_valido"], step=paso, size="sm",
                  placeholder="vacío = sin sismos" if e["admite_faltante"] else None),
        html.Div(f"{e['descripcion']} · observado [{e['min']:g} – {e['max']:g}]", className="text-muted", style={"fontSize": "11px"}),
    ], xs=6, md=4, lg=3, className="mb-2")


def bloque_en_vivo():
    """Tarjeta de predicción en vivo (formulario, predicción y explicación). Se reutiliza en la página de presentación."""
    esquema = D.esquema_formulario()
    return _t("Predicción en vivo", "Elija una celda: el formulario se llena con sus valores actuales; puede modificarlos y predecir. "
                       "Cada campo valida tipo y límites físicos; si un valor supera lo observado en el entrenamiento, se avisa que el pronóstico extrapola.",
                       dcc.Dropdown(id="p-celda", options=_celdas_opciones(), value=_celda_inicial(),
                       placeholder="Elija una celda", className="mb-2"),
                       dbc.Row([_campo(e) for e in esquema], className="g-1"),
                       dbc.Button("Predecir", id="p-boton", color="primary", className="mt-1"),
                       html.Div(id="p-alerta", className="mt-2"),
                       dcc.Loading(html.Div(id="p-salida", className="mt-2"), type="dot", color=pl.AZUL))


def _fuente(objeto):
    """Código fuente tal como está en src/propuesto.py (se lee al abrir la página, así no se desincroniza)."""
    return textwrap.dedent(inspect.getsource(objeto))


def _tabla_param(filas):
    return html.Table([html.Thead(html.Tr([html.Th("Símbolo o parámetro"), html.Th("Qué es")])),
                       html.Tbody([html.Tr([html.Td(html.Code(a)), html.Td(b)]) for a, b in filas])], className="tabla-pres")


def _bloque_codigo():
    """Pipeline y código de A (softmax) y de B (boosting ZINB), con sus parámetros, en secciones desplegables."""
    o = D.original()
    K = len(o["pesos_medios"])
    pasos = [("1 · Fuera de pliegue", "cada miembro predice bloques de meses que no vio al entrenarse"),
             ("2 · Modelo B", "boosting ZINB con parada temprana cronológica"),
             ("3 · Combinador A", f"pesos softmax por celda-mes sobre {K} miembros"),
             ("4 · Calibración", "recta c0 + c1·ŷ con el 20 % final de los meses"),
             ("5 · Pronóstico", "expm1 del resultado, con recorte en 0")]
    filas_A = [
        ("Z (n × d)", "meta-características de cada celda-mes: sismos recientes (1, 3 y 12 meses), sismos fuertes, vecinos, "
                      "excitación, componentes principales del historial, fracción de meses sin sismos, magnitud máxima, "
                      "mes (seno y coseno) y una constante."),
        ("B (n × K)", f"pronósticos de los K = {K} miembros en log1p, fuera de pliegue cronológico."),
        ("y", "objetivo en log1p: log(1 + número de sismos)."),
        ("T (d × K)", "matriz de parámetros aprendida. Cada columna dice cuánto empuja cada rasgo de contexto hacia un miembro."),
        ("W = softmax(Z·T)", "pesos por fila: positivos y suman 1. softmax(v)_k = e^(v_k) / Σ_j e^(v_j)."),
        ("λ (l2)", "penalización sobre T, sin el intercepto. Se elige entre 0,001, 0,01, 0,1 y 1 con el 20 % final de los meses. "
                   "Un λ grande acerca los pesos a iguales."),
        ("Pérdida", "error cuadrático medio en log1p más λ·ΣT². Se minimiza con L-BFGS-B usando el gradiente analítico."),
    ]
    filas_B = [
        ("n_max = 300", f"máximo de iteraciones. La parada temprana eligió {o['n_iter_B']} iteraciones en la última corrida."),
        ("lr = 0,08", "tasa de aprendizaje: cuánto aporta cada árbol nuevo."),
        ("depth = 3", "profundidad máxima de cada árbol CART propio."),
        ("min_leaf = 50", "mínimo de filas por hoja."),
        ("η", "log de la media μ = e^η de la binomial negativa. Se actualiza sumando los aportes de los árboles."),
        ("ξ", "logit de la probabilidad de cero estructural π = σ(ξ)."),
        ("α (la)", "dispersión de la binomial negativa; se re-estima por máxima verosimilitud cada 20 iteraciones."),
        ("g / h", "gradiente y curvatura de la verosimilitud. El árbol se ajusta a g/h (recortado a [−3, 3]) con pesos h."),
        ("Parada", "iteración con menor NLL en el 20 % final de los meses; después se reentrena con todos los datos."),
    ]
    return html.Div([
        html.H4("9. Código y pipeline", className="mt-4"),
        html.P("Secciones desplegables: haga clic en cada título para abrirlo o cerrarlo.", className="sub"),
        html.Details([html.Summary(html.B("Pipeline completo (5 pasos)")),
                      html.Div(html.Div([html.Div(f"{a}: {b}", className="paso-flujo") for a, b in pasos], className="flujo-pres")),
                      html.Pre(PSEUDOCODIGO, className="pseudo mt-2")], className="tarjeta mt-2"),
        html.Details([html.Summary(html.B("Combinador A: softmax contextual, con sus parámetros")),
                      _tabla_param(filas_A),
                      html.H5("Código (src/propuesto.py, clase CombinadorSoftmax)", className="mt-3"),
                      html.Pre(_fuente(PR.CombinadorSoftmax._f) + "\n" + _fuente(PR.CombinadorSoftmax.fit) + "\n"
                               + _fuente(PR.CombinadorSoftmax.pesos) + "\n" + _fuente(PR.CombinadorSoftmax.predict), className="pseudo")],
                     className="tarjeta mt-2"),
        html.Details([html.Summary(html.B("Modelo B: boosting ZINB propio, con sus parámetros")),
                      _tabla_param(filas_B),
                      html.H5("Código (src/propuesto.py, clase BoostingZINB y función grads_zinb)", className="mt-3"),
                      html.Pre(_fuente(PR.grads_zinb) + "\n" + _fuente(PR.BoostingZINB._run), className="pseudo")],
                     className="tarjeta mt-2"),
    ])


@lru_cache(maxsize=1)
def _celdas_mapa():
    """Actividad media (sismos M ≥ 2,5 por mes) de cada celda activa, igual que en Contexto."""
    P = D.panel()
    obs = P[P.y.notna()]
    return obs.groupby(["celda", "lat", "lon"]).y.mean().reset_index()


def _figura_mapa(celda=None):
    """Mapa de celdas con un círculo en la celda elegida."""
    c = _celdas_mapa()
    texto = [f"({a:.2f}, {o:.2f}) · {v:.2f} sismos/mes" for a, o, v in zip(c.lat, c.lon, c.y)]
    f = pl.mapa_celdas(c.lon, c.lat, c.y, "", "sismos/mes", texto=texto)
    r = c[c.celda == celda] if celda is not None else c.iloc[:0]
    if len(r):
        r = r.iloc[0]
        f.add_trace(go.Scatter(x=[r.lon], y=[r.lat], mode="markers", hoverinfo="skip", showlegend=False,
                               marker=dict(symbol="circle-open", size=20, color="#1d2330", line=dict(width=3))))
    f.update_layout(height=430, margin=dict(l=40, r=10, t=10, b=40))
    return f


def _tarjeta_mapa():
    inicial = _celda_inicial()
    return _t("Mapa de celdas", "Color = sismos por mes (escala logarítmica). El círculo marca la celda elegida en la predicción.",
              dcc.Graph(id="p-mapa", figure=_figura_mapa(inicial), config={"displayModeBar": False}))


@callback(Output("p-mapa", "figure"), Input("p-celda", "value"))
def mapa_celda(celda):
    return _figura_mapa(celda)


# ---------------------------------------------------------------- sección 5: variables y faltantes
GRUPOS_VARIABLES = [
    ("Conteos propios de la celda", "n_1m, n_3m, n_6m, n_12m, n35_3m, n35_12m, n_mismo_mes_anio_prev",
     "la autocorrelación es positiva hasta 12 meses: la actividad de un mes anticipa la del siguiente (EDA temporal)"),
    ("Magnitudes", "mag_media_3m, mag_media_12m, mag_max_3m, mag_max_12m",
     "el número de réplicas crece con la magnitud del sismo principal (productividad sísmica)"),
    ("Vecindad", "vec_1m, vec_3m", "I de Moran > 0 y prueba de Knox: la actividad se agrupa en espacio y tiempo (EDA espacial)"),
    ("Profundidad", "prof_media_hist", "separa el nido de Bucaramanga (profundidad intermedia) de la sismicidad superficial"),
    ("Ubicación", "lat, lon", "la relación con la ubicación no es monótona: hay diferencias regionales (EDA bivariado y espacial)"),
    ("Calendario", "sen_mes, cos_mes", "no hay estacionalidad clara (fuerza 0,04); se incluyen por completitud y se ponen a prueba"),
]
NOMBRES_CONJUNTO = {"todas (18)": "Todas las variables", "sin lat/lon (16)": "Sin ubicación (lat, lon)",
                    "sin calendario (16)": "Sin calendario (sen_mes, cos_mes)", "sin vecindad (16)": "Sin vecindad (vec_1m, vec_3m)",
                    "solo conteos propios (7)": "Solo conteos propios de la celda", "sin magnitudes (14)": "Sin las 4 magnitudes"}
CONTEOS_PROPIOS = ["n_1m", "n_3m", "n_6m", "n_12m"]


def _tabla_html(cabecera, filas, num=(), clase_fila=None):
    """Tabla HTML con el estilo del tablero. ``num``: índices de columnas numéricas (a la derecha)."""
    th = [html.Th(c, className="num" if i in num else None) for i, c in enumerate(cabecera)]
    trs = []
    for k, f in enumerate(filas):
        tds = [html.Td(v, className="num" if i in num else None) for i, v in enumerate(f)]
        trs.append(html.Tr(tds, className=clase_fila(k) if clase_fila else None))
    return html.Div(html.Table([html.Thead(html.Tr(th)), html.Tbody(trs)], className="tabla-pres"), className="wrap-tabla")


def _bloque_variables():
    """Cómo se eligieron las variables de la predicción y qué se hizo con los faltantes (observaciones del docente)."""
    conj = D.tabla("seleccion_conjuntos.csv")
    imp = D.tabla("seleccion_imputacion.csv").set_index("prueba").RMSLE_CV_media
    falt = D.tabla("faltantes_panel.csv").set_index("index").iloc[:, 0]
    vif_max = float(D.tabla("vif.csv").VIF.max())
    sg = D.shap_global()
    n_pl = sum(c.startswith("pliegue_") for c in conj.columns)
    ref = conj[conj.conjunto == "todas (18)"].iloc[0]
    umbral = ref.RMSLE_CV_media + ref.RMSLE_CV_desv / np.sqrt(n_pl)
    filas_cv = []
    for r in conj.itertuples():
        cambio = r.RMSLE_CV_media - ref.RMSLE_CV_media
        lectura = ("referencia" if r.conjunto == "todas (18)" else
                   _insignia("igual (dentro de 1 error estándar)", "neutra") if r.RMSLE_CV_media <= umbral else _insignia("empeora", "no"))
        filas_cv.append([NOMBRES_CONJUNTO.get(r.conjunto, r.conjunto), str(int(r.n_variables)), P._n(r.RMSLE_CV_media),
                         "—" if r.conjunto == "todas (18)" else "+" + P._n(cambio), lectura])
    c_imp, m_imp = imp["imputación de magnitud: constante"], imp["imputación de magnitud: mediana"]
    sin_mag = float(conj.loc[conj.conjunto == "sin magnitudes (14)", "RMSLE_CV_media"].iloc[0])

    # importancia SHAP (% del total) en los dos campeones de árboles
    def pct(nombre):
        v = pd.Series(sg[nombre]["valores"], dtype=float)
        return 100 * v / v.sum()
    xgb, rf = pct("XGBoost · TPE"), pct("Random Forest · AG")
    orden = xgb.sort_values().index
    fig = go.Figure([go.Bar(y=list(orden), x=rf[orden].values, orientation="h", name="Random Forest · AG", marker_color=P.COL_NEUTRO,
                            hovertemplate="%{y}: %{x:.1f} %<extra>Random Forest</extra>"),
                     go.Bar(y=list(orden), x=xgb[orden].values, orientation="h", name="XGBoost · TPE", marker_color=P.COL_OTRO,
                            hovertemplate="%{y}: %{x:.1f} %<extra>XGBoost</extra>")])
    fig.update_layout(barmode="group", xaxis_title="% de la importancia total (|SHAP| medio)",
                      legend=dict(orientation="h", y=1.04, x=0, traceorder="reversed"))
    P._layout_fig(fig, 560, 150)
    fig.update_layout(margin=dict(l=150, r=30, t=40, b=45))
    cp_x, cp_r = xgb[CONTEOS_PROPIOS].sum(), rf[CONTEOS_PROPIOS].sum()
    cal = max(xgb[["sen_mes", "cos_mes"]].sum(), rf[["sen_mes", "cos_mes"]].sum())

    tarjeta_var = html.Div([
        html.H3("¿Cómo determinamos las variables que usa la predicción?", className="h5"),
        html.P("No elegimos las variables con p-values una por una: esa prueba supone un modelo lineal, y aquí los conteos son "
               "asimétricos y con exceso de ceros. Seguimos cinco pasos, y cada decisión se probó solo con datos de entrenamiento."),
        html.Div([html.Div(x, className="paso-flujo") for x in (
            "1 · Candidatas: cada variable sale de un hallazgo del EDA",
            "2 · Sin fuga: solo información hasta el cierre del mes t",
            f"3 · Colinealidad: VIF máximo {P._n(vif_max, 1)} (< 10), no se elimina ninguna",
            "4 · Validación cruzada temporal: se quita cada grupo y se mide el RMSLE",
            "5 · Importancia: SHAP en los campeones")], className="flujo-pres"),
        html.H4("Paso 1 · Grupos de variables y el hallazgo que los justifica", className="h6 mt-3"),
        _tabla_html(["Grupo", "Variables", "Por qué entra"], [[g, html.Code(v), r] for g, v, r in GRUPOS_VARIABLES]),
        html.H4(f"Paso 4 · Qué pasa si se quita cada grupo (validación cruzada temporal, {n_pl} pliegues)", className="h6 mt-3"),
        _tabla_html(["Conjunto de variables", "N.º", "RMSLE medio", "Cambio frente a todas", "Lectura"], filas_cv, num=(1, 2, 3),
                    clase_fila=lambda k: "fila-ab" if k == 0 else None),
        html.P(f"Modelo usado en esta prueba: SVR lineal base (Entregable 2, capítulo 10). Umbral de un error estándar: "
               f"{P._n(umbral)}.", className="sub mt-1"),
        html.H4("Paso 5 · Qué variables pesan más en los modelos entrenados (SHAP)", className="h6 mt-3"),
        dcc.Graph(figure=fig, config=CFG),
        html.P("Para B y para el contexto de A se agregaron, además, 13 variables de excitación: actividad sísmica reciente de la "
               "celda y sus alrededores, ponderada por magnitud, distancia (núcleos de 15, 30 y 60 km) y tiempo, al estilo de los "
               "modelos ETAS, y la historia de los últimos 12 meses de la celda. Su aporte se midió quitando B: sin B, el RMSLE "
               "sube de 0,3260 a 0,3267.", className="mt-2"),
        P._conclusion(f"Las variables más importantes son los conteos recientes de la celda (n_12m, n_6m, n_3m y n_1m: "
                      f"{P._n(cp_x, 0)} % de la importancia en XGBoost y {P._n(cp_r, 0)} % en Random Forest), la actividad de las "
                      f"celdas vecinas (vec_3m) y la profundidad. Quitar las magnitudes empeora el RMSLE de {P._n(ref.RMSLE_CV_media)} a "
                      f"{P._n(sin_mag)}. Quitar el calendario, la ubicación o la vecindad no cambia el error del modelo lineal; se "
                      f"conservan porque quedan dentro de un error estándar y los modelos de árboles sí usan la vecindad y la ubicación. "
                      f"El calendario aporta menos del {P._n(np.ceil(cal), 0)} %."),
    ], className="tarjeta")

    p3, p12, pp = (P._n(falt[c], 1) + " %" for c in ("mag_media_3m", "mag_media_12m", "prof_media_hist"))
    si, no, pred = _insignia("Sí", "si"), _insignia("No", "no"), _insignia("Solo para predecir", "neutra")
    filas_f = [
        [html.Code("mag_media_3m, mag_max_3m"), p3, "No hubo sismos en los últimos 3 meses: no hay magnitud que medir. Es un faltante "
         "estructural (la prueba de Little rechaza que sea al azar).", "Benchmarks: se imputa 2,5, el umbral del catálogo. B: se codifica "
         "como −1, para que los árboles lo traten como «sin sismos».", si],
        [html.Code("mag_media_12m, mag_max_12m"), p12, "No hubo sismos en los últimos 12 meses.",
         "Igual que la fila anterior. En el contexto de A, mag_max_12m faltante se reemplaza por 2,5.", si],
        [html.Code("prof_media_hist"), pp, "La celda todavía no había tenido sismos.", "Benchmarks: mediana del entrenamiento. B: −1.", si],
        ["Las otras 13 variables", "0 %", "—", "Sin cambios.", si],
        ["Objetivo (sismos del mes siguiente)", "1 fila por celda", "Es el mes que se pronostica: todavía no ocurre.",
         "No se usa para entrenar; es la fila que el modelo predice.", pred],
        [html.Code("depto"), "solo en el catálogo crudo", "Sismos en el mar o en zonas de frontera (lo explica la ubicación).",
         "No se imputa: esos registros quedan fuera por la regla de homogeneidad entre portales del catálogo.", no],
    ]
    tarjeta_falt = html.Div([
        html.H3("¿Qué hicimos con los faltantes?", className="h5"),
        html.P([html.B("Ninguna fila del panel se eliminó por tener faltantes en las variables: "), "todas se usan para entrenar y predecir. "
                "Los faltantes se imputan dentro del pipeline, que se ajusta solo con datos de entrenamiento para no filtrar "
                "información del futuro."]),
        _tabla_html(["Variable", "% faltante (entrenamiento)", "Por qué falta", "Qué se hizo", "¿Se usa?"], filas_f, num=(1,)),
        html.P(["¿Por qué 2,5 y no la mediana? Con validación cruzada temporal, imputar 2,5 dio menor error que imputar la mediana "
                f"(RMSLE {P._n(c_imp)} frente a {P._n(m_imp)}), y además conserva el significado «no hubo sismos por encima del umbral»."],
               className="mt-2"),
        P._conclusion("Los faltantes de magnitud no son errores de medición: significan que no hubo sismos. Por eso no se eliminan "
                      "filas, sino que se imputa un valor que conserva ese significado. En los modelos de árboles el valor exacto "
                      "importa poco, porque n_3m = 0 ya marca el caso. En la simulación, dejar vacío un campo de magnitud equivale a "
                      "decir que no hubo sismos en esa ventana."),
    ], className="tarjeta")
    return html.Div([
        html.H4("5. Variables del modelo: cómo se eligieron y qué se hizo con los faltantes", className="mt-4"),
        tarjeta_var, tarjeta_falt])


def _lideres_card():
    L = D.lideres().copy()
    L["RMSLE bucle externo (media ± d.e.)"] = L["RMSLE bucle externo (media ± d.e.)"].str.replace(".", ",", regex=False)
    cols = ["modelo", "optimizador", "RMSLE bucle externo (media ± d.e.)", "RMSLE validación < 2016",
            "RMSLE origen móvil 2021–2026", "inferencia (ms/fila)"]
    return _t("Tabla de líderes", "Mejor combinación de cada familia y el modelo original (tarea de regresión, sin balanceo de clases). "
              "Ordene con los encabezados y filtre escribiendo en la fila superior (por ejemplo «XGB» o «< 0.33»).",
              _dt(L, cols, numericas={"RMSLE validación < 2016": 4, "RMSLE origen móvil 2021–2026": 4, "inferencia (ms/fila)": 4},
                  derecha=("RMSLE bucle externo (media ± d.e.)",), resaltar=("modelo", "A+B"), sort_action="native",
                  filter_action="native", style_filter=FILTRO,
                  filter_options={"placeholder_text": "filtrar…", "case": "insensitive"}),
              html.Div([html.B("Criterio de selección: "), "en cada familia gana la combinación con menor RMSLE en la validación interna con "
                        "datos anteriores a 2016 (la misma que fija sus hiperparámetros); con ella se evalúa el origen móvil 2021–2026 y se "
                        "hace la comparación estadística. A+B se evalúa directamente en el origen móvil, por eso no tiene bucle externo ni "
                        "validación < 2016."], className="insight"))


def layout():
    """Sección única «Modelos»: A+B, pruebas, B, líderes, modelo original y, al final, la simulación en vivo."""
    dm, gw, spa, mcs, ab, reg = P._datos()
    return dbc.Container([
        html.H2("Modelos: A+B y las pruebas que lo respaldan", className="mt-3"),
        html.P("Pronóstico mensual de sismos M ≥ 2,5 por celda de 0,5° en Colombia · origen móvil 2021–2026 (69 meses).",
               className="text-muted"),
        P._explicacion_ab(),
        html.H4("2. Pruebas estadísticas: qué comprueba cada una", className="mt-4"),
        html.P("Cada prueba responde a la pregunta «¿la diferencia de error entre A+B y el campeón es real o es azar?». "
               "p menor a 0,05 significa que, si no hubiera diferencia, sería muy raro observar lo que observamos."),
        *P._pruebas(dm, gw, spa, mcs),
        html.H4("3. Comportamiento de B y comparación con combinaciones", className="mt-4"),
        *P._modelo_b_y_combinaciones(ab, reg),
        html.H4("4. Líderes y modelo original", className="mt-4"),
        _lideres_card(),
        _original(D.original()),
        _bloque_variables(),
        html.H4("6. Simulación en vivo", className="mt-4"),
        html.P("Elija el modelo y una celda, revise o cambie sus valores, y pulse «Predecir». Los campos de magnitud y "
               "profundidad se pueden dejar vacíos: significa que no hubo sismos en esa ventana, y el modelo lo trata igual que en "
               "el entrenamiento (sección 5)."),
        P._tabla_entradas_salidas(),
        dbc.Row([dbc.Col(html.Div([
            html.Label("Modelo para la predicción, el diagnóstico y la interpretabilidad", className="fw-bold"),
            dcc.Dropdown(id="m-sel", options=[{"label": "A+B v2 (modelo original)" if m == AB else m, "value": m} for m in D.modelos()],
                         value=AB, clearable=False, className="mb-3"),
        ]), md=6)]),
        dbc.Row([dbc.Col(bloque_en_vivo(), lg=7), dbc.Col(_tarjeta_mapa(), lg=5)], className="mt-2"),
        _bloque_diagnostico(),
        _bloque_codigo(),
    ], fluid=True, className="pagina-pres pb-4")


def _tabla_pruebas():
    dm, gc = D.tabla("e2_stats_dm.csv").set_index("rival"), D.tabla("e2_stats_gw_cw.csv").set_index("rival")
    mcs = D.tabla("e2_stats_mcs_origen_movil.csv")
    peor = dm["p Holm (7)"].idxmax()
    c = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    filas = [
        {"prueba": "Model Confidence Set (Hansen et al., 2011)", "hipótesis nula": "todos los modelos del conjunto tienen igual pérdida esperada",
         "supuestos verificados": "pérdidas mensuales estacionarias y débilmente dependientes (bootstrap estacionario)",
         "estadístico": "Tmax", "p crudo": _p_txt(mcs.p_MCS.iloc[1]), "p ajustado": "no requiere (controla la familia)",
         "efecto": "—", "conclusión": "solo A+B en el conjunto al 90 %"},
        {"prueba": "Diebold-Mariano con corrección HLN", "hipótesis nula": "igual pérdida esperada que el campeón",
         "supuestos verificados": "d_t sin autocorrelación fuerte (EL, BG); varianza Newey-West",
         "estadístico": f"DM = {c(dm.loc[peor, 'DM-HLN'])} (vs {peor})", "p crudo": _p_txt(dm.loc[peor, "p crudo"]),
         "p ajustado": f"Holm (7): {_p_txt(dm['p Holm (7)'].max())} (máx.)",
         "efecto": f"d = {c(dm.loc[peor, 'd de Cohen'])}; mín. {c(dm['d de Cohen'].min())}", "conclusión": "A+B mejor que los 7 campeones"},
        {"prueba": "Giacomini-White (habilidad condicional)", "hipótesis nula": "E[d_t | constante, d_{t−1}] = 0",
         "supuestos verificados": "ventana de reestimación finita (origen móvil anual)",
         "estadístico": f"χ²(2) = {c(gc.loc[peor, 'GW χ²(2)'], 1)} (vs {peor})", "p crudo": _p_txt(gc.loc[peor, "p GW"]),
         "p ajustado": f"Holm (7): {_p_txt(gc['p GW Holm'].max())} (máx.)", "efecto": "—", "conclusión": "A+B mejor en los 7"},
    ]
    T = pd.DataFrame(filas)
    verde = [{"if": {"column_id": "conclusión"}, "color": "#2e7d4f", "fontWeight": "700"},
             {"if": {"column_id": "prueba"}, "fontWeight": "700", "minWidth": "150px"}]
    return _dt(T, list(T.columns), extra_cond=verde)


def _tabla_reglas():
    R = D.tabla("reglas_combinacion.csv").copy()
    R = R[~R["regla"].str.contains("stacking|sin calibración", case=False)]
    R["veredicto"] = np.where(R["p Holm"].isna(), "referencia", np.where(R["p Holm"] < 0.05, np.where(R["dif. pérdida vs A+B"] > 0, "A+B mejor", "A+B peor"),
                                                                        "sin diferencia significativa"))
    R.loc[R["p Holm"].isna(), "dif. pérdida vs A+B"] = np.nan          # la fila de referencia no se compara consigo misma
    R["p DM"], R["p Holm"] = R["p DM"].map(_p_txt), R["p Holm"].map(_p_txt)
    cols = ["regla", "RMSLE 2021–2026", "dif. pérdida vs A+B", "p DM", "p Holm", "veredicto"]
    verde = [{"if": {"filter_query": '{veredicto} = "A+B mejor"', "column_id": "veredicto"}, "color": "#2e7d4f", "fontWeight": "700"}]
    return html.Div([
        _dt(R, cols, numericas={"RMSLE 2021–2026": 4, "dif. pérdida vs A+B": 4}, derecha=("p DM", "p Holm"),
            resaltar=("regla", "A+B"), extra_cond=verde),
        html.Div([html.B("Lectura: "), "con los mismos miembros, la regla propia supera al mejor miembro, a los pesos por error global y al "
                  "promedio simple; la ventaja frente a los campeones viene sobre todo de los miembros (en particular del componente "
                  "propio B) y de combinarlos con pesos que cambian según el contexto."], className="insight")])


def _original(o):
    curva = np.array(o["curva_validacion_B"])
    f = go.Figure(go.Scatter(x=np.arange(1, len(curva) + 1), y=curva, mode="lines", line=dict(color=pl.NARANJA)))
    f.add_vline(x=o["n_iter_B"], line_dash="dot", line_color=pl.GRIS, annotation_text=f"parada: {o['n_iter_B']} iteraciones")
    f.update_layout(xaxis_title="iteración del boosting B", yaxis_title="NLL ZINB en validación (20 % final)", margin=dict(t=20))
    pesos = pd.Series(o["pesos_medios"]).sort_values(ascending=False)
    fw = pl.barras_h(pesos.index, pesos.values, "", "peso medio w_j", resaltar=lambda e: "propio" in e)
    dif = pd.DataFrame([
        {"método afín": "FFORMA (Montero-Manso et al., 2020)", "qué conserva A+B": "pesos por meta-características",
         "qué es nuevo": "pérdida en log1p para conteos, miembros ML + ZINB, OOF cronológico y calibración"},
        {"método afín": "Gradient boosting (Friedman, 2001)", "qué conserva A+B": "aditivo en árboles con pasos de Newton",
         "qué es nuevo": "pérdida ZINB con dos predictores (media y ceros) y dispersión re-estimada"},
        {"método afín": "Regresión ZINB / ETAS", "qué conserva A+B": "verosimilitud de conteos con exceso de ceros; excitación",
         "qué es nuevo": "estimada por boosting no paramétrico y combinada con ML"},
    ])
    return html.Div([
        html.H2("Modelo original A+B v2"),
        html.P("Combinador contextual (A) de pronosticadores que incluye un boosting propio de la verosimilitud binomial negativa inflada "
               "en ceros (B), con calibración final. Clase compatible con scikit-learn en src/original_model.py.", className="sub"),
        dbc.Row([
            dbc.Col([html.H3("Pseudocódigo"), html.Pre(PSEUDO, className="pseudo")], lg=7),
            dbc.Col([html.H3("Diferencias con la literatura"), _dt(dif, list(dif.columns), extra_cond=[{"if": {"column_id": "método afín"}, "fontWeight": "700"}]),
                     html.P("Tabla preliminar: la matriz de novedad completa está en el capítulo 4 del libro.", className="sub mt-1")], lg=5),
        ]),
        dbc.Row([
            dbc.Col([html.H3("Curva de convergencia de B"), dcc.Graph(figure=f, config=CFG, style={"height": "320px"})], lg=6),
            dbc.Col([html.H3("Peso medio de cada miembro en A"), dcc.Graph(figure=fw.update_layout(margin=dict(l=10, r=50, t=10, b=40)),
                                                                            config=CFG, style={"height": "320px"})], lg=6),
        ]),
        html.H3("Regla de combinación propia frente a reglas estándar (mismos miembros)"),
        _tabla_reglas(),
        html.H3("Pruebas de hipótesis contra los campeones (69 meses, 2021–2026)", className="mt-3"),
        html.P("Para cada prueba se muestra el campeón más difícil (el de mayor p ajustado) y el máximo de los p ajustados entre los 7.", className="sub"),
        _tabla_pruebas(),
    ], className="tarjeta")


@callback([Output(f"in-{c}", "value") for c in VARIABLES_BASE], Input("p-celda", "value"))
def llenar(celda):
    if celda is None:
        return [no_update] * len(VARIABLES_BASE)
    F = D.panel()
    r = F[(F.celda == celda) & F.y.isna()].iloc[0]
    return [None if pd.isna(r[c]) else round(float(r[c]), 4) for c in VARIABLES_BASE]


@callback([Output("p-alerta", "children"), Output("p-salida", "children")], Input("p-boton", "n_clicks"),
          [State("p-celda", "value"), State("m-sel", "value")] + [State(f"in-{c}", "value") for c in VARIABLES_BASE], prevent_initial_call=True)
def predecir(_, celda, m, *valores):
    if celda is None:
        return dbc.Alert("Elija primero una celda: sus variables de contexto (excitación e historia) se toman de ella.", color="warning"), no_update
    errores, avisos = [], []
    for e, v in zip(D.esquema_formulario(), valores):
        if v is None and not e["admite_faltante"]:
            errores.append(f"{e['columna']}: falta el valor o no es válido (límites [{e['min_valido']:g}, {e['max_valido']:g}])")
        elif v is not None and not (e["min_valido"] - 1e-9 <= v <= e["max_valido"] + 1e-9):
            errores.append(f"{e['columna']}: {v} está fuera de los límites válidos [{e['min_valido']:g}, {e['max_valido']:g}]")
        elif v is not None and e["entero"] and abs(v - round(v)) > 1e-9:
            errores.append(f"{e['columna']}: debe ser un número entero")
        elif v is not None and not (e["min"] - 1e-9 <= v <= e["max"] + 1e-9):
            avisos.append(f"{e['columna']} = {v:g} (observado en el entrenamiento: [{e['min']:g}, {e['max']:g}])")
    if errores:
        return dbc.Alert([html.B("Revise las entradas:"), html.Ul([html.Li(x) for x in errores])], color="danger"), no_update
    aviso = (dbc.Alert([html.B("Extrapolación: "), "estos valores superan lo observado en el entrenamiento; el pronóstico es menos confiable: ",
                        "; ".join(avisos)], color="warning") if avisos else None)
    try:
        F = D.panel()
        fila = F[(F.celda == celda) & F.y.isna()].iloc[[0]].copy()
        for c, v in zip(VARIABLES_BASE, valores):
            fila[c] = np.nan if v is None else float(v)
        fila["h0"], fila["h11"] = fila["n_1m"], fila["n_mismo_mes_anio_prev"]       # coherencia con la historia
        mod = A.modelo(m)
        X = fila[COLUMNAS] if m == AB else fila[VARIABLES_BASE]
        p = float(np.clip(mod.predict(X), 0, None)[0])
        lo, hi = D.intervalo(m, p)
        if m == AB:
            E = mod.explicar(X).iloc[0].drop("calibración (c0)")
            base_txt = (f"cada barra es el aporte c1·w·f del miembro; las barras y c0 = {P._n(mod.c0_, 3)} suman el pronóstico en escala "
                        f"log1p ({P._n(np.log1p(p), 2)})")
            nombres, val = E.index, E.values
            eje = "aporte al pronóstico log1p (c1·w·f)"
        else:
            from src.explicacion import shap_filas, valor_base
            S = shap_filas(mod, X, D.fondo_shap()[VARIABLES_BASE], max_evals=300, n_fondo=50)[0]
            orden = np.argsort(-np.abs(S))[:10]
            nombres, val = np.array(VARIABLES_BASE)[orden], S[orden]
            base_txt = f"valor base (pronóstico log1p medio del fondo) = {P._n(valor_base(mod, D.fondo_shap()[VARIABLES_BASE].iloc[:100]), 3)}"
            eje = "SHAP (escala log1p)"
        fig = go.Figure(go.Bar(y=list(nombres)[::-1], x=list(val)[::-1], orientation="h",
                               marker_color=[pl.NARANJA if v > 0 else pl.AZUL for v in list(val)[::-1]],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(xaxis_title=eje, height=360, margin=dict(l=10, r=10, t=10, b=40))
        salida = html.Div([
            html.Div([html.Span("Pronóstico: ", className="fw-bold"), html.Span(f"{P._n(p, 2)} sismos M ≥ 2,5", className="cifra"),
                      html.Div(f"Intervalo de predicción al 80 %: {P._n(lo[0], 1)} – {P._n(hi[0], 1)} sismos (cuantiles de los errores fuera de muestra "
                               f"2021–2026 del mismo tramo de pronóstico)", className="sub")]),
            html.H3("Explicación local", className="mt-2"),
            html.P((f"{base_txt}." if m == AB else f"Barras naranjas suben el pronóstico y azules lo bajan respecto del {base_txt}."), className="sub"),
            dcc.Graph(figure=fig, config=CFG),
        ])
        return aviso, salida
    except Exception as e:  # el error se muestra; la aplicación sigue funcionando
        return dbc.Alert(f"No se pudo calcular la predicción: {e}", color="danger"), no_update


def _bloque_diagnostico():
    return html.Div([
        html.H4("7. Diagnóstico del modelo seleccionado", className="mt-4"),
        _t("Reales, residuos y autocorrelación",
           "Predicciones fuera de muestra del origen móvil 2021–2026, en la escala log1p de la métrica. Cambie el modelo en el "
           "selector de modelo de la sección 6.", _g("d-diag", 660), html.Div(id="i-diag", className="insight")),
        html.H4("8. Interpretabilidad global del modelo seleccionado", className="mt-4"),
        _t("Importancia global",
           "Benchmarks: |SHAP| medio por variable (muestra de 200 celdas-mes del último año observado). A+B: |c1·w·f| medio de cada "
           "miembro, una descomposición aditiva exacta del pronóstico en log1p (alternativa a SHAP justificada en el capítulo 5).",
           _g("d-shap", 520), html.Div(id="i-shap", className="insight")),
    ])


@callback([Output("d-diag", "figure"), Output("i-diag", "children"), Output("d-shap", "figure"), Output("i-shap", "children")],
          Input("m-sel", "value"))
def diagnostico(m):
    if not m:
        return pl.vacia("Elija un modelo"), "", pl.vacia("Elija un modelo"), ""
    O = D.oos()
    if m not in O:
        return pl.vacia("Sin predicciones fuera de muestra para este modelo"), "", pl.vacia(), ""
    y, p, tm = O["y"], O[m], O["tm"]
    f = pl.diagnostico_regresion(y, p, tm)
    lp, ly = np.log1p(np.clip(p, 0, None)), np.log1p(y)
    r = ly - lp
    meses = np.unique(tm)
    pos = np.searchsorted(meses, tm)
    r_mes = np.bincount(pos, r) / np.bincount(pos)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = float(acorr_ljungbox(r_mes, lags=[12], return_df=True).lb_pvalue.iloc[0])
    alto = np.quantile(lp, 0.9)
    texto = [html.B("Lectura: "),
             f"RMSLE {P._n(float(np.sqrt(np.mean(r ** 2))))}; sesgo medio en log1p {P._n(float(r.mean()), 3)} "
             f"({'subestima' if r.mean() > 0 else 'sobrestima'} en promedio). En el 10 % de celdas-mes con pronóstico más alto el "
             f"residuo medio es {P._n(float(r[lp >= alto].mean()), 3)}. Los puntos alejados de la diagonal son enjambres y secuencias de réplicas "
             "que ningún modelo anticipa. "
             + (f"El residuo medio mensual no muestra autocorrelación significativa (Ljung-Box, 12 rezagos: p = {P._n(lb, 3)})."
                if lb >= 0.05 else f"El residuo medio mensual conserva autocorrelación (Ljung-Box, 12 rezagos: p = {P._n(lb, 3)}): "
                "hay estructura temporal que el modelo no capturó.")]
    sg = D.shap_global()
    if m not in sg:
        return f, texto, pl.vacia("Importancia global no disponible para este modelo"), ""
    v = pd.Series(sg[m]["valores"], dtype=float).sort_values(ascending=False).head(15)
    fs = pl.barras_h(v.index, v.values, "", "|SHAP| medio (log1p)" if "SHAP" in sg[m]["tipo"] else "|c1·w·f| medio (log1p)",
                     resaltar=(lambda e: "propio" in e) if m.startswith("A+B") else None, formato=".4f")
    fs.update_layout(margin=dict(l=10, r=60, t=10, b=50))
    pct = 100 * v.iloc[:3].sum() / pd.Series(sg[m]["valores"], dtype=float).sum()
    i_shap = [html.B("Lectura: "), f"las tres entradas con más peso ({', '.join(v.index[:3])}) explican el {P._n(pct, 0)} % de la "
              "importancia total. " + ("En A+B los miembros de árboles y el componente propio B concentran el aporte; las referencias "
                                       "simples pesan poco." if m.startswith("A+B") else
                                       "Dominan los conteos recientes de la celda: el modelo aprende sobre todo persistencia.")
              + " La importancia describe al modelo; no implica causalidad."]
    return f, texto, fs, i_shap
