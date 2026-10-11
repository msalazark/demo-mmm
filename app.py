"""Demo de Marketing Mix Modeling con Google Meridian — app Streamlit.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from mmm_core import modelo  # noqa: E402
from mmm_core.datos_sinteticos import (  # noqa: E402
    CANALES, CONTROLES, NOMBRES_CANALES, PARAMETROS_DEFECTO, ConfigSimulacion,
    ParametrosCanal, generar_datos,
)
from meridian.analysis import summarizer  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(BASE, "data", "datos_sinteticos.csv")
RUTA_VERDAD = os.path.join(BASE, "data", "verdad_terreno.json")
RUTA_MODELO = os.path.join(BASE, "modelos", "modelo_demo.pkl")
RUTA_GRAFICOS = os.path.join(BASE, "modelos", "graficos_demo.json")

COLORES = {"tv": "#1f4e79", "youtube": "#c0392b", "meta": "#2e86de",
           "google_search": "#27ae60", "tiktok": "#8e44ad"}

st.set_page_config(page_title="MMM con Google Meridian", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def soles(x: float) -> str:
    if abs(x) >= 1e6:
        return f"S/ {x / 1e6:,.1f} M"
    if abs(x) >= 1e3:
        return f"S/ {x / 1e3:,.0f} mil"
    return f"S/ {x:,.0f}"


@st.cache_resource(show_spinner="Cargando modelo pre-entrenado…")
def cargar_modelo_demo():
    return modelo.cargar_modelo(RUTA_MODELO)


@st.cache_data
def cargar_datos_demo():
    df = pd.read_csv(RUTA_CSV)
    with open(RUTA_VERDAD, encoding="utf-8") as f:
        verdad = json.load(f)
    return df, verdad


def estado():
    """Inicializa el estado con el modelo pre-entrenado."""
    if "mmm" not in st.session_state:
        df, verdad = cargar_datos_demo()
        st.session_state.df = df
        st.session_state.verdad = verdad
        st.session_state.mmm = None
        st.session_state.origen = "Sin modelo — entrena desde el paso 3"
        if os.path.exists(RUTA_MODELO):
            try:
                st.session_state.mmm = cargar_modelo_demo()
                st.session_state.origen = "Modelo pre-entrenado (datos sintéticos)"
            except Exception:  # pickle de otra versión de Meridian/JAX
                st.session_state.origen = ("El modelo pre-entrenado no es compatible con tu "
                                           "versión de Meridian: corre `python entrenar.py` "
                                           "o entrena desde el paso 3")
        st.session_state.version = 0
    return st.session_state


def clave_modelo() -> int:
    """Identificador de la versión del modelo, para cachear cálculos pesados."""
    return st.session_state.version


@st.cache_data(show_spinner="Calculando ROI…")
def roi_cache(_mmm, version, _verdad):
    return modelo.tabla_roi(_mmm, _verdad)


@st.cache_data(show_spinner="Calculando métricas por canal…")
def resumen_cache(_mmm, version):
    return modelo.tabla_resumen(_mmm)


@st.cache_data(show_spinner="Calculando diagnósticos…")
def diagnostico_cache(_mmm, version):
    return modelo.ajuste_y_diagnostico(_mmm)


@st.cache_resource(show_spinner="Generando gráficos de Meridian (≈1 min la primera vez)…")
def graficos_cache(_mmm, version):
    """Gráficos nativos de Meridian. Para el modelo pre-entrenado se leen del JSON
    pre-calculado; para un modelo nuevo se generan una vez y quedan en caché."""
    if version == 0 and os.path.exists(RUTA_GRAFICOS):
        with open(RUTA_GRAFICOS, encoding="utf-8") as f:
            return json.load(f)
    return modelo.graficos_meridian(_mmm)


def grafico(spec: dict) -> None:
    # Los gráficos con paneles (facet/concat) mantienen su ancho original
    compuesto = any(k in spec for k in ("facet", "hconcat", "vconcat", "concat"))
    st.vega_lite_chart(spec, width="content" if compuesto else "stretch")


@st.cache_resource(show_spinner="Optimizando presupuesto (puede tardar ~1 min)…")
def optimizar_cache(_mmm, version, presupuesto, fijo, roi_obj, bajo, alto):
    return modelo.optimizar_presupuesto(_mmm, presupuesto, fijo, roi_obj, bajo, alto)


s = estado()

# ----------------------------------------------------------------------------
# Barra lateral: datos, priors y entrenamiento
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(f"**Modelo activo:** {s.origen}")

    with st.expander("1 · Datos", expanded=False):
        fuente = st.radio("Fuente", ["Sintéticos (laboratorio)", "Subir CSV propio"],
                          label_visibility="collapsed")
        if fuente == "Sintéticos (laboratorio)":
            st.caption("Define el **ROI real** de cada canal. Luego entrena y "
                       "comprueba si Meridian lo descubre.")
            rois = {}
            for c in CANALES:
                rois[c] = st.slider(NOMBRES_CANALES[c], 0.2, 5.0,
                                    float(PARAMETROS_DEFECTO[c].roi), 0.1, key=f"roi_{c}")
            semilla = st.number_input("Semilla", 1, 99999, 2026)
            if st.button("Generar datos", width="stretch"):
                cfg = ConfigSimulacion(semilla=int(semilla))
                for c in CANALES:
                    cfg.canales[c] = ParametrosCanal(**{**PARAMETROS_DEFECTO[c].__dict__,
                                                        "roi": rois[c]})
                s.df, s.verdad = generar_datos(cfg)
                s.mmm = None
                s.origen = "Datos nuevos — falta entrenar"
                st.success("Datos generados. Ahora entrena el modelo (paso 3).")
        else:
            st.caption("Formato largo geo × semana con columnas: `geo`, `time`, "
                       "`population`, `ventas`, `indice_precio`, `evento_comercial`, "
                       "`<canal>_impresiones`, `<canal>_inversion` para "
                       + ", ".join(f"`{c}`" for c in CANALES) + ".")
            archivo = st.file_uploader("CSV", type="csv")
            if archivo is not None:
                nuevo = pd.read_csv(archivo)
                requeridas = (["geo", "time", "population", "ventas"] + CONTROLES
                              + [f"{c}_{t}" for c in CANALES for t in ("impresiones", "inversion")])
                faltan = [c for c in requeridas if c not in nuevo.columns]
                if faltan:
                    st.error(f"Faltan columnas: {faltan}")
                elif st.button("Usar este CSV", width="stretch"):
                    s.df, s.verdad = nuevo, None
                    s.mmm = None
                    s.origen = "CSV propio — falta entrenar"

    with st.expander("2 · Priors de ROI", expanded=False):
        usar_priors = st.toggle("Usar priors propios", value=False,
                                help="Por defecto Meridian usa LogNormal(0.2, 0.9) para todos "
                                     "los canales (ROI mediano ≈ 1.2). Con priors propios "
                                     "incorporas benchmarks o resultados de experimentos.")
        priors = None
        if usar_priors:
            priors = {}
            for c in CANALES:
                col1, col2 = st.columns(2)
                m = col1.number_input(f"{NOMBRES_CANALES[c]} media", 0.1, 10.0, 2.0, 0.1,
                                      key=f"pm_{c}")
                d = col2.number_input("desv.", 0.1, 10.0, 1.0, 0.1, key=f"pd_{c}")
                priors[c] = (m, d)

    with st.expander("3 · Entrenar", expanded=s.mmm is None):
        preset_nombre = st.selectbox("Muestreo MCMC", list(modelo.PRESETS))
        preset = modelo.PRESETS[preset_nombre]
        st.caption(f"{preset.descripcion}  \n{preset.n_chains} cadenas · "
                   f"{preset.n_adapt} adapt · {preset.n_burnin} burn-in · {preset.n_keep} draws")
        if st.button("🚀 Entrenar modelo", type="primary", width="stretch"):
            with st.status("Entrenando Meridian…", expanded=True) as status:
                st.write("Construyendo InputData…")
                input_data = modelo.construir_input_data(s.df)
                spec = modelo.construir_spec(s.df["time"].nunique(), priors)
                st.write("Muestreando prior y posterior (NUTS)… esto toma unos minutos.")
                s.mmm, seg = modelo.ajustar_modelo(input_data, spec, preset)
                s.version += 1
                s.origen = f"Entrenado en esta sesión ({preset_nombre}, {seg / 60:.1f} min)"
                status.update(label=f"Modelo listo en {seg / 60:.1f} min", state="complete")
            st.rerun()

    if s.mmm is not None:
        with st.expander("Exportar", expanded=False):
            if st.button("Generar reporte HTML de Meridian", width="stretch"):
                with st.spinner("Generando reporte…"):
                    carpeta = tempfile.mkdtemp()
                    summarizer.Summarizer(s.mmm).output_model_results_summary(
                        "reporte_meridian.html", carpeta)
                    with open(os.path.join(carpeta, "reporte_meridian.html"), "rb") as f:
                        s.reporte = f.read()
            if s.get("reporte"):
                st.download_button("Descargar reporte", s.reporte, "reporte_meridian.html",
                                   "text/html", width="stretch")

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("Marketing Mix Modeling con Google Meridian")
st.markdown(
    "**Tiendas Pacífico** *(retailer ficticio)* · 10 regiones del Perú · 104 semanas · "
    "5 canales pagados. Los datos son sintéticos: conocemos el ROI real de cada canal "
    "y podemos comprobar si el modelo bayesiano lo recupera."
)

df = s.df
mmm = s.mmm
verdad = s.verdad

tabs = st.tabs(["📊 Datos", "✅ Ajuste del modelo", "💰 ROI y contribución",
                "📈 Curvas de respuesta", "🎯 Optimizador", "📘 Cómo funciona"])

# ----------------------------------------------------------------------------
# Tab 1: Datos
# ----------------------------------------------------------------------------
with tabs[0]:
    nacional = df.groupby("time", as_index=False).sum(numeric_only=True)
    nacional["time"] = pd.to_datetime(nacional["time"])
    inv_cols = [f"{c}_inversion" for c in CANALES]
    inversion_total = nacional[inv_cols].sum().sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ventas totales", soles(nacional["ventas"].sum()))
    k2.metric("Inversión en medios", soles(inversion_total))
    k3.metric("Semanas", df["time"].nunique())
    k4.metric("Regiones", df["geo"].nunique())

    fig = go.Figure()
    semanas_evento = nacional.loc[nacional["evento_comercial"] > 0, "time"]
    for f in semanas_evento:
        fig.add_vrect(x0=f - pd.Timedelta(days=3), x1=f + pd.Timedelta(days=4),
                      fillcolor="#f1c40f", opacity=0.18, line_width=0, layer="below")
    fig.add_scatter(x=nacional["time"], y=nacional["ventas"], name="Ventas",
                    line=dict(color="#2c3e50", width=2.5))
    fig.update_layout(title="Ventas nacionales por semana (franjas amarillas = eventos "
                            "comerciales: Cyber Wow, Día de la Madre, Fiestas Patrias, "
                            "Black Friday, Navidad)",
                      height=360, yaxis_title="S/", margin=dict(t=60, b=10),
                      showlegend=False, title_font_size=15)
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for c in CANALES:
        fig.add_scatter(x=nacional["time"], y=nacional[f"{c}_inversion"], name=NOMBRES_CANALES[c],
                        stackgroup="uno", line=dict(width=0.5, color=COLORES[c]))
    fig.update_layout(title="Inversión semanal por canal", height=360, yaxis_title="S/",
                      margin=dict(t=60, b=10), title_font_size=15,
                      legend=dict(orientation="h", y=-0.12, x=0))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Por región")
        por_geo = df.groupby("geo").agg(poblacion=("population", "first"),
                                        ventas=("ventas", "sum"))
        por_geo["inversion"] = df.groupby("geo")[inv_cols].sum().sum(axis=1)
        por_geo = por_geo.sort_values("ventas", ascending=False)
        st.dataframe(por_geo.style.format({"poblacion": "{:,.0f}", "ventas": "S/ {:,.0f}",
                                           "inversion": "S/ {:,.0f}"}), width="stretch")
    with c2:
        st.subheader("Muestra del dataset")
        st.dataframe(df.head(50), width="stretch", height=380)
        st.download_button("Descargar CSV completo", df.to_csv(index=False).encode(),
                           "datos_mmm.csv", "text/csv")

if mmm is None:
    for t in tabs[1:5]:
        with t:
            st.info("Entrena el modelo desde la barra lateral (paso 3) para ver esta sección.")
else:
    version = clave_modelo()
    g = graficos_cache(mmm, version)

    # ------------------------------------------------------------------------
    # Tab 2: Ajuste
    # ------------------------------------------------------------------------
    with tabs[1]:
        diag = diagnostico_cache(mmm, version)
        prec = diag["precision"]
        nac = prec[prec["geo_granularity"] == "national"].set_index("metric")["value"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("R² nacional", f"{nac.get('R_Squared', np.nan):.3f}")
        k2.metric("MAPE nacional", f"{nac.get('MAPE', np.nan) * 100:.1f} %")
        k3.metric("wMAPE nacional", f"{nac.get('wMAPE', np.nan) * 100:.1f} %")
        max_rhat = float(diag["rhat"]["max_r_hat"].max())
        k4.metric("R-hat máximo", f"{max_rhat:.2f}",
                  delta="convergió" if max_rhat < 1.2 else "revisar", delta_color="normal"
                  if max_rhat < 1.2 else "inverse")

        st.subheader("Ventas reales vs. esperadas por el modelo")
        st.caption("La línea base es lo que se habría vendido sin publicidad. "
                   "La brecha entre esperado y línea base es el aporte de los medios.")
        grafico(g["ajuste"])

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Convergencia (R-hat)")
            st.caption("R-hat < 1.2 indica que las cadenas MCMC coinciden. "
                       "Si algún parámetro supera ese umbral, aumenta el muestreo.")
            grafico(g["rhat"])
        with c2:
            st.subheader("Precisión por nivel")
            tabla = prec.pivot_table(index="metric", columns="geo_granularity", values="value")
            st.dataframe(tabla.style.format("{:.3f}"), width="stretch")

    # ------------------------------------------------------------------------
    # Tab 3: ROI
    # ------------------------------------------------------------------------
    with tabs[2]:
        roi = roi_cache(mmm, version, verdad)
        st.subheader("¿Recuperó Meridian el ROI real?" if verdad else "ROI por canal")

        if verdad and "roi_real" in roi:
            dentro = int(roi["real_dentro_ci"].sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Canales con ROI real dentro del IC 90 %", f"{dentro} de {len(roi)}")
            k2.metric("Error absoluto medio", f"{roi['error_%'].abs().mean():.0f} %")
            mejor_est = roi.loc[roi["roi_estimado"].idxmax(), "nombre"]
            mejor_real = roi.loc[roi["roi_real"].idxmax(), "nombre"]
            k3.metric("Canal más rentable según el modelo", mejor_est,
                      f"real: {mejor_real}", delta_color="off")

        fig = go.Figure()
        fig.add_bar(
            x=roi["nombre"], y=roi["roi_estimado"], name="ROI estimado (media posterior)",
            marker_color=[COLORES.get(c, "#7f8c8d") for c in roi["canal"]], opacity=0.85,
            error_y=dict(type="data", symmetric=False,
                         array=roi["ci_alto"] - roi["roi_estimado"],
                         arrayminus=roi["roi_estimado"] - roi["ci_bajo"],
                         color="#555", thickness=1.5, width=8),
        )
        if verdad and "roi_real" in roi:
            fig.add_scatter(x=roi["nombre"], y=roi["roi_real"], mode="markers",
                            name="ROI real (simulación)",
                            marker=dict(symbol="diamond", size=16, color="#f1c40f",
                                        line=dict(color="#2c3e50", width=2)))
        fig.add_hline(y=1, line_dash="dot", line_color="#999",
                      annotation_text="Punto de equilibrio (ROI = 1)",
                      annotation_position="top left")
        fig.update_layout(height=430, yaxis_title="Ventas incrementales por S/ 1 invertido",
                          margin=dict(t=20, b=10), legend=dict(orientation="h", y=-0.12, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Barras = media posterior · bigotes = intervalo creíble 90 % · "
                   "rombos = ROI real con el que se generaron los datos.")

        roi_vista = roi.drop(columns=["canal"]).rename(columns={
            "nombre": "Canal", "roi_estimado": "ROI estimado", "ci_bajo": "IC 90 % bajo",
            "ci_alto": "IC 90 % alto", "roi_real": "ROI real", "error_%": "Error",
            "real_dentro_ci": "Real dentro del IC"})
        fmt = {"ROI estimado": "{:.2f}", "IC 90 % bajo": "{:.2f}", "IC 90 % alto": "{:.2f}",
               "ROI real": "{:.2f}", "Error": "{:+.0f} %"}
        st.dataframe(roi_vista.style.format(fmt), width="stretch", hide_index=True)

        st.subheader("Métricas por canal")
        res = resumen_cache(mmm, version)
        st.dataframe(
            res.drop(columns=["canal"]).rename(columns={
                "nombre": "Canal", "inversion": "Inversión", "pct_inversion": "% inversión",
                "ventas_incrementales": "Ventas incrementales",
                "pct_contribucion": "% de las ventas", "roi": "ROI", "mroi": "ROI marginal",
                "cpik": "Costo por S/ incremental"}).style.format({
                "Inversión": "S/ {:,.0f}", "% inversión": "{:.1f} %",
                "Ventas incrementales": "S/ {:,.0f}", "% de las ventas": "{:.1f} %",
                "ROI": "{:.2f}", "ROI marginal": "{:.2f}", "Costo por S/ incremental": "{:.2f}"}),
            width="stretch", hide_index=True)
        st.caption("**ROI marginal**: lo que devuelve el *siguiente* sol invertido. "
                   "Si es < 1, ese canal ya está saturado aunque su ROI promedio sea alto.")

        c1, c2 = st.columns(2)
        with c1:
            grafico(g["cascada"])
        with c2:
            grafico(g["inversion_vs_contribucion"])
        grafico(g["roi_vs_mroi"])

    # ------------------------------------------------------------------------
    # Tab 4: Curvas
    # ------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Curvas de respuesta")
        st.caption("Ventas incrementales esperadas según el nivel de inversión "
                   "(el punto marca la inversión histórica). La curvatura muestra la saturación.")
        grafico(g["respuesta"])
        st.subheader("Adstock (efecto arrastre)")
        st.caption("Qué fracción del efecto de una semana sigue activa en las siguientes "
                   "(azul = posterior, rojo = prior).")
        grafico(g["adstock"])
        if verdad:
            st.caption("Decaimiento real usado en la simulación: " + " · ".join(
                f"{NOMBRES_CANALES[c]} α={v['alpha_adstock']}"
                for c, v in verdad["canales"].items()))
        st.subheader("Saturación (curva Hill)")
        st.caption("Respuesta por persona según la exposición (azul = posterior, rojo = prior). "
                   "El histograma gris muestra los niveles de exposición observados.")
        for spec_hill in g["hill"]:
            grafico(spec_hill)

    # ------------------------------------------------------------------------
    # Tab 5: Optimizador
    # ------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("¿Cómo redistribuir el presupuesto?")
        historico = float(df[[f"{c}_inversion" for c in CANALES]].sum().sum())
        c1, c2, c3 = st.columns(3)
        with c1:
            modo = st.radio("Escenario", ["Presupuesto fijo", "Presupuesto flexible (ROI objetivo)"])
        with c2:
            if modo == "Presupuesto fijo":
                pct = st.slider("Presupuesto vs. histórico", 50, 150, 100, 5, format="%d %%")
                presupuesto = historico * pct / 100
                st.caption(f"Presupuesto: **{soles(presupuesto)}** (histórico {soles(historico)})")
                roi_obj = None
            else:
                roi_obj = st.slider("ROI objetivo mínimo", 1.0, 4.0, 1.5, 0.1)
                presupuesto = None
        with c3:
            lim = st.slider("Cambio máximo por canal", 10, 100, 30, 10, format="± %d %%")

        st.caption("Usa la curva de respuesta de cada canal para encontrar la mezcla que "
                   "maximiza las ventas incrementales. Tarda ~30–60 s en CPU.")
        if st.button("Optimizar", type="primary"):
            s.opt_params = (presupuesto, modo == "Presupuesto fijo", roi_obj, lim / 100, lim / 100)

        if s.get("opt_params"):
            res_opt = optimizar_cache(mmm, version, *s.opt_params)
            tabla = modelo.tabla_optimizacion(res_opt)
            inc_act = tabla["ventas_inc_actual"].sum()
            inc_opt = tabla["ventas_inc_optima"].sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Inversión optimizada", soles(tabla["inversion_optima"].sum()),
                      f"{(tabla['inversion_optima'].sum() / tabla['inversion_actual'].sum() - 1) * 100:+.1f} %")
            k2.metric("Ventas incrementales", soles(inc_opt), f"{(inc_opt / inc_act - 1) * 100:+.1f} %")
            k3.metric("ROI del mix", f"{inc_opt / tabla['inversion_optima'].sum():.2f}",
                      f"{inc_opt / tabla['inversion_optima'].sum() - inc_act / tabla['inversion_actual'].sum():+.2f}")

            fig = go.Figure()
            fig.add_bar(x=tabla["nombre"], y=tabla["inversion_actual"], name="Actual",
                        marker_color="#bdc3c7")
            fig.add_bar(x=tabla["nombre"], y=tabla["inversion_optima"], name="Optimizado",
                        marker_color="#2e86de",
                        text=[f"{v:+.0f} %" for v in tabla["cambio_inversion_%"]],
                        textposition="outside")
            fig.update_layout(barmode="group", height=380, yaxis_title="S/",
                              title="Inversión por canal: actual vs. optimizada",
                              margin=dict(t=60, b=10), title_font_size=15,
                              legend=dict(orientation="h", y=-0.12, x=0))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                tabla.drop(columns=["canal"]).rename(columns={
                    "nombre": "Canal", "inversion_actual": "Inversión actual",
                    "inversion_optima": "Inversión óptima",
                    "ventas_inc_actual": "Ventas inc. actuales",
                    "ventas_inc_optima": "Ventas inc. óptimas",
                    "cambio_inversion_%": "Cambio"}).style.format({
                    "Inversión actual": "S/ {:,.0f}", "Inversión óptima": "S/ {:,.0f}",
                    "Ventas inc. actuales": "S/ {:,.0f}", "Ventas inc. óptimas": "S/ {:,.0f}",
                    "Cambio": "{:+.0f} %"}),
                width="stretch", hide_index=True)
            st.altair_chart(res_opt.plot_incremental_outcome_delta(), width="stretch")
            st.subheader("Curvas de respuesta con la inversión óptima")
            st.altair_chart(res_opt.plot_response_curves(), width="content")
        else:
            st.info("Elige un escenario y presiona **Optimizar**.")

# ----------------------------------------------------------------------------
# Tab 6: Cómo funciona
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown("""
### Qué hace Meridian
Meridian es el framework open source de Google para **Marketing Mix Modeling bayesiano**.
Estima cuánto aporta cada canal a las ventas usando solo datos agregados (sin cookies ni
datos a nivel usuario), lo que lo hace robusto a las restricciones de privacidad.

**Ecuación simplificada** (por región *g* y semana *t*):

`ventas = línea_base(t, g) + Σ canales β · Hill(Adstock(medios)) + Σ controles γ · z + ruido`

| Componente | Qué captura | En esta demo |
|---|---|---|
| **Adstock** | El efecto de la publicidad se arrastra varias semanas | TV dura mucho (α = 0.7); Search casi nada (α = 0.1) |
| **Hill** | Rendimientos decrecientes: cada sol extra rinde menos | Search está más saturado que el resto |
| **Línea base con *knots*** | Tendencia y estacionalidad sin medios | Un nudo cada 8 semanas |
| **Controles** | Factores no publicitarios | Índice de precio y eventos (Cyber Wow, Navidad, Día de la Madre…) |
| **Jerarquía geo** | Cada región tiene su propio efecto, ligado a un promedio nacional | 10 regiones con población real aproximada |

### Por qué bayesiano
- **Priors de ROI**: se puede incorporar conocimiento previo (benchmarks, geo-experimentos,
  conversion lift) directamente sobre el ROI, que es la métrica que entiende el negocio.
- **Incertidumbre explícita**: cada resultado viene con un intervalo creíble. Un canal con
  intervalo muy ancho pide un experimento antes de moverle presupuesto.
- **MCMC (NUTS)**: el modelo se ajusta muestreando miles de combinaciones de parámetros
  plausibles. Por eso entrenar toma minutos y la convergencia se revisa con R-hat.

### Qué observar en la demo
1. **Google Search** suele quedar subestimado: su inversión sigue a la demanda y casi no
   varía, así que el modelo tiene poca señal para separarlo de la línea base. Es un caso
   real de *endogeneidad* y la razón por la que se recomiendan priors calibrados con experimentos.
2. **TikTok** empezó a mitad del periodo y con poca inversión: su intervalo es ancho.
3. Activa **priors propios** y vuelve a entrenar para ver cómo cambian las estimaciones.
4. En el **Laboratorio de datos** cambia los ROI reales y verifica si el modelo los sigue.

### Datos que necesita un MMM real
- 2 a 3 años de datos semanales (mínimo ~80 semanas), idealmente por región.
- Inversión **y** exposición (impresiones, GRPs, clics) por canal.
- KPI (ventas, conversiones) y variables de control (precio, promociones, estacionalidad).

Documentación oficial: [developers.google.com/meridian](https://developers.google.com/meridian)
""")
