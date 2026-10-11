"""Construye notebooks/mmm_meridian_demo.ipynb (se ejecuta una vez; el .ipynb es el entregable)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# Marketing Mix Modeling con Google Meridian — demo paso a paso

**Caso:** *Tiendas Pacífico* (retailer ficticio) vende en 10 regiones del Perú y pauta en
TV, YouTube, Meta, Google Search y TikTok. Tenemos 104 semanas de datos **sintéticos**.

**¿Por qué datos sintéticos?** Porque nosotros fijamos el ROI real de cada canal al fabricar
los datos. Así podemos hacer algo imposible con datos reales: **verificar si el modelo
descubre la verdad**.

**Ruta del notebook**
1. Generar y explorar los datos
2. Cargar los datos en Meridian (`InputData`)
3. Especificar el modelo y los *priors* (`ModelSpec`)
4. Ajustar con MCMC (NUTS)
5. Diagnosticar: convergencia y ajuste
6. Resultados: ROI vs. verdad, contribución, curvas de respuesta, adstock
7. Optimizar el presupuesto
8. Guardar el modelo y exportar el reporte
""")

code("""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Permite importar mmm_core desde la carpeta del proyecto
RAIZ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, RAIZ)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import meridian
from meridian import constants
from meridian.data import data_frame_input_data_builder as dfb
from meridian.model import model, spec, prior_distribution
from meridian.analysis import analyzer, visualizer, optimizer, summarizer

from mmm_core.datos_sinteticos import generar_datos, CANALES, NOMBRES_CANALES
from mmm_core import modelo as utilidades  # funciones auxiliares (tablas, símbolo S/)

print("Meridian", meridian.__version__)
plt.rcParams.update({"figure.figsize": (12, 4), "axes.spines.top": False, "axes.spines.right": False})
""")

md("""
## 1. Datos sintéticos

El generador simula, para cada región *g* y semana *t*:

$$\\text{ventas}_{g,t} = \\text{base}_{g,t} \\cdot \\text{tendencia} \\cdot \\text{estacionalidad} \\cdot \\text{precio}^{-1.2} \\cdot (1 + 0.35\\,\\text{evento}) + \\sum_m \\beta_m \\cdot \\text{pob}_g \\cdot \\text{Hill}\\big(\\text{Adstock}(x_{m,g,t})\\big) + \\varepsilon$$

donde $x$ son las impresiones per cápita. Cada $\\beta_m$ se calibra para que el ROI real
sea exactamente el que fijamos.
""")

code("""
df, verdad = generar_datos()          # semilla fija = resultados reproducibles
print(df.shape)
df.head()
""")

code("""
pd.DataFrame(verdad["canales"]).T[["roi_real", "alpha_adstock", "inversion_total", "contribucion_total", "share_ventas"]] \\
    .rename(index=NOMBRES_CANALES).style.format({"roi_real": "{:.2f}", "alpha_adstock": "{:.2f}",
        "inversion_total": "S/ {:,.0f}", "contribucion_total": "S/ {:,.0f}", "share_ventas": "{:.1%}"})
""")

md("""
> **La verdad de terreno.** Search es el canal más rentable (ROI 3.2) y TikTok el único que
> pierde dinero (ROI 0.9). TV tiene el arrastre más largo (α = 0.7). En total, los medios
> explican ~13 % de las ventas: el resto es línea base. Es una proporción realista y hace
> que el problema **no** sea fácil.
""")

code("""
nacional = df.groupby("time").sum(numeric_only=True)
nacional.index = pd.to_datetime(nacional.index)

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(nacional.index, nacional["ventas"] / 1e6, color="#2c3e50")
for f in nacional.index[nacional["evento_comercial"] > 0]:
    ax[0].axvspan(f - pd.Timedelta(days=3), f + pd.Timedelta(days=4), color="#f1c40f", alpha=0.25)
ax[0].set_title("Ventas nacionales semanales (S/ millones) — franjas = eventos comerciales")
ax[1].stackplot(nacional.index, *[nacional[f"{c}_inversion"] / 1e3 for c in CANALES],
                labels=[NOMBRES_CANALES[c] for c in CANALES])
ax[1].set_title("Inversión semanal por canal (S/ miles)")
ax[1].legend(loc="upper left", ncol=5, frameon=False)
plt.tight_layout()
""")

md("""
Fíjate en los patrones de inversión: **TV va en *flights*** (4 semanas sí, 3 no), **TikTok
arranca a mitad del periodo** y **Search sigue a la demanda** (sube en eventos). Este último
detalle creará un problema de identificación que veremos en los resultados.

## 2. Cargar los datos en Meridian

Meridian recibe un objeto `InputData`. El `DataFrameInputDataBuilder` lo arma desde un
DataFrame largo (una fila por región-semana). Usamos `kpi_type="revenue"` porque el KPI ya
está en soles; si fueran conversiones, agregaríamos `revenue_per_kpi`.
""")

code("""
builder = (
    dfb.DataFrameInputDataBuilder(kpi_type=constants.REVENUE)
    .with_kpi(df, kpi_col="ventas")
    .with_population(df, population_col="population")
    .with_controls(df, control_cols=["indice_precio", "evento_comercial"])
    .with_media(
        df,
        media_cols=[f"{c}_impresiones" for c in CANALES],      # exposición
        media_spend_cols=[f"{c}_inversion" for c in CANALES],  # costo
        media_channels=CANALES,
    )
)
builder.currency_code = "PEN"
input_data = builder.build()
input_data
""")

md("""
## 3. Especificación del modelo y *priors*

Decisiones clave del `ModelSpec`:

| Parámetro | Valor | Por qué |
|---|---|---|
| `media_prior_type="roi"` | prior sobre el **ROI** | Es la métrica que el negocio entiende; permite usar benchmarks o experimentos |
| `knots` | 13 (uno cada 8 semanas) | Línea base flexible pero sin absorber el efecto de los medios |
| `max_lag` | 8 semanas | Ventana máxima de adstock |

El prior por defecto del ROI es `LogNormal(0.2, 0.9)`: mediana ≈ 1.2, y 90 % de la masa
entre ~0.3 y ~5.4. Es deliberadamente vago.

**Ejercicio:** cambia `PRIORS_ROI` por un dict `{canal: (media, desviación)}` (por ejemplo,
resultados de un geo-lift para Search) y compara.
""")

code("""
PRIORS_ROI = None   # ej.: {"tv": (1.5, 0.7), "youtube": (1.2, 0.6), "meta": (2, 1), "google_search": (3, 0.8), "tiktok": (1, 0.6)}

if PRIORS_ROI:
    from meridian import backend
    d = prior_distribution.lognormal_dist_from_mean_std([PRIORS_ROI[c][0] for c in CANALES],
                                                        [PRIORS_ROI[c][1] for c in CANALES])
    prior = prior_distribution.PriorDistribution(
        roi_m=backend.tfd.LogNormal(d.loc, d.scale, name=constants.ROI_M))
else:
    prior = prior_distribution.PriorDistribution()

model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
    knots=13,
    max_lag=8,
)
mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
""")

md("""
## 4. Ajuste con MCMC

Meridian usa **NUTS** (No-U-Turn Sampler). Primero muestrea del prior (útil para comparar
prior vs posterior) y luego del posterior.

- `n_chains`: cadenas independientes (permiten medir convergencia).
- `n_adapt` + `n_burnin`: iteraciones de calentamiento que se descartan.
- `n_keep`: muestras que se conservan por cadena.

Para la clase usamos un preset rápido (~2–3 min en CPU). Para resultados finales Google
recomienda `n_chains=4, n_adapt=2000, n_burnin=500, n_keep=1000`.
""")

code("""
%%time
PRESET = utilidades.PRESETS["Rápido (demo en vivo)"]   # o "Estándar" / "Producción"

mmm.sample_prior(300, seed=42)
mmm.sample_posterior(n_chains=PRESET.n_chains, n_adapt=PRESET.n_adapt,
                     n_burnin=PRESET.n_burnin, n_keep=PRESET.n_keep, seed=42)
""")

md("""
## 5. Diagnóstico

### 5.1 Convergencia: R-hat
R-hat compara la varianza entre cadenas con la varianza dentro de cada cadena. **Si R-hat < 1.2**
(idealmente < 1.1) las cadenas coinciden y podemos confiar en el posterior.
""")

code("""
an = analyzer.Analyzer(mmm)
an.rhat_summary()[["param", "n_params", "avg_r_hat", "max_r_hat", "percent_bad_r_hat"]].round(3)
""")

code("""
visualizer.ModelDiagnostics(mmm).plot_rhat_boxplot()
""")

md("""
### 5.2 Ajuste: ventas reales vs. esperadas
La **línea base** es lo que se habría vendido sin publicidad. La distancia entre lo esperado
y la línea base es el aporte de los medios.
""")

code("""
visualizer.ModelFit(mmm).plot_model_fit()
""")

code("""
an.predictive_accuracy().to_dataframe().reset_index() \\
  .pivot_table(index="metric", columns="geo_granularity", values="value").round(3)
""")

md("""
> Un R² alto **no** garantiza buenos ROI: un modelo puede predecir muy bien las ventas
> atribuyendo efectos al canal equivocado. Por eso la validación causal (experimentos)
> sigue siendo necesaria.

## 6. Resultados

### 6.1 ¿Recuperó el modelo el ROI real?
""")

code("""
tabla = utilidades.tabla_roi(mmm, verdad)
tabla.drop(columns="canal").round(2)
""")

code("""
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(tabla))
ax.bar(x, tabla["roi_estimado"], color="#2e86de", alpha=0.75, label="ROI estimado (media posterior)")
ax.errorbar(x, tabla["roi_estimado"],
            yerr=[tabla["roi_estimado"] - tabla["ci_bajo"], tabla["ci_alto"] - tabla["roi_estimado"]],
            fmt="none", color="#333", capsize=6, label="IC 90 %")
ax.scatter(x, tabla["roi_real"], marker="D", s=110, color="#f1c40f", edgecolor="#2c3e50",
           zorder=3, label="ROI real")
ax.axhline(1, ls=":", color="#888")
ax.set_xticks(x, tabla["nombre"])
ax.set_ylabel("S/ incrementales por S/ invertido")
ax.set_title("ROI estimado por Meridian vs. ROI real de la simulación")
ax.legend(frameon=False)
plt.tight_layout()
""")

md("""
**Qué observar**

- Si el ROI real (rombo) cae dentro del intervalo, el modelo es **consistente** con la verdad.
- **Google Search** tiende a quedar subestimado: su inversión sigue a la demanda y varía poco,
  así que el modelo atribuye parte de su efecto a la línea base. Es *endogeneidad*, el problema
  más común en MMM con canales de búsqueda.
- **TikTok** tiene el intervalo más ancho: empezó tarde y con poca inversión, hay poca señal.
- La incertidumbre es el mensaje: un canal con intervalo que cruza ROI = 1 pide un
  **experimento** antes de mover presupuesto.

### 6.2 Resumen por canal
""")

code("""
media_summary = visualizer.MediaSummary(mmm)
media_summary.summary_table()
""")

code("""
media_summary.plot_contribution_waterfall_chart()
""")

code("""
media_summary.plot_roi_vs_mroi()
""")

md("""
**ROI marginal (mROI)**: lo que rinde el *siguiente* sol. Un canal con ROI alto pero mROI < 1
ya está saturado: conviene mover presupuesto hacia canales con mROI alto.

### 6.3 Curvas de respuesta, adstock y saturación
""")

code("""
media_effects = visualizer.MediaEffects(mmm)
media_effects.plot_response_curves(plot_separately=False)
""")

code("""
media_effects.plot_adstock_decay()
""")

code("""
hill = media_effects.plot_hill_curves()
list(hill.values())[0] if isinstance(hill, dict) else hill
""")

md("""
## 7. Optimización del presupuesto

El `BudgetOptimizer` usa las curvas de respuesta para encontrar la mezcla que maximiza las
ventas incrementales, con restricciones de cuánto puede moverse cada canal.

### 7.1 Presupuesto fijo (mismo total, distinta mezcla, máximo ±30 % por canal)
""")

code("""
%%time
opt = optimizer.BudgetOptimizer(mmm)
res_fijo = opt.optimize(fixed_budget=True, spend_constraint_lower=0.3, spend_constraint_upper=0.3)
utilidades.tabla_optimizacion(res_fijo).drop(columns="canal").round(0)
""")

code("""
res_fijo.plot_spend_delta()
""")

code("""
res_fijo.plot_incremental_outcome_delta()
""")

md("""
### 7.2 Presupuesto flexible: ¿cuánto invertir manteniendo un ROI mínimo de 1.5?
""")

code("""
res_flex = opt.optimize(fixed_budget=False, target_roi=1.5,
                        spend_constraint_lower=0.5, spend_constraint_upper=0.5)
t = utilidades.tabla_optimizacion(res_flex)
print(f"Inversión total: S/ {t.inversion_actual.sum():,.0f} -> S/ {t.inversion_optima.sum():,.0f}")
t.drop(columns="canal").round(0)
""")

md("""
## 8. Guardar el modelo y exportar el reporte HTML de Meridian
""")

code("""
os.makedirs(os.path.join(RAIZ, "modelos"), exist_ok=True)
model.save_mmm(mmm, os.path.join(RAIZ, "modelos", "modelo_notebook.pkl"))

summarizer.Summarizer(mmm).output_model_results_summary(
    "reporte_meridian.html", os.path.join(RAIZ, "modelos"))
res_fijo.output_optimization_summary("reporte_optimizacion.html", os.path.join(RAIZ, "modelos"))
print("Guardado en", os.path.join(RAIZ, "modelos"))
""")

md("""
## Para seguir explorando

1. **Priors calibrados:** define `PRIORS_ROI` con un prior informativo para Search
   (p. ej. `(3.0, 0.8)`, como si viniera de un geo-experimento) y observa cómo se corrige el sesgo.
2. **Más muestras:** usa el preset `"Estándar"` y compara R-hat e intervalos.
3. **Cambia la verdad:** `generar_datos(ConfigSimulacion(...))` con otros ROI o más ruido.
   ¿Con cuánto ruido el modelo deja de distinguir canales?
4. **Menos datos:** recorta a 52 semanas. ¿Qué pasa con la incertidumbre?
5. **Modelo nacional:** agrega todas las regiones en una sola y compara con el modelo geo.
""")

nb["cells"] = celdas
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "mmm_meridian_demo.ipynb")
print("ok")
