"""Funciones de modelado con Google Meridian, compartidas por el notebook y la app.

Flujo:  DataFrame -> InputData -> ModelSpec (priors) -> Meridian -> MCMC
        -> Analyzer (ROI, contribución, curvas) -> BudgetOptimizer
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from meridian import constants  # noqa: E402
from meridian.analysis import analyzer, optimizer  # noqa: E402
from meridian.data import data_frame_input_data_builder as dfb  # noqa: E402
from meridian.model import model, prior_distribution, spec  # noqa: E402

from meridian.common import currency as _currency  # noqa: E402

from .datos_sinteticos import CANALES, CONTROLES, NOMBRES_CANALES  # noqa: E402

# Los gráficos nativos de Meridian usan "$" por defecto. Para la demo mostramos
# soles: interceptamos el símbolo de moneda (PEN o sin código -> "S/").
_simbolo_original = _currency.get_currency_symbol


def _simbolo_soles(currency_code=None, default="S/"):
    if not currency_code or str(currency_code).upper() == "PEN":
        return "S/"
    return _simbolo_original(currency_code, default)


_currency.get_currency_symbol = _simbolo_soles


# ----------------------------------------------------------------------------
# Presets de muestreo MCMC
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PresetMCMC:
    n_chains: int
    n_adapt: int
    n_burnin: int
    n_keep: int
    descripcion: str


PRESETS = {
    "Rápido (demo en vivo)": PresetMCMC(2, 300, 200, 300,
                                        "Para mostrar en clase o reunión. Diagnósticos aproximados."),
    "Estándar": PresetMCMC(4, 500, 500, 500,
                           "Equilibrio entre tiempo y calidad."),
    "Producción": PresetMCMC(4, 2000, 500, 1000,
                             "Recomendado por Google para resultados finales."),
}


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def construir_input_data(df: pd.DataFrame, canales: list[str] | None = None,
                         controles: list[str] | None = None):
    """Convierte el DataFrame largo (geo x semana) en InputData de Meridian."""
    canales = canales or CANALES
    controles = controles if controles is not None else CONTROLES
    builder = (
        dfb.DataFrameInputDataBuilder(kpi_type=constants.REVENUE)
        .with_kpi(df, kpi_col="ventas")
        .with_population(df, population_col="population")
        .with_media(
            df,
            media_cols=[f"{c}_impresiones" for c in canales],
            media_spend_cols=[f"{c}_inversion" for c in canales],
            media_channels=canales,
        )
    )
    if controles:
        builder = builder.with_controls(df, control_cols=controles)
    builder.currency_code = "PEN"
    return builder.build()


# ----------------------------------------------------------------------------
# Especificación del modelo
# ----------------------------------------------------------------------------
def construir_spec(
    n_semanas: int,
    priors_roi: dict[str, tuple[float, float]] | None = None,
    canales: list[str] | None = None,
    knots_cada: int = 8,
    max_lag: int = 8,
) -> spec.ModelSpec:
    """Arma el ModelSpec.

    Args:
        priors_roi: {canal: (media, desviación)} del ROI esperado. Si es None se
            usa el prior por defecto de Meridian (LogNormal(0.2, 0.9)).
            Aquí es donde entra el conocimiento de negocio o los experimentos
            (geo-lift, conversion lift) para calibrar el modelo.
        knots_cada: un nudo de la línea base temporal cada N semanas.
    """
    canales = canales or CANALES
    if priors_roi:
        medias = [priors_roi[c][0] for c in canales]
        desv = [priors_roi[c][1] for c in canales]
        dist = prior_distribution.lognormal_dist_from_mean_std(medias, desv)
        from meridian import backend
        dist = backend.tfd.LogNormal(dist.loc, dist.scale, name=constants.ROI_M)
        prior = prior_distribution.PriorDistribution(roi_m=dist)
    else:
        prior = prior_distribution.PriorDistribution()

    n_knots = max(2, int(np.ceil(n_semanas / knots_cada)))
    return spec.ModelSpec(
        prior=prior,
        media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
        knots=n_knots,
        max_lag=max_lag,
    )


# ----------------------------------------------------------------------------
# Ajuste
# ----------------------------------------------------------------------------
def ajustar_modelo(input_data, model_spec, preset: PresetMCMC, semilla: int = 42,
                   n_prior: int = 300):
    """Crea el objeto Meridian y corre prior + posterior (NUTS). Retorna (mmm, segundos)."""
    inicio = time.time()
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    mmm.sample_prior(n_prior, seed=semilla)
    mmm.sample_posterior(
        n_chains=preset.n_chains,
        n_adapt=preset.n_adapt,
        n_burnin=preset.n_burnin,
        n_keep=preset.n_keep,
        seed=semilla,
    )
    return mmm, time.time() - inicio


def guardar_modelo(mmm, ruta: str) -> None:
    model.save_mmm(mmm, ruta)


def cargar_modelo(ruta: str):
    return model.load_mmm(ruta)


# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------
def _resumen_muestras(muestras: np.ndarray, nivel: float = 0.9) -> tuple:
    """Media y CI a partir de un array (chains, draws, canales)."""
    plano = muestras.reshape(-1, muestras.shape[-1])
    q = (1 - nivel) / 2
    return plano.mean(0), np.quantile(plano, q, 0), np.quantile(plano, 1 - q, 0)


def tabla_roi(mmm, verdad: dict | None = None, nivel: float = 0.9) -> pd.DataFrame:
    """ROI posterior por canal (media + intervalo creíble) vs ROI real si se conoce."""
    an = analyzer.Analyzer(mmm)
    roi = np.asarray(an.roi())
    media, bajo, alto = _resumen_muestras(roi, nivel)
    canales = list(mmm.input_data.media_channel.values)
    tabla = pd.DataFrame({
        "canal": canales,
        "nombre": [NOMBRES_CANALES.get(c, c) for c in canales],
        "roi_estimado": media,
        "ci_bajo": bajo,
        "ci_alto": alto,
    })
    if verdad:
        tabla["roi_real"] = [verdad["canales"][c]["roi_real"] for c in canales]
        tabla["error_%"] = (tabla["roi_estimado"] / tabla["roi_real"] - 1) * 100
        tabla["real_dentro_ci"] = (tabla["roi_real"] >= tabla["ci_bajo"]) & (
            tabla["roi_real"] <= tabla["ci_alto"])
    return tabla


def tabla_resumen(mmm, nivel: float = 0.9) -> pd.DataFrame:
    """Inversión, contribución incremental, ROI, mROI y CPIK por canal."""
    an = analyzer.Analyzer(mmm)
    ds = an.summary_metrics(confidence_level=nivel, include_non_paid_channels=False)
    post = ds.sel(distribution=constants.POSTERIOR, metric=constants.MEAN)
    canales = [c for c in ds.channel.values if c != constants.ALL_CHANNELS]
    filas = []
    for c in canales:
        p = post.sel(channel=c)
        filas.append({
            "canal": c,
            "nombre": NOMBRES_CANALES.get(c, c),
            "inversion": float(ds[constants.SPEND].sel(channel=c)),
            "pct_inversion": float(ds[constants.PCT_OF_SPEND].sel(channel=c)),
            "ventas_incrementales": float(p[constants.INCREMENTAL_OUTCOME]),
            "pct_contribucion": float(p[constants.PCT_OF_CONTRIBUTION]),
            "roi": float(p[constants.ROI]),
            "mroi": float(p[constants.MROI]),
            "cpik": float(post.sel(channel=c)[constants.CPIK]) if constants.CPIK in post else np.nan,
        })
    return pd.DataFrame(filas)


def ajuste_y_diagnostico(mmm) -> dict:
    """R-hat máximo por parámetro y métricas de precisión (R2, MAPE, wMAPE)."""
    an = analyzer.Analyzer(mmm)
    rhat = an.rhat_summary()
    precision = an.predictive_accuracy().to_dataframe().reset_index()
    return {"rhat": rhat, "precision": precision}


def serie_ajuste_nacional(mmm, nivel: float = 0.9) -> pd.DataFrame:
    """Ventas reales vs esperadas (agregado nacional) y línea base, por semana."""
    an = analyzer.Analyzer(mmm)
    ds = an.expected_vs_actual_data(confidence_level=nivel, aggregate_geos=True)
    df = ds.to_dataframe().reset_index()
    return df


def graficos_meridian(mmm) -> dict:
    """Genera los gráficos nativos de Meridian como especificaciones Vega-Lite (dict).

    Se guardan en JSON junto al modelo pre-entrenado para que la app cargue al
    instante (las curvas de respuesta tardan ~1 min en CPU).
    """
    from meridian.analysis import visualizer

    fit = visualizer.ModelFit(mmm)
    diag = visualizer.ModelDiagnostics(mmm)
    efectos = visualizer.MediaEffects(mmm)
    resumen = visualizer.MediaSummary(mmm)

    hill = efectos.plot_hill_curves()
    hill = list(hill.values()) if isinstance(hill, dict) else [hill]
    graficos = {
        "ajuste": fit.plot_model_fit(),
        "rhat": diag.plot_rhat_boxplot(),
        "respuesta": efectos.plot_response_curves(plot_separately=False),
        "adstock": efectos.plot_adstock_decay(),
        "cascada": resumen.plot_contribution_waterfall_chart(),
        "inversion_vs_contribucion": resumen.plot_spend_vs_contribution(),
        "roi_vs_mroi": resumen.plot_roi_vs_mroi(),
    }
    salida = {k: v.to_dict() for k, v in graficos.items()}
    salida["hill"] = [h.to_dict() for h in hill]
    return salida


def optimizar_presupuesto(
    mmm,
    presupuesto: float | None = None,
    presupuesto_fijo: bool = True,
    roi_objetivo: float | None = None,
    limite_bajo: float = 0.3,
    limite_alto: float = 0.3,
):
    """Optimiza la distribución del presupuesto entre canales.

    limite_bajo / limite_alto: variación máxima permitida por canal respecto a
    su inversión histórica (0.3 = ±30 %).
    """
    opt = optimizer.BudgetOptimizer(mmm)
    kwargs = dict(
        fixed_budget=presupuesto_fijo,
        spend_constraint_lower=limite_bajo,
        spend_constraint_upper=limite_alto,
    )
    if presupuesto_fijo:
        kwargs["budget"] = presupuesto
    else:
        kwargs["target_roi"] = roi_objetivo
    return opt.optimize(**kwargs)


def tabla_optimizacion(resultado) -> pd.DataFrame:
    """Compara inversión e incremental histórico vs optimizado por canal."""
    no_opt = resultado.nonoptimized_data
    opt = resultado.optimized_data
    canales = list(no_opt.channel.values)
    filas = []
    for c in canales:
        filas.append({
            "canal": c,
            "nombre": NOMBRES_CANALES.get(c, c),
            "inversion_actual": float(no_opt[constants.SPEND].sel(channel=c)),
            "inversion_optima": float(opt[constants.SPEND].sel(channel=c)),
            "ventas_inc_actual": float(no_opt[constants.INCREMENTAL_OUTCOME].sel(
                channel=c, metric=constants.MEAN)),
            "ventas_inc_optima": float(opt[constants.INCREMENTAL_OUTCOME].sel(
                channel=c, metric=constants.MEAN)),
        })
    t = pd.DataFrame(filas)
    t["cambio_inversion_%"] = (t["inversion_optima"] / t["inversion_actual"] - 1) * 100
    return t
