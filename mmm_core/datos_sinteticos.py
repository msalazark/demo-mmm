"""Generador de datos sintéticos para la demo de Marketing Mix Modeling.

Simula un retailer peruano ficticio ("Tiendas Pacífico") que vende en 10 regiones
y pauta en 5 canales. Como nosotros fabricamos los datos, conocemos el ROI real de
cada canal: eso permite comprobar si Meridian lo recupera ("verdad de terreno").

Proceso generador (por región g y semana t):
    ventas = base_g * tendencia * estacionalidad * efecto_precio * (1 + evento)
             + sum_m beta_m * poblacion_g * hill(adstock(impresiones_per_capita))
             + ruido
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# Regiones y población aproximada (millones de habitantes)
REGIONES = {
    "Lima": 10.9,
    "Piura": 2.1,
    "La Libertad": 2.0,
    "Arequipa": 1.5,
    "Cajamarca": 1.45,
    "Junin": 1.36,
    "Cusco": 1.36,
    "Lambayeque": 1.33,
    "Puno": 1.24,
    "Ancash": 1.2,
}

CANALES = ["tv", "youtube", "meta", "google_search", "tiktok"]

NOMBRES_CANALES = {
    "tv": "TV",
    "youtube": "YouTube",
    "meta": "Meta",
    "google_search": "Google Search",
    "tiktok": "TikTok",
}

CONTROLES = ["indice_precio", "evento_comercial"]


@dataclass
class ParametrosCanal:
    """Parámetros 'reales' de un canal (los que el modelo debe descubrir)."""

    roi: float              # ROI real: ventas incrementales / inversión
    alpha: float            # decaimiento del adstock geométrico (0 = sin arrastre)
    ec_factor: float        # media saturación relativa a la mediana de exposición
    pendiente: float        # pendiente de la curva Hill
    inversion_semanal: float  # inversión nacional promedio por semana (S/)
    cpm: float              # costo por mil impresiones (S/)


PARAMETROS_DEFECTO: dict[str, ParametrosCanal] = {
    "tv": ParametrosCanal(roi=1.6, alpha=0.70, ec_factor=1.3, pendiente=1.0,
                          inversion_semanal=120_000, cpm=12.0),
    "youtube": ParametrosCanal(roi=1.1, alpha=0.50, ec_factor=1.0, pendiente=1.0,
                               inversion_semanal=40_000, cpm=8.0),
    "meta": ParametrosCanal(roi=2.4, alpha=0.30, ec_factor=1.0, pendiente=1.0,
                            inversion_semanal=60_000, cpm=6.0),
    "google_search": ParametrosCanal(roi=3.2, alpha=0.10, ec_factor=0.7, pendiente=1.0,
                                     inversion_semanal=50_000, cpm=25.0),
    "tiktok": ParametrosCanal(roi=0.9, alpha=0.35, ec_factor=1.0, pendiente=1.0,
                              inversion_semanal=25_000, cpm=5.0),
}


@dataclass
class ConfigSimulacion:
    """Configuración general de la simulación."""

    n_semanas: int = 104
    fecha_inicio: str = "2024-10-07"   # lunes
    venta_per_capita: float = 0.11     # ventas base semanales por habitante (S/)
    ruido_relativo: float = 0.04       # desviación del ruido como % de la base
    max_lag: int = 8
    semilla: int = 2026
    canales: dict[str, ParametrosCanal] = field(
        default_factory=lambda: {k: ParametrosCanal(**asdict(v))
                                 for k, v in PARAMETROS_DEFECTO.items()}
    )


# ----------------------------------------------------------------------------
# Transformaciones de medios
# ----------------------------------------------------------------------------
def adstock_geometrico(x: np.ndarray, alpha: float, max_lag: int) -> np.ndarray:
    """Adstock geométrico normalizado sobre el eje de tiempo (última dimensión)."""
    pesos = alpha ** np.arange(max_lag + 1)
    pesos = pesos / pesos.sum()
    salida = np.zeros_like(x, dtype=float)
    for lag, w in enumerate(pesos):
        if lag == 0:
            salida += w * x
        else:
            salida[..., lag:] += w * x[..., :-lag]
    return salida


def hill(x: np.ndarray, ec: float, pendiente: float) -> np.ndarray:
    """Curva de saturación Hill: 0 sin exposición, 0.5 en x = ec, tiende a 1."""
    return x**pendiente / (x**pendiente + ec**pendiente)


# ----------------------------------------------------------------------------
# Componentes del calendario peruano
# ----------------------------------------------------------------------------
def _evento_comercial(fechas: pd.DatetimeIndex) -> np.ndarray:
    """Intensidad de eventos comerciales: Navidad, Cyber Wow, Día de la Madre,
    Fiestas Patrias y Black Friday (0 = semana normal)."""
    evento = np.zeros(len(fechas))
    for i, f in enumerate(fechas):
        fin = f + pd.Timedelta(days=6)
        m, d = f.month, f.day
        if m == 12 and d >= 8:                       # campaña navideña
            evento[i] = 1.0 if d < 22 else 1.4
        elif m == 5 and 3 <= d <= 11:                # Día de la Madre (2.º domingo de mayo)
            evento[i] = 0.8
        elif (m == 7 and 15 <= d <= 28):             # Fiestas Patrias / gratificación
            evento[i] = 0.6
        elif fin.month == 11 and 22 <= fin.day <= 30:  # Black Friday
            evento[i] = 0.7
        elif (m in (4, 7, 10) and 1 <= d <= 7):      # Cyber Wow (aprox.)
            evento[i] = 0.5
    return evento


def _patron_inversion(canal: str, n: int, evento: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Patrón temporal (media ~1) de la inversión nacional de cada canal."""
    t = np.arange(n)
    if canal == "tv":
        # TV en flights: 4 semanas al aire, 3 fuera, más fuerte en eventos
        flight = ((t % 7) < 4).astype(float)
        patron = flight * (1 + 0.8 * evento) * rng.lognormal(0, 0.15, n)
    elif canal == "youtube":
        campañas = ((t % 10) < 6).astype(float) * 0.9 + 0.25
        patron = campañas * rng.lognormal(0, 0.25, n)
    elif canal == "meta":
        patron = (1 + 0.4 * evento) * rng.lognormal(0, 0.30, n)
    elif canal == "google_search":
        # Search sigue la demanda (endogeneidad leve, como en la vida real)
        patron = (1 + 0.3 * evento + 0.1 * np.sin(2 * np.pi * t / 52)) * rng.lognormal(0, 0.12, n)
    elif canal == "tiktok":
        # TikTok arranca a mitad del periodo y va creciendo
        inicio = int(n * 0.4)
        rampa = np.clip((t - inicio) / 20, 0, 1)
        patron = rampa * rng.lognormal(0, 0.35, n)
    else:
        patron = rng.lognormal(0, 0.3, n)
    return patron / patron[patron > 0].mean()


# ----------------------------------------------------------------------------
# Generador principal
# ----------------------------------------------------------------------------
def generar_datos(config: ConfigSimulacion | None = None) -> tuple[pd.DataFrame, dict]:
    """Genera el dataset geo x semana y la verdad de terreno.

    Retorna:
        df: DataFrame largo con columnas geo, time, population, ventas,
            controles, <canal>_impresiones y <canal>_inversion.
        verdad: dict con ROI, contribución y parámetros reales por canal.
    """
    cfg = config or ConfigSimulacion()
    rng = np.random.default_rng(cfg.semilla)

    geos = list(REGIONES)
    poblacion = np.array([REGIONES[g] * 1e6 for g in geos])
    share_pob = poblacion / poblacion.sum()
    n_g, n_t = len(geos), cfg.n_semanas
    fechas = pd.date_range(cfg.fecha_inicio, periods=n_t, freq="W-MON")
    t = np.arange(n_t)

    # --- Controles ---------------------------------------------------------
    evento = _evento_comercial(fechas)
    evento_gt = np.tile(evento, (n_g, 1))
    # índice de precio: paseo aleatorio suave alrededor de 1, con variación regional
    precio_nac = 1 + np.cumsum(rng.normal(0, 0.006, n_t))
    precio_nac = precio_nac - precio_nac.mean() + 1
    precio = precio_nac[None, :] + rng.normal(0, 0.01, (n_g, n_t))

    # --- Línea base (ventas sin publicidad) -------------------------------
    base_geo = poblacion * cfg.venta_per_capita * rng.uniform(0.85, 1.15, n_g)
    tendencia = 1 + 0.08 * t / n_t
    estacionalidad = 1 + 0.06 * np.sin(2 * np.pi * (t + 10) / 52)
    efecto_precio = precio ** (-1.2)
    linea_base = (base_geo[:, None] * tendencia * estacionalidad
                  * efecto_precio * (1 + 0.35 * evento_gt))

    # --- Medios ------------------------------------------------------------
    columnas = {}
    contribucion = {}
    verdad_canales = {}
    for canal, p in cfg.canales.items():
        patron = _patron_inversion(canal, n_t, evento, rng)
        inversion_nac = p.inversion_semanal * patron
        # TV concentrada en Lima; digital proporcional a la población con ruido
        peso_geo = share_pob ** (1.15 if canal == "tv" else 1.0)
        peso_geo = peso_geo / peso_geo.sum()
        ruido_geo = rng.lognormal(0, 0.15, (n_g, n_t))
        inversion = inversion_nac[None, :] * peso_geo[:, None] * ruido_geo
        cpm_geo = p.cpm * rng.uniform(0.9, 1.1, (n_g, 1))
        impresiones = inversion / cpm_geo * 1000

        # Respuesta: hill(adstock(impresiones per cápita))
        per_capita = impresiones / poblacion[:, None]
        ad = adstock_geometrico(per_capita, p.alpha, cfg.max_lag)
        ec = p.ec_factor * np.median(ad[ad > 0])
        respuesta = hill(ad, ec, p.pendiente) * poblacion[:, None]

        # Calibramos beta para que el ROI real sea exactamente p.roi
        beta = p.roi * inversion.sum() / respuesta.sum()
        contribucion[canal] = beta * respuesta

        columnas[f"{canal}_impresiones"] = impresiones
        columnas[f"{canal}_inversion"] = inversion
        verdad_canales[canal] = {
            "roi_real": round(float(contribucion[canal].sum() / inversion.sum()), 4),
            "inversion_total": float(inversion.sum()),
            "contribucion_total": float(contribucion[canal].sum()),
            "alpha_adstock": p.alpha,
            "ec_factor": p.ec_factor,
            "pendiente_hill": p.pendiente,
            "cpm": p.cpm,
        }

    contrib_medios = sum(contribucion.values())
    ruido = rng.normal(0, cfg.ruido_relativo, (n_g, n_t)) * linea_base
    ventas = linea_base + contrib_medios + ruido

    # --- Ensamblado en formato largo --------------------------------------
    filas = {
        "geo": np.repeat(geos, n_t),
        "time": np.tile(fechas.strftime("%Y-%m-%d"), n_g),
        "population": np.repeat(poblacion, n_t),
        "ventas": ventas.ravel(),
        "indice_precio": precio.ravel(),
        "evento_comercial": evento_gt.ravel(),
    }
    for col, matriz in columnas.items():
        filas[col] = matriz.ravel()
    df = pd.DataFrame(filas)

    ventas_tot = float(ventas.sum())
    for canal in verdad_canales:
        verdad_canales[canal]["share_ventas"] = round(
            verdad_canales[canal]["contribucion_total"] / ventas_tot, 4)

    verdad = {
        "descripcion": "Verdad de terreno de la simulación (Tiendas Pacífico, ficticio)",
        "ventas_totales": ventas_tot,
        "linea_base_total": float(linea_base.sum()),
        "share_medios": round(float(contrib_medios.sum() / ventas_tot), 4),
        "config": {"n_semanas": n_t, "fecha_inicio": cfg.fecha_inicio,
                   "semilla": cfg.semilla, "max_lag": cfg.max_lag},
        "canales": verdad_canales,
    }
    return df, verdad


def guardar(df: pd.DataFrame, verdad: dict, ruta_csv: str, ruta_json: str) -> None:
    df.to_csv(ruta_csv, index=False)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(verdad, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "data")
    df, verdad = generar_datos()
    guardar(df, verdad, os.path.join(base, "datos_sinteticos.csv"),
            os.path.join(base, "verdad_terreno.json"))
    print(df.head())
    print(json.dumps({k: v["roi_real"] for k, v in verdad["canales"].items()}, indent=2))
    print("share medios:", verdad["share_medios"])
