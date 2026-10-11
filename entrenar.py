"""Entrena el modelo de la demo desde la terminal y lo guarda en modelos/.

Uso:
    python entrenar.py                      # preset "Rápido (demo en vivo)"
    python entrenar.py --preset Estándar
"""

import argparse
import json
import os

import pandas as pd

from mmm_core import modelo
from mmm_core.datos_sinteticos import generar_datos, guardar

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(BASE, "data", "datos_sinteticos.csv")
RUTA_VERDAD = os.path.join(BASE, "data", "verdad_terreno.json")
RUTA_MODELO = os.path.join(BASE, "modelos", "modelo_demo.pkl")
RUTA_GRAFICOS = os.path.join(BASE, "modelos", "graficos_demo.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="Rápido (demo en vivo)", choices=list(modelo.PRESETS))
    parser.add_argument("--regenerar", action="store_true", help="vuelve a generar los datos sintéticos")
    args = parser.parse_args()

    if args.regenerar or not os.path.exists(RUTA_CSV):
        df, verdad = generar_datos()
        guardar(df, verdad, RUTA_CSV, RUTA_VERDAD)
    df = pd.read_csv(RUTA_CSV)
    with open(RUTA_VERDAD, encoding="utf-8") as f:
        verdad = json.load(f)

    input_data = modelo.construir_input_data(df)
    spec = modelo.construir_spec(n_semanas=df["time"].nunique())
    preset = modelo.PRESETS[args.preset]
    print(f"Entrenando con preset '{args.preset}': {preset}")
    mmm, segundos = modelo.ajustar_modelo(input_data, spec, preset)
    print(f"Listo en {segundos / 60:.1f} min")
    modelo.guardar_modelo(mmm, RUTA_MODELO)
    print("Pre-calculando gráficos para la app…")
    with open(RUTA_GRAFICOS, "w", encoding="utf-8") as f:
        json.dump(modelo.graficos_meridian(mmm), f)
    print(modelo.tabla_roi(mmm, verdad).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
