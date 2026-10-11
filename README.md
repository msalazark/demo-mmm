# Demo de Marketing Mix Modeling con Google Meridian

Demo completa de MMM bayesiano con [Google Meridian](https://developers.google.com/meridian)
2.1 sobre datos sintéticos de un retailer ficticio (*Tiendas Pacífico*: 10 regiones del Perú,
104 semanas, 5 canales). Como los datos son simulados, **conocemos el ROI real de cada canal**
y podemos comprobar si el modelo lo recupera.

Trae dos formas de uso que comparten el mismo código (`mmm_core/`):

| | Para qué | Cómo |
|---|---|---|
| **App Streamlit** (`app.py`) | Demo en vivo a clientes o en clase: explorar resultados, mover priors, reentrenar, optimizar presupuesto | `streamlit run app.py` |
| **Notebook** (`notebooks/mmm_meridian_demo.ipynb`) | Enseñar el pipeline paso a paso, con explicación y ejercicios | `jupyter lab` |

## ¿Por qué Streamlit y no React?

Meridian es una librería de Python (JAX/TensorFlow Probability) que estima con MCMC.
**No puede correr en el navegador**, así que una app React necesitaría además un backend
(p. ej. FastAPI) que entrene y sirva resultados, más una cola de trabajos, porque el ajuste
toma minutos. Para una demo eso duplica el trabajo sin ganar nada.

- **Streamlit:** llama a Meridian directamente y muestra sus gráficos nativos (Altair).
  Es la opción para demos internas, clases y pilotos con clientes.
- **Notebook:** es la mejor herramienta para enseñar y auditar el modelo, pero no sirve
  para mostrar a un cliente no técnico.
- **React + FastAPI:** solo vale la pena si el MMM se vuelve un **producto** (multi-cliente,
  login, escenarios guardados, marca propia). En ese caso, `mmm_core/` sirve tal cual como
  capa de servicio del backend.

## Instalación

> **Importante:** Meridian 2.x necesita TensorFlow 2.21 y JAX 0.11, que **no tienen versión
> para Mac con Intel**. Opciones: un Mac con Apple Silicon (M1–M4), Linux/Windows (WSL)
> o Docker.

### Mac con Apple Silicon, Linux o WSL

```bash
cd mmm_meridian_demo
python3.13 -m venv .venv          # 3.11, 3.12 o 3.13
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py              # http://localhost:8501
# o
jupyter lab notebooks/mmm_meridian_demo.ipynb
```

### Mac con Intel (o cualquier equipo) con Docker / Colima

```bash
colima start --cpu 4 --memory 8   # solo si usas Colima
docker build -t mmm-meridian .
docker run -p 8501:8501 mmm-meridian                      # app
docker run -p 8888:8888 mmm-meridian \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser      # notebook
```

## Estructura

```
mmm_meridian_demo/
├── app.py                      # App Streamlit (6 pestañas)
├── entrenar.py                 # Reentrena el modelo pre-entrenado desde la terminal
├── mmm_core/
│   ├── datos_sinteticos.py     # Generador con ROI real conocido (adstock + Hill)
│   └── modelo.py               # InputData, ModelSpec, ajuste, ROI, optimizador
├── data/
│   ├── datos_sinteticos.csv    # 10 regiones × 104 semanas
│   └── verdad_terreno.json     # ROI, adstock y contribución reales por canal
├── modelos/
│   ├── modelo_demo.pkl         # Modelo pre-entrenado (preset Estándar: 4 × 500)
│   └── graficos_demo.json      # Gráficos pre-calculados para que la app abra al instante
├── notebooks/
│   ├── mmm_meridian_demo.ipynb # Notebook ejecutado, con salidas
│   └── construir_notebook.py   # Genera el .ipynb
├── requirements.txt
└── Dockerfile
```

## La app

| Pestaña | Contenido |
|---|---|
| 📊 Datos | Ventas con eventos comerciales, inversión por canal y por región, descarga del CSV |
| ✅ Ajuste del modelo | R², MAPE, R-hat, ventas reales vs. esperadas vs. línea base |
| 💰 ROI y contribución | **ROI estimado (IC 90 %) vs. ROI real**, ROI marginal, cascada de contribución |
| 📈 Curvas de respuesta | Curvas de respuesta, adstock y saturación Hill (prior vs. posterior) |
| 🎯 Optimizador | Presupuesto fijo o con ROI objetivo y límite de cambio por canal |
| 📘 Cómo funciona | Explicación del modelo para audiencias no técnicas |

En la barra lateral:

1. **Datos**: el *laboratorio* permite fijar otro ROI real por canal y regenerar los datos, o
   subir un CSV propio con el mismo formato.
2. **Priors de ROI**: usar priors propios por canal (media y desviación), como si vinieran de
   experimentos o benchmarks.
3. **Entrenar**: presets *Rápido* (~2–3 min), *Estándar* (~6 min) y *Producción*.
4. **Exportar**: descarga el reporte HTML nativo de Meridian.

Los tiempos son los de una CPU de 2 núcleos; en un M4 deberían ser menores.

## Resultados del modelo pre-entrenado

| Canal | ROI real | ROI estimado | IC 90 % |
|---|---|---|---|
| TV | 1.60 | 1.60 | 0.98 – 2.53 |
| YouTube | 1.10 | 1.10 | 0.36 – 2.18 |
| Meta | 2.40 | 3.00 | 1.34 – 4.87 |
| Google Search | 3.20 | 2.39 | 0.80 – 4.34 |
| TikTok | 0.90 | 1.37 | 0.28 – 3.43 |

El ROI real cae dentro del intervalo creíble en los 5 canales. **Google Search queda
subestimado**: su inversión sigue a la demanda y varía poco, así que el modelo atribuye parte
de su efecto a la línea base. Es endogeneidad, y sirve para explicar por qué conviene
calibrar con experimentos (geo-lift) mediante priors de ROI.

## Usar datos reales

Formato largo, una fila por región y semana:

`geo, time (YYYY-MM-DD), population, ventas, indice_precio, evento_comercial,
<canal>_impresiones, <canal>_inversion` para `tv, youtube, meta, google_search, tiktok`.

Para otros canales o controles, edita `CANALES` y `CONTROLES` en
`mmm_core/datos_sinteticos.py` (y la validación del CSV en `app.py`). Si el KPI son
conversiones, cambia a `kpi_type="non_revenue"` y agrega `revenue_per_kpi`.

## Notas

- El modelo pre-entrenado es un pickle (`save_mmm`) y solo carga con las versiones exactas de
  `requirements.txt`. Si falla, corre `python entrenar.py --preset Estándar`.
- Los gráficos nativos de Meridian tienen títulos en inglés; los montos se muestran en soles (S/).
