# Ejecutar la demo MMM con Meridian en una Mac Intel

Meridian 2.x necesita TensorFlow 2.21 y JAX 0.11, que **no tienen versión para macOS con
Intel**. La solución es correrlo dentro de **Docker usando Colima**, que crea una pequeña
máquina Linux en la Mac. En Linux, Meridian sí se instala.

> En una Mac con Apple Silicon (M1–M4) no hace falta Docker: basta con
> `pip install -r requirements.txt` y `streamlit run app.py`.

---

## 1. Preparar Colima y Docker (solo la primera vez)

```bash
# Si no tienes el comando docker:
brew install docker colima

# Levanta Colima con recursos suficientes (por defecto trae 2 CPU y 2 GB, que no alcanzan)
colima start --cpu 4 --memory 8 --disk 30
docker info   # debe responder sin error
```

- Con 32 GB de RAM puedes subir a `--cpu 6 --memory 12` para entrenar más rápido.
- Si Colima ya estaba creado con menos recursos, primero haz `colima stop` y luego
  `colima start` con los nuevos valores.

## 2. Construir la imagen

```bash
cd ~/Downloads            # o donde guardaste el zip
unzip mmm_meridian_demo.zip
cd mmm_meridian_demo
docker build -t mmm-meridian .
```

La primera vez tarda entre 5 y 10 minutos y la imagen pesa unos 3 GB (descarga TensorFlow
y JAX).

## 3a. Correr la app (Streamlit)

```bash
docker run --rm -p 8501:8501 -v "$PWD/modelos:/app/modelos" mmm-meridian
```

Abre **http://localhost:8501** en el navegador.

El `-v` monta la carpeta `modelos/`, así que los modelos que entrenes desde la app quedan
guardados en tu Mac.

## 3b. Correr el notebook (Jupyter)

```bash
docker run --rm -p 8888:8888 -v "$PWD:/app" mmm-meridian \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

En la terminal aparece un enlace del tipo `http://127.0.0.1:8888/lab?token=...`. Ábrelo y
entra a `notebooks/mmm_meridian_demo.ipynb`.

Aquí se monta la carpeta completa, así que los cambios que hagas al notebook quedan en tu Mac.

## 4. Apagar todo

1. Presiona `Ctrl+C` en la terminal donde corre el contenedor.
2. Ejecuta `colima stop` para liberar memoria.

---

## Qué esperar en la Mac Intel

| Acción | Tiempo aproximado |
|---|---|
| Abrir la app con el modelo pre-entrenado | Al instante |
| Entrenar con el preset "Rápido" | 3 – 5 min |
| Entrenar con el preset "Estándar" | 8 – 12 min |
| Optimizar presupuesto | ~1 min |

Para presentar, conviene usar el **modelo pre-entrenado** (ya viene incluido y abre al
instante) y dejar el reentrenamiento para mostrarlo como parte de la demo.

## Alternativa: solo el notebook en Google Colab

Si solo necesitas el notebook y no quieres usar Docker:

1. Sube `mmm_meridian_demo.zip` a Colab.
2. Descomprímelo:
   ```python
   !unzip mmm_meridian_demo.zip
   ```
3. Instala las dependencias:
   ```python
   !pip install -r mmm_meridian_demo/requirements.txt
   ```
4. Abre `mmm_meridian_demo/notebooks/mmm_meridian_demo.ipynb`.

La app de Streamlit no funciona bien en Colab; para la app usa Docker.

---

## Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Cannot connect to the Docker daemon` | Colima no está corriendo | `colima start --cpu 4 --memory 8` |
| El contenedor se cierra durante el entrenamiento (`Killed`) | Poca memoria en Colima | `colima stop` y `colima start --cpu 4 --memory 12` |
| `port is already allocated` | El puerto 8501 u 8888 ya está en uso | Usa otro puerto: `-p 8502:8501` y abre `localhost:8502` |
| La app dice que el modelo pre-entrenado no es compatible | Versiones distintas de Meridian/JAX | Reentrena desde la barra lateral (paso 3) o con `docker run --rm -v "$PWD:/app" mmm-meridian python entrenar.py --preset Estándar` |
