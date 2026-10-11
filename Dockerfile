# Para Macs con Intel (TensorFlow 2.21 no tiene versión nativa para macOS x86_64)
# o para desplegar en un servidor / Cloud Run.
#
#   docker build -t mmm-meridian .
#   docker run -p 8501:8501 mmm-meridian            # app en http://localhost:8501
#   docker run -p 8888:8888 mmm-meridian jupyter lab --ip=0.0.0.0 --allow-root --no-browser
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8888
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
