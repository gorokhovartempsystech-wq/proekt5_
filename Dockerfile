# Инференс на CPU (требование кейса): без torch и CUDA, только бустинг и строковые метрики.
FROM python:3.12-slim

WORKDIR /app

# Зависимости ставим отдельным слоем, чтобы правки кода не пересобирали их заново
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py normalize.py augment.py matcher.py ./
COPY api ./api
COPY work/matcher.pkl ./work/matcher.pkl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
