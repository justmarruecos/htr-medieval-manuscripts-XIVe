FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dataset_nlp/splits/ ./dataset_nlp/splits/

RUN python3 -c "from transformers import pipeline; pipeline('ner', model='Jean-Baptiste/camembert-ner', aggregation_strategy='simple')"

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
