FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

WORKDIR /app/src

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
