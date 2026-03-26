FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    build-essential \
    curl \
    unoconv \
    libreoffice-writer \
    libreoffice-core \
    # Fonts for PDF conversion - Liberation fonts are metrically compatible with MS fonts
    fonts-liberation \
    fonts-liberation2 \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-freefont-ttf \
    fonts-noto-core \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    # Add contrib repository for Microsoft Core Fonts
    && echo "deb http://deb.debian.org/debian bookworm main contrib" > /etc/apt/sources.list.d/contrib.list \
    && apt-get update \
    && echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt-get install -y --no-install-recommends ttf-mscorefonts-installer \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f -v

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

WORKDIR /app/src

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
