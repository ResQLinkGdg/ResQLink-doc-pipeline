FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# system deps for pdfplumber (pdfminer needs no extra libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip deps
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# scripts
COPY scripts/ scripts/

# default entrypoint: run build.py
ENTRYPOINT ["python", "scripts/build.py"]
