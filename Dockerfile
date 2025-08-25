FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends make gcc g++ libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Makefile ./
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
COPY requirements-test.txt requirements-test.txt

RUN make install

COPY challenge/ challenge/
COPY data/ data/

EXPOSE 8080

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
