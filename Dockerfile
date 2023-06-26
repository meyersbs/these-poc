FROM python:3

WORKDIR /usr/src/app

COPY these-classifier.py ./
RUN pip install --no-cache-dir sentence-transformers

COPY . .
