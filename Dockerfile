FROM python:3

WORKDIR /usr/src/app

COPY these-classifier.py ./
RUN pip install --no-cache-dir sentence-transformers

COPY . .
RUN python3 these-classifier.py dirty precision "I am the Walrus"
