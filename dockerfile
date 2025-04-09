FROM python:3.11-slim

ENV VERSION=0
ENV MLFLOW_HOST=http://localhost
ENV MLFLOW_PORT=8080
ENV GIT_URI = https://github.com/hanabi70/m2i_formation.git
ENV GIT_BRANCH = mlflow
ENV MLFLOW_EXPERIMENT_NAME = mlops_formation
ENV MODEL_NAME = iris_model


RUN python -m pip install --upgrade pip
RUN pip install uv
RUN uv sync --no-dev

WORKDIR /app
COPY . /app


CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]