# Use Python 3.12 slim as base image for smaller footprint
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and PostgreSQL driver
RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary

# Create directories for MLflow
RUN mkdir -p /app/mlruns

# Expose MLflow UI port
EXPOSE 5000

# Create a non-root user for security
RUN useradd -m -u 1000 mlflow && chown -R mlflow:mlflow /app
USER mlflow

# Default command to start MLflow server
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"]