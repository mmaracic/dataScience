# MLflow Docker Setup

This directory contains a Docker setup for running MLflow with Python 3.12.

## Files

- `Dockerfile`: Docker image definition with Python 3.12 slim and MLflow
- `docker-compose.yml`: Docker Compose configuration for easy deployment
- `README.md`: This documentation file

## Features

- **Base Image**: Python 3.12 slim for smaller footprint
- **MLflow**: Latest version installed
- **Database**: PostgreSQL 15 for robust data storage
- **Additional Libraries**: psycopg2-binary for PostgreSQL support
- **Security**: Runs as non-root user
- **Persistence**: Data volumes for MLflow runs and PostgreSQL database
- **Ports**: MLflow UI on port 5000, PostgreSQL on port 5432

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the MLflow server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t mlflow-server .

# Run the container
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v mlflow-data:/app/mlruns \
  mlflow-server
```

## Access MLflow UI

Once running, access the MLflow UI at: http://localhost:5000

## Configuration

### Environment Variables

- `MLFLOW_BACKEND_STORE_URI`: Backend store for experiment metadata (configured for PostgreSQL)
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: Artifact storage location (default: ./mlruns)
- `POSTGRES_DB`: PostgreSQL database name (default: mlflow)
- `POSTGRES_USER`: PostgreSQL username (default: mlflow)
- `POSTGRES_PASSWORD`: PostgreSQL password (default: mlflow_password)

### Database Configuration

The setup uses PostgreSQL by default. To change database credentials, modify the docker-compose.yml file:

```yaml
postgres:
  environment:
    POSTGRES_DB: your_db_name
    POSTGRES_USER: your_username
    POSTGRES_PASSWORD: your_secure_password

mlflow:
  environment:
    - MLFLOW_BACKEND_STORE_URI=postgresql://your_username:your_secure_password@postgres:5432/your_db_name
```

### Using Different Storage

To use local directories or network storage for artifacts:

```yaml
environment:
  - MLFLOW_DEFAULT_ARTIFACT_ROOT=/path/to/your/artifacts
volumes:
  - /host/artifacts:/app/artifacts
```

## Development

To extend this setup:

1. Modify the Dockerfile to add additional Python packages
2. Update docker-compose.yml for additional services (database, storage)
3. Mount local directories for development

Example for development with local code:

```yaml
volumes:
  - ./src:/app/src
  - mlflow-data:/app/mlruns
```

## Troubleshooting

- Check container logs: `docker-compose logs mlflow`
- Ensure port 5000 is not in use by another service
- For permission issues, check volume ownership