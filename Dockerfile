FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install only production API dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/

# Copy trained model artifacts
COPY artifacts/final_extra_trees_model.joblib ./artifacts/final_extra_trees_model.joblib
COPY artifacts/feature_columns.joblib ./artifacts/feature_columns.joblib
COPY artifacts/model_metadata.json ./artifacts/model_metadata.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]