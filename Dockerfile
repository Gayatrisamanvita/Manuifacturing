# Multi-stage/lightweight Dockerfile for FactorySmart AI Backend & Dashboard
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies needed for C++ builds (LightGBM, XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files with proper ownership
COPY --chown=user:user . .

USER user

# Expose ports (7860 for Hugging Face Spaces, 8000 for unified API, 8501 for Streamlit)
EXPOSE 7860 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command: launch Unified FastAPI server
CMD ["python", "run_api.py", "--host", "0.0.0.0"]
