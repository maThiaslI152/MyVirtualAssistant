# Use a lightweight Python image
FROM python:3.11-slim

# Optional: Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip before anything else
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variable for Python module resolution
ENV PYTHONPATH=/app

# Optional: expose if running FastAPI
EXPOSE 8001

# Default command can be overridden in docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
