#!/bin/bash
set -e

# Navigate to the backend directory
cd /Users/tim/Desktop/Owlynn

# Create and activate virtual environment
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test ChromaDB connection
echo "Testing ChromaDB connection..."
python tests/test_chroma_connection.py
if [ $? -ne 0 ]; then
    echo "Failed to connect to ChromaDB. Please ensure the service is running."
    exit 1
fi
echo "ChromaDB connection successful!"

# Run the FastAPI app
exec uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001 