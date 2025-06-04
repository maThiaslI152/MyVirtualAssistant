#!/bin/bash
set -e

# Navigate to the backend directory
cd "$(dirname "$0")"

# Create and activate virtual environment
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the FastAPI app
exec uvicorn main:app --reload 