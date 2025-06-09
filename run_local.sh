#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Owlynn local development environment...${NC}"

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo -e "${YELLOW}Python 3.11 is not installed. Please install it first.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3.11 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${GREEN}Creating .env file...${NC}"
    cat > .env << EOL
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/owlynn

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-secret-key-here
EOL
    echo -e "${YELLOW}Please update the .env file with your specific configuration.${NC}"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Start Docker services
echo -e "${GREEN}Starting Docker services...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${GREEN}Waiting for services to be ready...${NC}"
sleep 5

# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:/Users/tim/Desktop/Owlynn

# Run the backend server
echo -e "${GREEN}Starting backend server...${NC}"
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Note: The script will keep running until you stop it with Ctrl+C 