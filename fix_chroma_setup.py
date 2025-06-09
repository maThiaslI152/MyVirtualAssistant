#!/usr/bin/env python3
"""
ChromaDB Setup Fix Script for Owlynn AI Assistant

This script:
1. Checks Docker setup
2. Tests ChromaDB connectivity
3. Verifies dependencies
4. Provides troubleshooting guidance
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_docker():
    """Check if Docker is running and accessible."""
    print("🔍 Checking Docker setup...")
    
    # Check if docker is installed
    success, _, _ = run_command("docker --version")
    if not success:
        print("❌ Docker is not installed or not in PATH")
        return False
    
    # Check if docker-compose is available
    success, _, _ = run_command("docker-compose --version")
    if not success:
        success, _, _ = run_command("docker compose version")
        if not success:
            print("❌ Docker Compose is not available")
            return False
    
    print("✅ Docker and Docker Compose are available")
    return True

def check_containers():
    """Check the status of Docker containers."""
    print("\n🔍 Checking container status...")
    
    success, output, _ = run_command("docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'")
    if success:
        print("Current container status:")
        print(output)
    
    # Check specifically for ChromaDB container
    success, output, _ = run_command("docker ps --filter name=owlynn-chroma --format '{{.Status}}'")
    if success and output.strip():
        print(f"✅ ChromaDB container status: {output.strip()}")
        return True
    else:
        print("❌ ChromaDB container is not running")
        return False

def start_chroma_container():
    """Start the ChromaDB container."""
    print("\n🚀 Starting ChromaDB container...")
    
    # Stop any existing container first
    run_command("docker-compose stop chroma")
    
    # Start ChromaDB container
    success, output, error = run_command("docker-compose up -d chroma")
    if success:
        print("✅ ChromaDB container started")
        # Wait a bit for the container to be ready
        print("⏳ Waiting for ChromaDB to be ready...")
        time.sleep(10)
        return True
    else:
        print(f"❌ Failed to start ChromaDB container: {error}")
        return False

def test_chroma_connectivity():
    """Test ChromaDB connectivity."""
    print("\n🔍 Testing ChromaDB connectivity...")
    
    # Test if we can import chromadb
    try:
        import chromadb
        print("✅ ChromaDB Python package is available")
    except ImportError:
        print("❌ ChromaDB Python package is not installed")
        print("   Run: pip install chromadb>=0.5.15")
        return False
    
    # Test HTTP connectivity
    try:
        import requests
        response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=10)
        if response.status_code == 200:
            print("✅ ChromaDB server is responding!")
        else:
            print(f"❌ ChromaDB server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to ChromaDB server: {e}")
        return False
    
    # Test client connection
    try:
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        collections = client.list_collections()
        print(f"✅ ChromaDB client connected successfully! Found {len(collections)} collections.")
        return True
    except Exception as e:
        print(f"❌ ChromaDB client connection failed: {e}")
        return False

def check_logs():
    """Check ChromaDB container logs for errors."""
    print("\n📋 Checking ChromaDB container logs...")
    
    success, output, _ = run_command("docker logs owlynn-chroma --tail 20")
    if success:
        print("Recent logs from ChromaDB container:")
        print("=" * 50)
        print(output)
        print("=" * 50)
    else:
        print("❌ Could not retrieve container logs")

def create_env_file():
    """Create a .env file if it doesn't exist."""
    env_file = Path("backend/.env")
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("📝 Creating .env file from template...")
    
    env_content = """# Owlynn Backend Configuration
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DATABASE_URL=postgresql://owlynn:owlynnpass@localhost:5432/owlynn

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# OpenAI Configuration (Required for LLM functionality)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here

# Security (Change these in production!)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-key-change-in-production

# CORS Settings
FRONTEND_URL=http://localhost:3000

# Logging
LOG_LEVEL=INFO
"""
    
    try:
        env_file.write_text(env_content)
        print("✅ Created .env file")
        print("⚠️  Remember to add your OPENAI_API_KEY to the .env file!")
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")

def main():
    """Main function to run all checks and fixes."""
    print("🦉 Owlynn AI Assistant - ChromaDB Setup Fix")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found. Please run this script from the project root.")
        return False
    
    # Create .env file if needed
    create_env_file()
    
    # Check Docker
    if not check_docker():
        return False
    
    # Check if containers are running
    if not check_containers():
        # Try to start ChromaDB container
        if not start_chroma_container():
            return False
    
    # Test connectivity
    if not test_chroma_connectivity():
        print("\n🔧 Troubleshooting steps:")
        print("1. Check container logs for errors")
        check_logs()
        print("\n2. Try restarting the ChromaDB container:")
        print("   docker-compose restart chroma")
        print("\n3. Check if port 8000 is available:")
        print("   lsof -i :8000")
        print("\n4. Try rebuilding the container:")
        print("   docker-compose down && docker-compose up -d chroma")
        return False
    
    print("\n🎉 ChromaDB setup is working correctly!")
    print("\nNext steps:")
    print("1. Set your OPENAI_API_KEY in backend/.env")
    print("2. Start all services: docker-compose up -d")
    print("3. Test the backend: python backend/test_chroma_connection.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 