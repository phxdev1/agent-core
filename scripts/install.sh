#!/bin/bash

# Installation script for Agent Core

echo "========================================="
echo "     Agent Core Installation"
echo "========================================="
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.8"

if [ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l) -eq 0 ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"
echo ""

# Parse arguments
MODE=${1:-basic}

case $MODE in
    basic)
        echo "Installing basic requirements..."
        pip install --user -r requirements.txt
        ;;
    
    dev)
        echo "Installing development requirements..."
        pip install --user -r requirements-dev.txt
        echo ""
        echo "Setting up pre-commit hooks..."
        pre-commit install
        ;;
    
    docker)
        echo "Checking Docker installation..."
        if ! command -v docker &> /dev/null; then
            echo "Error: Docker is not installed"
            echo "Please install Docker from: https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo "Error: Docker Compose is not installed"
            echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
            exit 1
        fi
        
        echo "Docker and Docker Compose are installed ✓"
        echo ""
        echo "Building Docker images..."
        docker-compose -f docker-compose.api.yml build
        ;;
    
    full)
        echo "Full installation..."
        
        # Install Python requirements
        echo "Installing Python packages..."
        pip install --user -r requirements.txt
        pip install --user -r requirements-dev.txt
        
        # Install system dependencies based on OS
        if [ "$OS" == "linux" ]; then
            echo ""
            echo "Installing system dependencies (may require sudo)..."
            sudo apt-get update
            sudo apt-get install -y redis-server
        elif [ "$OS" == "macos" ]; then
            echo ""
            echo "Installing system dependencies (requires Homebrew)..."
            if ! command -v brew &> /dev/null; then
                echo "Homebrew not found. Please install from: https://brew.sh"
                exit 1
            fi
            brew install redis
        fi
        
        # Create necessary directories
        echo ""
        echo "Creating directories..."
        mkdir -p data logs .env
        
        # Copy environment template if not exists
        if [ ! -f .env ]; then
            echo "Creating .env file from template..."
            cp config/.env.example .env
            echo "Please edit .env file with your API keys"
        fi
        ;;
    
    test)
        echo "Running installation tests..."
        
        # Test imports
        python -c "import fastapi; print('✓ FastAPI installed')"
        python -c "import uvicorn; print('✓ Uvicorn installed')"
        python -c "import aiohttp; print('✓ Aiohttp installed')"
        python -c "import redis; print('✓ Redis-py installed')"
        python -c "import openai; print('✓ OpenAI installed')"
        
        # Test API startup
        echo ""
        echo "Testing API startup..."
        timeout 5 python -c "from api import app; print('✓ API imports successfully')" || true
        ;;
    
    *)
        echo "Usage: ./install.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  basic  - Install basic requirements only (default)"
        echo "  dev    - Install development requirements"
        echo "  docker - Build Docker images"
        echo "  full   - Full installation with system dependencies"
        echo "  test   - Test installation"
        echo ""
        echo "Examples:"
        echo "  ./install.sh basic"
        echo "  ./install.sh dev"
        echo "  ./install.sh full"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "     Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Set your OPENROUTER_API_KEY in .env file"
echo "2. Run the API: python api.py"
echo "3. Visit docs: http://localhost:8000/docs"
echo ""