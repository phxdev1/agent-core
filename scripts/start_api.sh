#!/bin/bash

# Agent Core API Startup Script

echo "========================================="
echo "     Agent Core API Server"
echo "========================================="

# Check for required environment variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY not set"
    echo "The agent will have limited functionality"
fi

# Parse command line arguments
MODE=${1:-dev}
PORT=${2:-8000}
WORKERS=${3:-1}

case $MODE in
    dev)
        echo "Starting in DEVELOPMENT mode..."
        echo "API will be available at: http://localhost:$PORT"
        echo "Documentation at: http://localhost:$PORT/docs"
        echo ""
        python api.py
        ;;
    
    prod)
        echo "Starting in PRODUCTION mode..."
        echo "API will be available at: http://localhost:$PORT"
        echo "Workers: $WORKERS"
        echo ""
        uvicorn api:app \
            --host 0.0.0.0 \
            --port $PORT \
            --workers $WORKERS \
            --log-level info \
            --access-log
        ;;
    
    docker)
        echo "Starting with Docker..."
        docker-compose -f docker-compose.api.yml up
        ;;
    
    test)
        echo "Running API tests..."
        python api_client_example.py
        ;;
    
    *)
        echo "Usage: ./start_api.sh [mode] [port] [workers]"
        echo ""
        echo "Modes:"
        echo "  dev    - Development mode with hot reload (default)"
        echo "  prod   - Production mode with multiple workers"
        echo "  docker - Run with Docker Compose"
        echo "  test   - Run client tests"
        echo ""
        echo "Examples:"
        echo "  ./start_api.sh dev"
        echo "  ./start_api.sh prod 8000 4"
        echo "  ./start_api.sh docker"
        exit 1
        ;;
esac