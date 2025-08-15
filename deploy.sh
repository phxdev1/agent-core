#!/bin/bash

# Fly.io Deployment Script for Agent Core
# This script sets up and deploys the application to Fly.io

echo "========================================="
echo "Agent Core - Fly.io Deployment"
echo "========================================="

# Check if fly CLI is installed
if ! command -v flyctl &> /dev/null; then
    echo "Error: Fly CLI is not installed."
    echo "Please install it from: https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "Not logged in to Fly.io. Please log in:"
    flyctl auth login
fi

# Launch the app if it doesn't exist
echo ""
echo "Checking if app exists..."
if ! flyctl apps list | grep -q "agent-core"; then
    echo "App doesn't exist. Creating it..."
    flyctl launch --name agent-core --region phx --no-deploy
else
    echo "App 'agent-core' already exists."
fi

# Set secrets from .env file
echo ""
echo "Setting environment secrets..."

# Function to set a secret
set_secret() {
    local key=$1
    local value=$2
    if [ -n "$value" ]; then
        echo "Setting $key..."
        echo "$value" | flyctl secrets set "$key"=-
    fi
}

# Read .env file and set secrets
if [ -f .env ]; then
    # OpenRouter API Key
    OPENROUTER_API_KEY=$(grep "^OPENROUTER_API_KEY=" .env | cut -d '=' -f2-)
    set_secret "OPENROUTER_API_KEY" "$OPENROUTER_API_KEY"
    
    # Redis Configuration
    REDIS_HOST=$(grep "^REDIS_HOST=" .env | cut -d '=' -f2-)
    REDIS_PORT=$(grep "^REDIS_PORT=" .env | cut -d '=' -f2-)
    REDIS_USERNAME=$(grep "^REDIS_USERNAME=" .env | cut -d '=' -f2-)
    REDIS_PASSWORD=$(grep "^REDIS_PASSWORD=" .env | cut -d '=' -f2-)
    
    set_secret "REDIS_HOST" "$REDIS_HOST"
    set_secret "REDIS_PORT" "$REDIS_PORT"
    set_secret "REDIS_USERNAME" "$REDIS_USERNAME"
    set_secret "REDIS_PASSWORD" "$REDIS_PASSWORD"
    
    # Search API Keys
    SERPAPI_API_KEY=$(grep "^SERPAPI_API_KEY=" .env | cut -d '=' -f2-)
    PERPLEXITY_API_KEY=$(grep "^PERPLEXITY_API_KEY=" .env | cut -d '=' -f2-)
    TAVILY_API_KEY=$(grep "^TAVILY_API_KEY=" .env | cut -d '=' -f2-)
    
    set_secret "SERPAPI_API_KEY" "$SERPAPI_API_KEY"
    set_secret "PERPLEXITY_API_KEY" "$PERPLEXITY_API_KEY"
    set_secret "TAVILY_API_KEY" "$TAVILY_API_KEY"
    
    echo ""
    echo "All secrets have been set."
else
    echo "Warning: .env file not found. Please set secrets manually:"
    echo "  flyctl secrets set OPENROUTER_API_KEY=your_key"
    echo "  flyctl secrets set REDIS_HOST=your_host"
    echo "  flyctl secrets set REDIS_PORT=your_port"
    echo "  flyctl secrets set REDIS_USERNAME=your_username"
    echo "  flyctl secrets set REDIS_PASSWORD=your_password"
    echo "  flyctl secrets set SERPAPI_API_KEY=your_key"
    echo "  flyctl secrets set PERPLEXITY_API_KEY=your_key"
    echo "  flyctl secrets set TAVILY_API_KEY=your_key"
fi

# Deploy the application
echo ""
echo "========================================="
echo "Deploying to Fly.io..."
echo "========================================="
flyctl deploy

# Check deployment status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Deployment successful!"
    echo "========================================="
    echo ""
    echo "Your app is available at:"
    echo "  https://agent-core.fly.dev"
    echo ""
    echo "API endpoints:"
    echo "  https://agent-core.fly.dev/docs"
    echo "  https://agent-core.fly.dev/health"
    echo ""
    echo "Web interface:"
    echo "  https://agent-core-autumn-pond-8052.fly.dev"
    echo ""
    echo "To view logs:"
    echo "  flyctl logs"
    echo ""
    echo "To SSH into the container:"
    echo "  flyctl ssh console"
    echo ""
else
    echo ""
    echo "Deployment failed. Please check the errors above."
    echo "You can view more details with: flyctl logs"
    exit 1
fi