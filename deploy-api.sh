#!/bin/bash

# Deploy script for Agent Core API

set -e

echo "Deploying Agent Core API to Fly.io..."

# Check if the app exists, if not create it
if ! flyctl apps list | grep -q "agent-core-api"; then
    echo "Creating new Fly.io app: agent-core-api"
    flyctl apps create agent-core-api --org personal
fi

# Set secrets from environment
echo "Setting secrets..."
flyctl secrets set \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    SERPAPI_API_KEY="${SERPAPI_API_KEY}" \
    REDIS_HOST="${REDIS_HOST}" \
    REDIS_PORT="${REDIS_PORT}" \
    REDIS_PASSWORD="${REDIS_PASSWORD}" \
    --app agent-core-api

# Deploy the API
echo "Deploying API..."
flyctl deploy --config fly.api.toml --app agent-core-api

# Show status
echo "Deployment complete!"
flyctl status --app agent-core-api

echo ""
echo "API endpoint:"
echo "  https://agent-core-api.fly.dev"
echo ""
echo "API documentation:"
echo "  https://agent-core-api.fly.dev/docs"