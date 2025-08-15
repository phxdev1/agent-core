#!/bin/bash

# Deploy script for Agent Core Web Interface

set -e

echo "Deploying Agent Core Web Interface to Fly.io..."

# Check if the app exists, if not create it
if ! flyctl apps list | grep -q "agent-core-web"; then
    echo "Creating new Fly.io app: agent-core-web"
    flyctl apps create agent-core-web --org personal
fi

# Deploy the web interface
echo "Deploying Web Interface..."
flyctl deploy --config fly.web.toml --app agent-core-web

# Show status
echo "Deployment complete!"
flyctl status --app agent-core-web

echo ""
echo "Web interface:"
echo "  https://agent-core-web.fly.dev"