#!/bin/bash

# Global multi-region deployment script for Agent Core

set -e

echo "=============================================="
echo "Deploying Agent Core Globally (Multi-Region)"
echo "=============================================="

# Regions to deploy
REGIONS=("phx" "lax" "sea" "ord" "iad" "lhr" "fra" "ams" "nrt" "sin" "syd")
PRIMARY_REGION="phx"

# 1. Deploy API Service to multiple regions
echo ""
echo "Step 1: Deploying API Service to Multiple Regions..."
echo "----------------------------------------------------"

if ! flyctl apps list | grep -q "agent-core-api"; then
    flyctl apps create agent-core-api --org personal
fi

# Set API secrets
echo "Setting API secrets..."
flyctl secrets set \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    SERPAPI_API_KEY="${SERPAPI_API_KEY}" \
    REDIS_HOST="${REDIS_HOST}" \
    REDIS_PORT="${REDIS_PORT}" \
    REDIS_PASSWORD="${REDIS_PASSWORD}" \
    --app agent-core-api

# Deploy API to primary region first
echo "Deploying API to primary region: $PRIMARY_REGION"
flyctl deploy --config fly.api.toml --app agent-core-api --region $PRIMARY_REGION

# Scale API to other regions
echo "Scaling API to additional regions..."
for region in "${REGIONS[@]}"; do
    if [ "$region" != "$PRIMARY_REGION" ]; then
        echo "  Adding API instance in $region..."
        flyctl regions add $region --app agent-core-api
        flyctl scale count 1 --app agent-core-api --region $region
    fi
done

# 2. Deploy Web Service to multiple regions
echo ""
echo "Step 2: Deploying Web Interface to Multiple Regions..."
echo "-------------------------------------------------------"

if ! flyctl apps list | grep -q "agent-core-web"; then
    flyctl apps create agent-core-web --org personal
fi

# Deploy Web to primary region
echo "Deploying Web to primary region: $PRIMARY_REGION"
flyctl deploy --config fly.web.toml --app agent-core-web --region $PRIMARY_REGION

# Scale Web to key regions (less than API for cost optimization)
WEB_REGIONS=("phx" "lhr" "sin")
for region in "${WEB_REGIONS[@]}"; do
    if [ "$region" != "$PRIMARY_REGION" ]; then
        echo "  Adding Web instance in $region..."
        flyctl regions add $region --app agent-core-web
        flyctl scale count 1 --app agent-core-web --region $region
    fi
done

# 3. Deploy API Gateway to all regions
echo ""
echo "Step 3: Deploying API Gateway to All Regions..."
echo "------------------------------------------------"

if ! flyctl apps list | grep -q "agent-core"; then
    flyctl apps create agent-core --org personal
fi

# Deploy Gateway with geo-routing config
echo "Deploying Gateway with geo-routing..."
flyctl deploy --config fly.multi-region.toml --app agent-core

# Scale Gateway to all regions
echo "Scaling Gateway to all regions..."
for region in "${REGIONS[@]}"; do
    echo "  Adding Gateway instance in $region..."
    flyctl regions add $region --app agent-core
    flyctl scale count 1 --app agent-core --region $region
done

# 4. Configure autoscaling
echo ""
echo "Step 4: Configuring Autoscaling..."
echo "-----------------------------------"

# Set autoscaling for API (scale based on load)
flyctl autoscale set min=1 max=5 --app agent-core-api

# Set autoscaling for Web (minimal scaling)
flyctl autoscale set min=1 max=2 --app agent-core-web

# Set autoscaling for Gateway
flyctl autoscale set min=1 max=3 --app agent-core

# 5. Show deployment summary
echo ""
echo "=============================================="
echo "Global Deployment Complete!"
echo "=============================================="
echo ""
echo "Primary URL: https://agent-core.fly.dev"
echo ""
echo "Regional Distribution:"
echo "----------------------"
echo "North America:"
echo "  - Phoenix (phx): Primary region"
echo "  - Los Angeles (lax), Seattle (sea)"
echo "  - Chicago (ord), Virginia (iad)"
echo ""
echo "Europe:"
echo "  - London (lhr), Frankfurt (fra)"
echo "  - Amsterdam (ams)"
echo ""
echo "Asia Pacific:"
echo "  - Tokyo (nrt), Singapore (sin)"
echo "  - Sydney (syd)"
echo ""
echo "Features:"
echo "  ✓ Automatic geo-routing (Anycast)"
echo "  ✓ Regional failover"
echo "  ✓ Edge caching"
echo "  ✓ Autoscaling per region"
echo "  ✓ Health monitoring"
echo ""
echo "Monitor global deployment:"
echo "  flyctl status --app agent-core"
echo "  flyctl regions list --app agent-core"
echo ""
echo "View metrics:"
echo "  flyctl dashboard --app agent-core"
echo ""
echo "Test regional routing:"
echo "  curl -I https://agent-core.fly.dev"
echo "  # Check X-Region header in response"
echo "=============================================="