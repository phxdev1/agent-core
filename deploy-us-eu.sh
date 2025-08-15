#!/bin/bash

# US-EU deployment script for Agent Core
# Minimal 2-region setup for cost-effective global coverage

set -e

echo "============================================"
echo "Deploying Agent Core (US-EU Configuration)"
echo "============================================"

PRIMARY_REGION="iad"  # US East (Virginia) - good for US/EU latency
EU_REGION="lhr"       # London - covers EU well

# 1. Deploy API Service
echo ""
echo "Step 1: Deploying API Service..."
echo "---------------------------------"

# Create or update API app
if ! flyctl apps list | grep -q "agent-core-api"; then
    flyctl apps create agent-core-api --org personal
    echo "Created new app: agent-core-api"
fi

# Set secrets
echo "Setting API secrets..."
flyctl secrets set \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    SERPAPI_API_KEY="${SERPAPI_API_KEY}" \
    REDIS_HOST="${REDIS_HOST}" \
    REDIS_PORT="${REDIS_PORT}" \
    REDIS_PASSWORD="${REDIS_PASSWORD}" \
    --app agent-core-api

# Deploy to US
echo "Deploying API to US ($PRIMARY_REGION)..."
flyctl regions set $PRIMARY_REGION --app agent-core-api
flyctl deploy --config fly.api.toml --app agent-core-api

# Add EU region
echo "Adding API instance in EU ($EU_REGION)..."
flyctl regions add $EU_REGION --app agent-core-api
flyctl scale count 2 --app agent-core-api

# 2. Deploy Web Service
echo ""
echo "Step 2: Deploying Web Interface..."
echo "-----------------------------------"

if ! flyctl apps list | grep -q "agent-core-web"; then
    flyctl apps create agent-core-web --org personal
    echo "Created new app: agent-core-web"
fi

# Deploy to US
echo "Deploying Web to US ($PRIMARY_REGION)..."
flyctl regions set $PRIMARY_REGION --app agent-core-web
flyctl deploy --config fly.web.toml --app agent-core-web

# Add EU region
echo "Adding Web instance in EU ($EU_REGION)..."
flyctl regions add $EU_REGION --app agent-core-web
flyctl scale count 2 --app agent-core-web

# 3. Deploy Gateway
echo ""
echo "Step 3: Deploying API Gateway..."
echo "---------------------------------"

if ! flyctl apps list | grep -q "agent-core"; then
    flyctl apps create agent-core --org personal
    echo "Created new app: agent-core"
fi

# Deploy gateway
echo "Deploying Gateway..."
flyctl regions set $PRIMARY_REGION --app agent-core
flyctl deploy --config fly.gateway.toml --app agent-core

# Add EU region
echo "Adding Gateway instance in EU ($EU_REGION)..."
flyctl regions add $EU_REGION --app agent-core
flyctl scale count 2 --app agent-core

# 4. Configure autoscaling for cost optimization
echo ""
echo "Step 4: Configuring Autoscaling..."
echo "-----------------------------------"

# Conservative autoscaling to manage costs
flyctl autoscale set min=1 max=3 --app agent-core-api
flyctl autoscale set min=1 max=2 --app agent-core-web
flyctl autoscale set min=1 max=2 --app agent-core

# 5. Allocate IPs if needed
echo ""
echo "Step 5: Checking IP allocation..."
echo "----------------------------------"

if ! flyctl ips list --app agent-core | grep -q "v4"; then
    echo "Allocating IPv4 address..."
    flyctl ips allocate-v4 --app agent-core
fi

if ! flyctl ips list --app agent-core | grep -q "v6"; then
    echo "Allocating IPv6 address..."
    flyctl ips allocate-v6 --app agent-core
fi

# 6. Show deployment summary
echo ""
echo "============================================"
echo "US-EU Deployment Complete!"
echo "============================================"
echo ""
echo "🌐 Public URL: https://agent-core.fly.dev"
echo ""
echo "📍 Regional Distribution:"
echo "  • US East (Virginia): $PRIMARY_REGION"
echo "  • Europe (London): $EU_REGION"
echo ""
echo "💰 Estimated Monthly Cost: ~\$15-17"
echo "  • Gateway: 2 × \$0.97 = \$1.94"  # 128MB scratch
echo "  • API: 2 × \$5.70 = \$11.40"     # 1GB distroless
echo "  • Web: 2 × \$0.97 = \$1.94"      # 128MB scratch
echo "  • IPv4: \$2.00"
echo "  • Bandwidth: ~\$1.00"
echo "  • Total: ~\$18.28/month"
echo ""
echo "  With autoscaling: ~\$12-15/month"
echo ""
echo "✅ Features:"
echo "  • Automatic geo-routing"
echo "  • US/EU coverage"
echo "  • Failover between regions"
echo "  • Autoscaling enabled"
echo ""
echo "📊 Monitor deployment:"
echo "  flyctl status --app agent-core"
echo "  flyctl status --app agent-core-api"
echo "  flyctl status --app agent-core-web"
echo ""
echo "📈 Scale up when needed:"
echo "  flyctl regions add sin --app agent-core-api  # Add Asia"
echo "  flyctl scale count 3 --app agent-core-api"
echo "============================================"