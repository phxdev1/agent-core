#!/bin/bash

# Complete deployment script for Agent Core with API Gateway architecture

set -e

echo "==========================================="
echo "Deploying Agent Core with API Gateway"
echo "==========================================="

# 1. Deploy API Service (internal only)
echo ""
echo "Step 1: Deploying API Service..."
echo "-----------------------------------"
if ! flyctl apps list | grep -q "agent-core-api"; then
    flyctl apps create agent-core-api --org personal
fi

# Set API secrets
flyctl secrets set \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    SERPAPI_API_KEY="${SERPAPI_API_KEY}" \
    REDIS_HOST="${REDIS_HOST}" \
    REDIS_PORT="${REDIS_PORT}" \
    REDIS_PASSWORD="${REDIS_PASSWORD}" \
    --app agent-core-api

flyctl deploy --config fly.api.toml --app agent-core-api

# 2. Deploy Web Service (internal only)
echo ""
echo "Step 2: Deploying Web Interface..."
echo "-----------------------------------"
if ! flyctl apps list | grep -q "agent-core-web"; then
    flyctl apps create agent-core-web --org personal
fi

flyctl deploy --config fly.web.toml --app agent-core-web

# 3. Deploy API Gateway (public facing)
echo ""
echo "Step 3: Deploying API Gateway..."
echo "-----------------------------------"
if ! flyctl apps list | grep -q "agent-core"; then
    flyctl apps create agent-core --org personal
fi

flyctl deploy --config fly.gateway.toml --app agent-core

# 4. Optional: Scale services
echo ""
echo "Step 4: Scaling Configuration"
echo "-----------------------------------"
echo "Current scaling:"
echo "  API instances: $(flyctl scale show --app agent-core-api | grep 'VM Resources' || echo '1')"
echo "  Web instances: $(flyctl scale show --app agent-core-web | grep 'VM Resources' || echo '1')"
echo "  Gateway instances: $(flyctl scale show --app agent-core | grep 'VM Resources' || echo '1')"
echo ""
echo "To scale services, use:"
echo "  flyctl scale count N --app agent-core-api"
echo "  flyctl scale count N --app agent-core-web"
echo "  flyctl scale count N --app agent-core"

# 5. Show deployment info
echo ""
echo "==========================================="
echo "Deployment Complete!"
echo "==========================================="
echo ""
echo "Public URL: https://agent-core.fly.dev"
echo ""
echo "Routes:"
echo "  - API: https://agent-core.fly.dev/api/*"
echo "  - Docs: https://agent-core.fly.dev/docs"
echo "  - Web: https://agent-core.fly.dev/"
echo ""
echo "Internal services (not publicly accessible):"
echo "  - API: http://agent-core-api.internal:8000"
echo "  - Web: http://agent-core-web.internal:8001"
echo ""
echo "Monitor services:"
echo "  flyctl logs --app agent-core        # Gateway logs"
echo "  flyctl logs --app agent-core-api    # API logs"
echo "  flyctl logs --app agent-core-web    # Web logs"
echo ""
echo "Scale services:"
echo "  flyctl scale count 3 --app agent-core-api"
echo "  flyctl scale count 2 --app agent-core-web"
echo "==========================================="