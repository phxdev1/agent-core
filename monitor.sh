#!/bin/bash

# Monitoring script for Agent Core deployment

echo "======================================"
echo "Agent Core Deployment Monitor"
echo "======================================"
echo ""

# Function to check app status
check_app() {
    local app_name=$1
    echo "📊 $app_name Status:"
    
    # Get instance count and regions
    flyctl status --app $app_name 2>/dev/null | grep -E "Instances|Region" | head -5
    
    # Get memory usage
    echo "  Memory usage:"
    flyctl ssh console --app $app_name --command "free -h" 2>/dev/null | grep "Mem:" || echo "    Unable to connect"
    
    echo ""
}

# Check each service
echo "🌐 Gateway (agent-core):"
check_app "agent-core"

echo "🔧 API Service (agent-core-api):"
check_app "agent-core-api"

echo "💻 Web Interface (agent-core-web):"
check_app "agent-core-web"

# Show current costs
echo "======================================"
echo "💰 Cost Estimates:"
echo "======================================"
echo ""
echo "Current Configuration (US-EU, 2 regions):"
echo "  • Gateway: 2 × 256MB = $3.88/month"
echo "  • API: 2 × 1GB = $11.40/month"
echo "  • Web: 2 × 256MB = $3.88/month"
echo "  • IPv4: $2.00/month"
echo "  • Est. Bandwidth: $1.00/month"
echo "  • TOTAL: ~$22.16/month"
echo ""
echo "With autoscaling (off-peak savings):"
echo "  • Estimated: ~$15-18/month"
echo ""

# Show scaling commands
echo "======================================"
echo "📈 Quick Scaling Commands:"
echo "======================================"
echo ""
echo "Scale up (add instances):"
echo "  flyctl scale count 3 --app agent-core-api"
echo "  flyctl scale count 3 --app agent-core-web"
echo ""
echo "Scale out (add regions):"
echo "  flyctl regions add sin --app agent-core-api  # Asia"
echo "  flyctl regions add lax --app agent-core-api  # US West"
echo ""
echo "Check logs:"
echo "  flyctl logs --app agent-core"
echo "  flyctl logs --app agent-core-api"
echo "  flyctl logs --app agent-core-web"
echo ""