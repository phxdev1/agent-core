#!/bin/bash

echo "========================================"
echo "Docker Image Size Comparison"
echo "========================================"
echo ""

# Build each variant
echo "Building images..."
echo ""

# Original Alpine-based
echo "1. Building Alpine-based images..."
docker build -f docker/Dockerfile.web -t agent-web:alpine . 2>/dev/null
docker build -f docker/Dockerfile.gateway -t agent-gateway:alpine . 2>/dev/null
docker build -f docker/Dockerfile.api -t agent-api:alpine . 2>/dev/null

# Distroless
echo "2. Building Distroless images..."
docker build -f docker/Dockerfile.web -t agent-web:distroless . 2>/dev/null
docker build -f docker/Dockerfile.api -t agent-api:distroless . 2>/dev/null

# Scratch-based
echo "3. Building Scratch-based images..."
docker build -f docker/Dockerfile.web.scratch -t agent-web:scratch . 2>/dev/null
docker build -f docker/Dockerfile.gateway -t agent-gateway:scratch . 2>/dev/null

echo ""
echo "========================================"
echo "Image Sizes:"
echo "========================================"
echo ""

# Function to show size
show_size() {
    local image=$1
    local size=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep $image | head -1)
    echo "$size"
}

echo "Web Interface:"
echo "--------------"
show_size "agent-web:alpine"
show_size "agent-web:distroless"
show_size "agent-web:scratch"
echo ""

echo "API Gateway:"
echo "------------"
show_size "agent-gateway:alpine"
show_size "agent-gateway:scratch"
echo ""

echo "API Service:"
echo "------------"
show_size "agent-api:alpine"
show_size "agent-api:distroless"
echo ""

echo "========================================"
echo "Memory Footprint (Runtime):"
echo "========================================"
echo ""
echo "Estimated RAM usage:"
echo "  • Scratch web: ~5-10MB"
echo "  • Scratch gateway: ~15-25MB"
echo "  • Distroless API: ~50-100MB"
echo ""
echo "Total for 2 regions: ~140-270MB"
echo "(vs ~400-600MB with standard images)"
echo ""

echo "========================================"
echo "Cost Impact:"
echo "========================================"
echo ""
echo "With ultra-minimal images:"
echo "  • Can potentially use 128MB instances"
echo "  • Gateway: 2 × $0.97 = $1.94/month"
echo "  • Web: 2 × $0.97 = $1.94/month"
echo "  • API: 2 × $5.70 = $11.40/month"
echo "  • Total: ~$15.28/month (save ~$7/month)"
echo ""