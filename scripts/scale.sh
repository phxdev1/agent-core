#!/bin/bash

# Scale services dynamically
echo "Current API instances:"
flyctl scale show -a agent-core-api

echo "Current Web instances:"
flyctl scale show -a agent-core-web

# Scale API to 3 instances
echo "Scaling API to 3 instances..."
flyctl scale count 3 -a agent-core-api

# Scale Web to 2 instances
echo "Scaling Web to 2 instances..."
flyctl scale count 2 -a agent-core-web

# The gateway automatically discovers all instances via DNS
echo "Testing service discovery..."
sleep 5

# Test that all instances are reachable
echo "Testing API health through gateway..."
for i in {1..5}; do
    curl -s https://agent-core-gateway.fly.dev/api/health | jq .
    sleep 1
done

echo "DNS resolution test:"
flyctl ssh console -a agent-core-gateway -C "nslookup agent-core-api.internal"