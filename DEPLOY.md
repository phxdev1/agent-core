# Agent Core - Fly.io Deployment Guide

## Prerequisites

1. **Install Fly CLI**:
   ```bash
   # Windows (PowerShell)
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   
   # macOS
   brew install flyctl
   
   # Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Create Fly.io Account**:
   ```bash
   flyctl auth signup
   # Or login if you have an account
   flyctl auth login
   ```

## Quick Deploy

1. **Run the deployment script**:
   ```bash
   # Make the script executable (Linux/macOS)
   chmod +x deploy.sh
   
   # Run the deployment
   ./deploy.sh
   ```

## Manual Deployment

If you prefer to deploy manually or the script doesn't work:

1. **Create the Fly app**:
   ```bash
   flyctl launch --name agent-core --region phx --no-deploy
   ```

2. **Set environment secrets**:
   ```bash
   # OpenRouter API Key (REQUIRED)
   flyctl secrets set OPENROUTER_API_KEY=your_openrouter_key
   
   # Redis Configuration (REQUIRED)
   flyctl secrets set REDIS_HOST=your_redis_host
   flyctl secrets set REDIS_PORT=your_redis_port
   flyctl secrets set REDIS_USERNAME=your_redis_username
   flyctl secrets set REDIS_PASSWORD=your_redis_password
   
   # Search API Keys (OPTIONAL but recommended)
   flyctl secrets set SERPAPI_API_KEY=your_serpapi_key
   flyctl secrets set PERPLEXITY_API_KEY=your_perplexity_key
   flyctl secrets set TAVILY_API_KEY=your_tavily_key
   ```

3. **Deploy the application**:
   ```bash
   flyctl deploy
   ```

## Post-Deployment

### Access Your Application

- **Main API**: https://agent-core.fly.dev
- **API Documentation**: https://agent-core.fly.dev/docs
- **Health Check**: https://agent-core.fly.dev/health
- **Web Interface**: https://agent-core-autumn-pond-8052.fly.dev (port 443 or 8001 with TLS)

### Monitor Your Application

```bash
# View logs
flyctl logs

# View app status
flyctl status

# SSH into the container
flyctl ssh console

# Scale your application
flyctl scale count=2  # Run 2 instances
```

### Update Your Application

After making changes to your code:

```bash
# Deploy updates
flyctl deploy

# Deploy with more verbose output
flyctl deploy --verbose
```

## Configuration Details

### Current Configuration (fly.toml)

- **Region**: Phoenix, Arizona (phx)
- **Machine Size**: shared-cpu-1x with 512MB RAM
- **Services**:
  - Port 80/443: Main API
  - Port 8001: Web Interface
- **Health Checks**: Every 15 seconds on `/health`
- **Auto-scaling**: Min 1 machine, auto-start/stop disabled

### Scaling Options

To change the machine size:

```bash
# Upgrade to more memory
flyctl scale memory=1024  # 1GB RAM

# Upgrade to dedicated CPU
flyctl scale vm=dedicated-cpu-1x
```

## Troubleshooting

### Common Issues

1. **Deployment fails with "port already in use"**:
   - The start_production.py script manages both services
   - Make sure no local instances are hardcoded to production ports

2. **Redis connection errors**:
   - Verify your Redis Cloud credentials
   - Ensure the Redis instance allows connections from Fly.io IPs
   - Check that all Redis secrets are set correctly

3. **API keys not working**:
   - Use `flyctl secrets list` to see which secrets are set
   - Re-set any missing secrets with `flyctl secrets set KEY=value`

4. **Out of memory errors**:
   - Scale up memory: `flyctl scale memory=1024`
   - Or use a larger machine: `flyctl scale vm=shared-cpu-2x`

### Debug Commands

```bash
# Check app status
flyctl status

# View recent logs
flyctl logs -n 100

# SSH into container for debugging
flyctl ssh console

# Inside container, check running processes
ps aux

# Inside container, test API locally
curl http://localhost:8000/health

# View secrets (names only, not values)
flyctl secrets list
```

## Cost Estimation

With the current configuration (shared-cpu-1x, 512MB RAM):
- **Free Tier**: Fly.io offers free tier with 3 shared-cpu-1x VMs
- **Estimated Cost**: $0/month if within free tier
- **If exceeding free tier**: ~$5-10/month

## Security Notes

1. **Never commit secrets to Git**
2. **Use Fly secrets for all sensitive data**
3. **Redis should use TLS/SSL (Redis Cloud provides this)**
4. **API keys should have minimal required permissions**

## Support

- **Fly.io Status**: https://status.fly.io/
- **Fly.io Community**: https://community.fly.io/
- **Agent Core Issues**: Create an issue in your repository