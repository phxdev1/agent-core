# RunPod Deployment Guide

Deploy Agent Core as a serverless endpoint on RunPod.

## Quick Deploy

### 1. Using RunPod CLI

```bash
# Install RunPod CLI
pip install runpodctl

# Login to RunPod
runpodctl login

# Deploy from this repository
runpodctl deploy --name agent-core --gpu-type NONE
```

### 2. Using RunPod Console

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select "Custom Container"
4. Enter GitHub URL: `https://github.com/phxdev1/agent-core`
5. Set environment variables:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `SERPAPI_KEY`: (Optional) Your SerpAPI key
6. Click "Deploy"

## Configuration

### Environment Variables

Required:
- `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM access

Optional:
- `SERPAPI_KEY`: For web search capabilities
- `LLM_MODEL`: Model to use (default: mistralai/mistral-medium-3.1)
- `LLM_MAX_TOKENS`: Max tokens per response (default: 2048)
- `MEMORY_WINDOW_SIZE`: Conversation history size (default: 20)

### Resource Requirements

Minimum:
- 2 vCPU
- 4 GB RAM
- 10 GB storage

Recommended:
- 4 vCPU
- 8 GB RAM
- 20 GB storage

## API Usage

### Request Format

```json
{
  "input": {
    "message": "Your message here",
    "session_id": "unique-session-id",
    "mode": "chat",
    "context_window": 4
  }
}
```

### Response Format

```json
{
  "response": "Agent's response",
  "session_id": "unique-session-id",
  "metadata": {
    "processing_time": 1.23,
    "tokens_used": 245,
    "confidence": 0.95
  },
  "status": "success"
}
```

### Example cURL Request

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "message": "Hello, how are you?",
      "session_id": "user123"
    }
  }'
```

## Modes

- **chat**: Regular conversation mode
- **research**: Triggers research capabilities (ArXiv, Scholar)
- **command**: System command mode (if enabled)

## Session Management

Each `session_id` maintains its own conversation history:
- Use the same `session_id` for continuous conversations
- Different `session_id` values create separate conversation threads
- History persists across serverless invocations

## Monitoring

### Health Check

```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/health
```

Returns:
```json
{
  "status": "healthy",
  "redis": "connected",
  "agent": "ready",
  "model": "mistralai/mistral-medium-3.1"
}
```

### Logs

View logs in RunPod console or via CLI:
```bash
runpodctl logs YOUR_ENDPOINT_ID
```

## Cost Optimization

1. **Cold Starts**: First request may take 10-30s to initialize
2. **Keep Warm**: Set minimum workers to 1 to avoid cold starts
3. **Batch Requests**: Process multiple messages in one session
4. **Cache**: Redis caches research results and conversation history

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   - Increase `container_start_timeout` in hub.json
   - Check if Redis is starting properly

2. **Memory Issues**
   - Reduce `MEMORY_WINDOW_SIZE`
   - Clear old sessions periodically

3. **API Key Errors**
   - Verify `OPENROUTER_API_KEY` is set correctly
   - Check API key has sufficient credits

### Debug Mode

Set environment variable:
```
LOG_LEVEL=DEBUG
```

## Advanced Features

### Custom Models

Change the LLM model:
```json
{
  "env": {
    "LLM_MODEL": "anthropic/claude-3-opus"
  }
}
```

### Persistent Knowledge Base

The `/data` volume persists:
- Redis database
- Downloaded PDFs
- Knowledge base entries

### Scaling

RunPod automatically scales based on request volume:
- Min workers: 0 (scales to zero)
- Max workers: 10 (configurable)
- Scale-up time: ~5-10 seconds

## Support

- GitHub Issues: https://github.com/phxdev1/agent-core/issues
- RunPod Discord: https://discord.gg/runpod