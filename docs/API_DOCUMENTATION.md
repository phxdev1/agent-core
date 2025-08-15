# Agent Core API Documentation

## Overview

Agent Core provides an OpenAPI-compatible REST API with WebSocket support for natural AI conversations, research capabilities, and memory management.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENROUTER_API_KEY="your-api-key"
export API_SECRET_KEY="your-secret-key"
```

### 3. Run the API Server
```bash
# Development
python api.py

# Production
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker-compose -f docker-compose.api.yml up
```

### 4. Access Documentation
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI spec: http://localhost:8000/openapi.json

## Authentication

The API supports two authentication methods:

### 1. API Key Authentication
```python
headers = {
    "Authorization": "Bearer sk-user-your-api-key"
}
```

Default API keys for testing:
- Admin: `sk-admin-` + first 32 chars of sha256("admin-key")
- User: `sk-user-` + first 32 chars of sha256("user-key")

### 2. JWT Authentication
```python
# Login to get tokens
response = requests.post("/auth/login", json={
    "username": "user",
    "password": "password"
})
tokens = response.json()

# Use access token
headers = {
    "Authorization": f"Bearer {tokens['access_token']}"
}
```

## Endpoints

### Core Endpoints

#### POST /chat
Send a message to the AI agent.

**Request:**
```json
{
  "message": "Hello, how are you?",
  "session_id": "optional-session-id",
  "agent_type": "natural"
}
```

**Response:**
```json
{
  "response": "I'm doing well, thank you!",
  "session_id": "generated-session-id",
  "timestamp": "2024-01-15T10:30:00",
  "metadata": {
    "agent_type": "natural",
    "response_time_ms": 250
  }
}
```

#### POST /research
Research a topic using multiple sources.

**Request:**
```json
{
  "topic": "quantum computing",
  "max_results": 10,
  "sources": ["arxiv", "scholar", "web"]
}
```

#### GET /sessions
List all active chat sessions.

#### DELETE /sessions/{session_id}
Delete a specific session.

### WebSocket Endpoint

#### WS /ws/{session_id}
Real-time chat with streaming responses.

**Client sends:**
```json
{
  "message": "Tell me a story",
  "agent_type": "natural"
}
```

**Server responds:**
```json
{
  "response": "Once upon a time...",
  "timestamp": "2024-01-15T10:30:00",
  "session_id": "your-session-id"
}
```

### OpenAI-Compatible Endpoint

#### POST /completions
Compatible with OpenAI's chat completions format.

**Request:**
```json
{
  "model": "agent-core",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ]
}
```

## Client Examples

### Python Client
```python
from api_client_example import AgentCoreClient

# Initialize client
client = AgentCoreClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Chat
response = await client.chat("Hello!")
print(response['response'])

# Research
results = await client.research("AI trends", max_results=5)
```

### cURL Examples
```bash
# Chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Health check
curl http://localhost:8000/health
```

### JavaScript/TypeScript
```javascript
// Using fetch
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Hello!',
    agent_type: 'natural'
  })
});

const data = await response.json();
console.log(data.response);
```

### WebSocket Client
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/my-session');

ws.onopen = () => {
  ws.send(JSON.stringify({
    message: 'Hello!',
    agent_type: 'natural'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent:', data.response);
};
```

## Rate Limiting

- Default: 60 requests per minute per user
- Can be configured in `api_auth.py`
- Returns 429 status when exceeded

## Error Handling

All errors return consistent JSON format:
```json
{
  "detail": "Error description"
}
```

Common status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Docker Deployment

### Development
```bash
docker-compose -f docker-compose.api.yml up
```

### Production with Nginx
```bash
docker-compose -f docker-compose.api.yml --profile production up
```

### With Background Workers
```bash
docker-compose -f docker-compose.api.yml --profile with-worker up
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `API_SECRET_KEY` | JWT secret key | Required for production |
| `REDIS_HOST` | Redis hostname | localhost |
| `REDIS_PORT` | Redis port | 6379 |
| `LLM_MODEL` | Language model | mistralai/mistral-medium-3.1 |
| `LOG_LEVEL` | Logging level | INFO |

## Security Considerations

1. **Change default credentials** in production
2. **Use HTTPS** with proper SSL certificates
3. **Set strong API_SECRET_KEY** for JWT
4. **Implement rate limiting** per your needs
5. **Use environment variables** for sensitive data
6. **Enable CORS** only for trusted origins
7. **Regular security updates** for dependencies

## Performance Tips

1. Use **connection pooling** for database/Redis
2. Enable **response caching** for common queries
3. Use **async/await** for all I/O operations
4. Deploy with **multiple workers** in production
5. Use **CDN** for static assets
6. Enable **gzip compression** in Nginx

## Monitoring

- Health endpoint: `/health`
- Metrics can be added using Prometheus
- Logs available in Docker: `docker logs agent-core-api`

## Support

For issues or questions:
- Check the [API docs](http://localhost:8000/docs)
- Review error messages in logs
- Open an issue on GitHub