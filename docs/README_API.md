# Agent Core API

A production-ready OpenAPI-compatible REST API for the Agent Core AI system, featuring natural conversation, research capabilities, and persistent memory.

## Features

- 🚀 **FastAPI** with automatic OpenAPI/Swagger documentation
- 💬 **REST & WebSocket** endpoints for flexible integration
- 🔐 **Authentication** via JWT tokens or API keys
- 🧠 **Multiple AI Agents** (natural, simple, fast modes)
- 🔍 **Research Capabilities** with arxiv, scholar, and web search
- 💾 **Persistent Memory** with Redis backend
- 🐳 **Docker Support** for easy deployment
- 📊 **Rate Limiting** and security features
- 🔄 **OpenAI-compatible** endpoint for drop-in replacement

## Quick Start

### 1. Install Dependencies

```bash
# Windows
install.bat basic

# Linux/Mac
./install.sh basic

# Or manually
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp config/.env.example .env

# Edit .env and add your OpenRouter API key
OPENROUTER_API_KEY=your-key-here
```

### 3. Run the API

```bash
# Development mode (with auto-reload)
python api.py

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker-compose -f docker-compose.api.yml up
```

### 4. Access Documentation

Open your browser to:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Chat Endpoints

#### `POST /chat`
Chat with the AI agent.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "agent_type": "natural"
  }'
```

#### `WS /ws/{session_id}`
WebSocket for real-time chat.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/my-session');
ws.send(JSON.stringify({message: 'Hello!'}));
```

### Research Endpoints

#### `POST /research`
Research a topic using multiple sources.

```python
import requests

response = requests.post(
    "http://localhost:8000/research",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "topic": "quantum computing",
        "max_results": 10
    }
)
```

### Session Management

#### `GET /sessions`
List active sessions.

#### `DELETE /sessions/{session_id}`
Delete a specific session.

### OpenAI Compatible

#### `POST /completions`
Drop-in replacement for OpenAI API.

```python
# Works with OpenAI client library
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="agent-core",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Authentication

### API Key Authentication

```python
headers = {
    "Authorization": "Bearer sk-user-your-api-key"
}
```

### JWT Authentication

```python
# Login to get token
response = requests.post("/auth/login", json={
    "username": "user",
    "password": "password"
})
token = response.json()["access_token"]

# Use token
headers = {
    "Authorization": f"Bearer {token}"
}
```

## Client Libraries

### Python Client

```python
from api_client_example import AgentCoreClient

client = AgentCoreClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Chat
response = await client.chat("Hello!")
print(response['response'])

# Research
results = await client.research("AI trends")
```

### JavaScript/TypeScript

```typescript
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

## Docker Deployment

### Basic Deployment

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

### Environment Variables

Create a `.env` file:

```env
# Required
OPENROUTER_API_KEY=your-openrouter-key
API_SECRET_KEY=your-secret-key-for-jwt

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
LLM_MODEL=mistralai/mistral-medium-3.1
LOG_LEVEL=INFO

# Research APIs (optional)
SERPAPI_KEY=your-serpapi-key
```

## Development

### Install Development Dependencies

```bash
# Install dev requirements
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Test specific module
pytest tests/test_api.py
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .

# Linting
flake8 .
```

## API Response Format

### Success Response

```json
{
  "response": "Hello! I'm doing well, thank you.",
  "session_id": "uuid-here",
  "timestamp": "2024-01-15T10:30:00",
  "metadata": {
    "agent_type": "natural",
    "response_time_ms": 250
  }
}
```

### Error Response

```json
{
  "detail": "Error description here"
}
```

## Rate Limiting

- Default: 60 requests per minute per user
- Configurable in `api_auth.py`
- Returns HTTP 429 when exceeded

## Security Best Practices

1. **Change default credentials** in production
2. **Use HTTPS** with SSL certificates
3. **Set strong `API_SECRET_KEY`** for JWT tokens
4. **Configure CORS** for your domains only
5. **Use environment variables** for secrets
6. **Enable rate limiting** appropriate for your use case
7. **Regular security updates** for dependencies

## Performance Optimization

- **Connection pooling** for Redis
- **Response caching** for common queries
- **Async/await** for all I/O operations
- **Multiple workers** in production
- **CDN** for static assets
- **Gzip compression** in reverse proxy

## Monitoring

- Health check: `GET /health`
- Metrics: Can integrate Prometheus
- Logs: `docker logs agent-core-api`

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **"Connection refused" to Redis**
   ```bash
   # Start Redis locally
   redis-server
   
   # Or use Docker
   docker run -p 6379:6379 redis
   ```

3. **"API key not valid"**
   - Check your OPENROUTER_API_KEY in .env
   - Ensure .env file is in the project root

4. **Slow responses**
   - Check your internet connection
   - Consider using the "fast" agent type
   - Enable response caching

## License

MIT License - See LICENSE file for details

## Support

- 📖 [API Documentation](http://localhost:8000/docs)
- 🐛 [Report Issues](https://github.com/yourusername/agent-core/issues)
- 💬 [Discussions](https://github.com/yourusername/agent-core/discussions)