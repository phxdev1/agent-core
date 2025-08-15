# Agent Core

[![Runpod](https://api.runpod.io/badge/phxdev1/agent-core)](https://console.runpod.io/hub/phxdev1/agent-core)

A sophisticated AI assistant system with natural conversation, persistent memory, research capabilities, and a production-ready API.

## 🚀 Features

- **🤖 Natural Conversation**: Human-like responses without robotic preambles
- **🌐 OpenAPI REST API**: Full-featured API with WebSocket support
- **🧠 Persistent Memory**: Redis-backed conversation history
- **🔬 Research Integration**: ArXiv, Google Scholar, and web search
- **📚 Knowledge Base**: RAG system with hybrid search
- **🎭 Multiple Agents**: Natural, Simple, and Fast modes
- **🔐 Authentication**: JWT and API key support
- **🐳 Docker Ready**: Easy deployment with Docker Compose

## 📁 Project Structure

```
agent-core/
├── api/              # REST API implementation
├── agents/           # Agent implementations
├── core/             # Core systems (memory, personality, etc.)
├── knowledge/        # Research and knowledge systems
├── utils/            # Utilities and helpers
├── docker/           # Docker configurations
├── scripts/          # Installation and utility scripts
├── docs/             # Documentation
├── main.py           # CLI chat interface
├── run.py            # Main entry point
└── requirements.txt  # Python dependencies
```

## 🏃 Quick Start

### Option 1: Chat Interface

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
export OPENROUTER_API_KEY="your-key-here"

# Start chat
python run.py chat
```

### Option 2: API Server

```bash
# Start development server
python run.py api

# Start production server
python run.py api --production --port 8000 --workers 4

# Access documentation
open http://localhost:8000/docs
```

### Option 3: Docker

```bash
# Start with Docker Compose
docker-compose -f docker/docker-compose.api.yml up

# Access API
curl http://localhost:8000/health
```

## 💻 Installation

### Windows
```bash
scripts\install.bat basic
```

### Linux/Mac
```bash
./scripts/install.sh basic
```

### Manual
```bash
pip install -r requirements.txt
cp config/.env.example .env
# Edit .env with your API keys
```

## 🔧 Configuration

Create a `.env` file:

```env
# Required
OPENROUTER_API_KEY=your-openrouter-key

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
LLM_MODEL=mistralai/mistral-medium-3.1
API_SECRET_KEY=your-secret-for-jwt
```

## 📡 API Usage

### Python Client
```python
from api.api_client_example import AgentCoreClient

client = AgentCoreClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

response = await client.chat("Hello!")
print(response['response'])
```

### cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session-id');
ws.send(JSON.stringify({message: 'Hello!'}));
```

## 🤖 Agent Modes

- **Natural**: Conversational, human-like responses (default)
- **Simple**: Lightweight with minimal resource usage
- **Fast**: Optimized for sub-500ms responses

## 📚 Documentation

- [API Documentation](docs/API_DOCUMENTATION.md)
- [Natural Agent Design](docs/NATURAL_AGENT_DESIGN.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [RunPod Deployment](docs/RUNPOD_DEPLOY.md)

## 🧪 Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .

# Type checking
mypy .
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile.api -t agent-core .

# Run container
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your-key \
  agent-core
```

## 📊 Performance

- Chat response time: 200-500ms (natural mode)
- API latency: <100ms overhead
- Memory usage: ~200MB base
- Concurrent sessions: 100+ supported

## 🔒 Security

- JWT authentication for API
- API key support
- Rate limiting (60 req/min default)
- CORS configuration
- Input validation with Pydantic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🆘 Support

- 📖 [Interactive API Docs](http://localhost:8000/docs)
- 🐛 [Report Issues](https://github.com/yourusername/agent-core/issues)
- 💬 [Discussions](https://github.com/yourusername/agent-core/discussions)

## 🙏 Acknowledgments

- OpenRouter for LLM access
- FastAPI for the excellent web framework
- Redis for persistent storage
- The open-source community