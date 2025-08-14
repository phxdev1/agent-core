# Agent Core

A sophisticated conversational AI agent with persistent memory, research capabilities, and self-evolving prompts.

## Features

- 🧠 **Persistent Memory**: Redis-backed conversation history that survives restarts
- 🔬 **Research Integration**: ArXiv, Google Scholar, and web search capabilities
- 📚 **RAG System**: Knowledge base with hybrid BM25 + vector search
- 📄 **PDF Processing**: Extract and summarize research papers
- 🔄 **Prompt Evolution**: Self-improving system prompts based on performance
- ⏰ **Temporal Awareness**: Understands time context and can schedule tasks
- 🎭 **Adaptive Personality**: Configurable personality system
- 🔍 **Multi-modal Search**: Combines semantic and keyword search

## Quick Start

### Prerequisites

- Python 3.8+
- Redis server (local or cloud)
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-core.git
cd agent-core
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and Redis configuration
```

4. Start Redis (if running locally):
```bash
docker run -d -p 6379:6379 redis:alpine
```

5. Run the agent:
```bash
python agent_chat.py
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key configurations:
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `LLM_MODEL`: Model to use (default: mistralai/mistral-medium-3.1)
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port

### Using Redis Cloud

For production, we recommend Redis Cloud:
1. Sign up at https://redis.com/cloud/
2. Create a database
3. Update `.env` with your Redis Cloud credentials

## Architecture

### Core Components

- **`agent_chat.py`**: Main conversational interface
- **`unified_memory_system.py`**: Tape-inspired memory with sliding windows
- **`research_system.py`**: Academic paper search and retrieval
- **`prompt_evolution_system.py`**: Dynamic prompt management
- **`temporal_context.py`**: Time-aware context and scheduling
- **`knowledge_graph_enhanced.py`**: Entity and relationship tracking
- **`hybrid_search_system.py`**: Combined BM25 + vector search

### Memory System

The agent uses a "tape" memory system inspired by recent AI research:
- Maintains a sliding window of recent conversation
- Persists to Redis for cross-session memory
- Automatically summarizes older conversations
- Supports semantic search across history

### Research Capabilities

- Search ArXiv for academic papers
- Query Google Scholar
- Extract and summarize PDFs
- Build a growing knowledge base
- RAG retrieval for informed responses

## Usage

### Basic Chat
```python
from agent_chat import ChatAgent

agent = ChatAgent()
response = await agent.process_message("Hello, how are you?")
```

### Research Mode
```python
# Agent automatically detects research requests
response = await agent.process_message("Research quantum computing papers from 2023")
```

### Background Workers

For research tasks, start the RQ workers:
```bash
python -m rq worker research default
```

## API Integration

The agent supports multiple LLM providers through OpenRouter:
- Claude (Anthropic)
- GPT-4 (OpenAI)  
- Mistral
- And many more

## Multi-line Input

The chat interface supports multi-line input:
- Use `'''` or `"""` for multi-line blocks
- Use `\` for line continuation
- Pipe input from files

## Prompt Management

View and manage the system prompt:
```
You: show prompt metrics
You: prompt history
You: prompt evolve
```

## Docker Support

```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  agent:
    build: .
    environment:
      - REDIS_HOST=redis
    env_file:
      - .env
    depends_on:
      - redis
    stdin_open: true
    tty: true

volumes:
  redis_data:
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Capabilities

1. Create a new module in the project
2. Import in `agent_chat.py`
3. Add initialization in `ChatAgent.__init__()`
4. Add handler in `process_message()`

## Troubleshooting

### Connection Issues
- Ensure Redis is running
- Check firewall settings
- Verify API keys are correct

### Memory Issues
- Reduce `MEMORY_WINDOW_SIZE` in `.env`
- Clear old Redis data: `redis-cli FLUSHDB`

### Performance
- Use local Redis for lower latency
- Reduce context window size
- Consider using smaller LLM models

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with inspiration from TapeAgents and other memory-augmented architectures
- Uses OpenRouter for flexible LLM access
- Redis for persistent, scalable storage