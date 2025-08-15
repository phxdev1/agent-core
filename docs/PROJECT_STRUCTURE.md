# Agent Core Project Structure

## 📁 Organized Folder Structure

```
agent-core/
├── agents/                # Agent implementations
│   ├── __init__.py
│   ├── simple_agent.py    # Lightweight agent with lazy loading
│   └── agent_chat.py      # Full-featured agent
│
├── core/                  # Core system components
│   ├── __init__.py
│   ├── core_systems.py    # Unified core functionality
│   ├── unified_memory_system.py
│   ├── adaptive_personality_system.py
│   ├── context_intelligence_system.py
│   ├── temporal_context.py
│   └── prompt_evolution_system.py
│
├── knowledge/             # Knowledge and research systems
│   ├── __init__.py
│   ├── research_system.py
│   ├── knowledge_graph_enhanced.py
│   └── hybrid_search_system.py
│
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── config_loader.py
│   ├── redis_logger.py
│   ├── pdf_extractor.py
│   ├── pdf_summarizer.py
│   └── rq_worker_system.py
│
├── config/                # Configuration files
│   ├── config.json
│   └── .env.example
│
├── tests/                 # Test files
│   └── (test files)
│
├── docs/                  # Documentation
│   ├── README_CLEANUP.md
│   └── cleanup_imports.py
│
├── .runpod/              # RunPod deployment
│   └── hub.json
│
├── main.py               # Main entry point
├── runpod_handler.py     # RunPod serverless handler
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker compose config
└── README.md            # Project documentation
```

## Module Organization

### 🎯 agents/
Agent implementations with different complexity levels:
- `simple_agent.py` - Lightweight with lazy loading
- `agent_chat.py` - Full-featured with all systems

### 🧠 core/
Core system components:
- Memory management
- Personality systems
- Context intelligence
- Temporal awareness
- Prompt evolution

### 📚 knowledge/
Research and knowledge management:
- Research system with web/arxiv/scholar
- Knowledge graph
- Search functionality

### 🔧 utils/
Utility functions and helpers:
- Configuration loading
- Logging
- PDF processing
- Worker management

## Import Examples

```python
# Import from agents
from agents.simple_agent import SimpleAgent
from agents.agent_chat import ChatAgent

# Import from core
from core import UnifiedMemorySystem, StepType
from core.core_systems import UnifiedCore

# Import from knowledge
from knowledge import ResearchSystem
from knowledge.hybrid_search_system import SimpleSearchSystem

# Import from utils
from utils import config
from utils.redis_logger import get_redis_logger
```

## Running the System

```bash
# Default simple mode
python main.py

# Full featured mode
python main.py --mode full

# Core systems only
python main.py --mode core

# Without Redis
python main.py --no-redis

# Research mode
python main.py --research "your query here"
```

## Benefits of Organization

1. **Clear Separation**: Each module has a specific purpose
2. **Easy Navigation**: Find code quickly by category
3. **Better Testing**: Test modules independently
4. **Maintainability**: Changes isolated to specific areas
5. **Scalability**: Easy to add new agents or systems
6. **Import Clarity**: Clear import paths show dependencies