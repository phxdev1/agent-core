#!/usr/bin/env python3
"""
Configuration loader with environment variable support
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

def load_config():
    """Load configuration from environment variables or config.json"""
    
    # Try to load config.json
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with environment variables
    config['openrouter_api_key'] = os.getenv('OPENROUTER_API_KEY', config.get('openrouter_api_key', ''))
    config['model'] = os.getenv('LLM_MODEL', config.get('model', 'mistralai/mistral-medium-3.1'))
    config['max_tokens'] = int(os.getenv('LLM_MAX_TOKENS', config.get('max_tokens', 2048)))
    config['temperature'] = float(os.getenv('LLM_TEMPERATURE', config.get('temperature', 0.7)))
    
    # Redis configuration
    config['redis'] = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'username': os.getenv('REDIS_USERNAME', 'default'),
        'password': os.getenv('REDIS_PASSWORD', ''),
        'db': int(os.getenv('REDIS_DB', 0))
    }
    
    # Research configuration
    config['serpapi_key'] = os.getenv('SERPAPI_KEY', config.get('serpapi_key', ''))
    config['arxiv_max_results'] = int(os.getenv('ARXIV_MAX_RESULTS', 5))
    config['scholar_max_results'] = int(os.getenv('SCHOLAR_MAX_RESULTS', 5))
    
    # System configuration
    config['memory_window_size'] = int(os.getenv('MEMORY_WINDOW_SIZE', 20))
    config['context_window_size'] = int(os.getenv('CONTEXT_WINDOW_SIZE', 4))
    config['enable_gpu'] = os.getenv('ENABLE_GPU', 'false').lower() == 'true'
    config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
    
    return config

# Export config
config = load_config()