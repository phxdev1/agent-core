#!/usr/bin/env python3
"""
Test Redis connection with the configured credentials
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from utils.config_loader import config
import redis

print("Testing Redis Connection")
print("=" * 40)

# Display configuration (without password)
print(f"Host: {config['redis']['host']}")
print(f"Port: {config['redis']['port']}")
print(f"Username: {config['redis'].get('username', 'default')}")
print(f"Password: {'*' * 8 if config['redis']['password'] else 'Not set'}")
print(f"DB: {config['redis']['db']}")
print()

# Test connection
try:
    print("Connecting to Redis...")
    
    # Create connection with username support
    client = redis.Redis(
        host=config['redis']['host'],
        port=config['redis']['port'],
        username=config['redis'].get('username', 'default'),
        password=config['redis']['password'],
        db=config['redis']['db'],
        decode_responses=True
    )
    
    # Test ping
    print("Testing ping...")
    result = client.ping()
    print(f"[OK] Ping successful: {result}")
    
    # Test set/get
    print("\nTesting set/get...")
    test_key = "agent:test:connection"
    test_value = "Connection successful!"
    
    client.set(test_key, test_value)
    print(f"[OK] Set '{test_key}' = '{test_value}'")
    
    retrieved = client.get(test_key)
    print(f"[OK] Get '{test_key}' = '{retrieved}'")
    
    # Clean up
    client.delete(test_key)
    print(f"[OK] Deleted test key")
    
    # Get some stats
    print("\nRedis Info:")
    info = client.info('server')
    print(f"  Version: {info.get('redis_version', 'Unknown')}")
    print(f"  Mode: {info.get('redis_mode', 'Unknown')}")
    
    # Check memory usage
    memory_info = client.info('memory')
    used_memory = memory_info.get('used_memory_human', 'Unknown')
    print(f"  Memory Used: {used_memory}")
    
    print("\n[OK] Redis connection successful!")
    
except redis.ConnectionError as e:
    print(f"\n[FAIL] Connection failed: {e}")
    print("\nPlease check:")
    print("1. Redis server is running")
    print("2. Connection details are correct")
    print("3. Firewall/network allows connection")
    
except redis.AuthenticationError as e:
    print(f"\n[FAIL] Authentication failed: {e}")
    print("\nPlease check:")
    print("1. Username and password are correct")
    print("2. User has proper permissions")
    
except Exception as e:
    print(f"\n[FAIL] Unexpected error: {e}")
    import traceback
    traceback.print_exc()