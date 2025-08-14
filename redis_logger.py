#!/usr/bin/env python3
"""
Redis-based logging system
Stores logs in Redis with automatic rotation and tailing
No SD card thrashing, just memory-based logging with persistence
"""

import redis
import json
import logging
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import traceback

class RedisLogHandler(logging.Handler):
    """Logging handler that stores logs in Redis"""
    
    def __init__(self, 
                 redis_client: redis.Redis,
                 log_key: str = "agent:logs",
                 max_logs: int = 1000,
                 ttl_seconds: int = 86400):  # 24 hours default
        super().__init__()
        self.redis_client = redis_client
        self.log_key = log_key
        self.max_logs = max_logs
        self.ttl_seconds = ttl_seconds
        
    def emit(self, record):
        """Store log record in Redis"""
        try:
            # Format log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "message": self.format(record),
                "thread": record.thread,
                "process": record.process
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = traceback.format_exception(*record.exc_info)
            
            # Store in Redis list (newest at front)
            self.redis_client.lpush(self.log_key, json.dumps(log_entry))
            
            # Trim to max size (keep only most recent)
            self.redis_client.ltrim(self.log_key, 0, self.max_logs - 1)
            
            # Set TTL on the key
            self.redis_client.expire(self.log_key, self.ttl_seconds)
            
            # Also store in level-specific keys for filtering
            level_key = f"{self.log_key}:{record.levelname.lower()}"
            self.redis_client.lpush(level_key, json.dumps(log_entry))
            self.redis_client.ltrim(level_key, 0, 100)  # Keep less per level
            self.redis_client.expire(level_key, self.ttl_seconds)
            
        except Exception as e:
            # Fall back to stderr if Redis fails
            print(f"Redis logging failed: {e}", file=sys.stderr)


class RedisLogger:
    """Redis-based logging system with tailing and search"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.redis_available = False
        
        if redis_client is None:
            # Use hardcoded credentials
            self.redis_config = {
                'host': 'redis-11364.c24.us-east-mz-1.ec2.redns.redis-cloud.com',
                'port': 11364,
                'username': 'default',
                'password': 'UQHtexzcYtFCoG7R55InSvmdYHn8fvcf',
                'decode_responses': True,
                'socket_connect_timeout': 2,  # 2 second timeout
                'retry_on_timeout': False
            }
            try:
                self.redis_client = redis.Redis(**self.redis_config)
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
            except Exception as e:
                print(f"Redis not available for logging: {e}", file=sys.stderr)
                self.redis_client = None
                self.redis_available = False
        
        self.log_key = "agent:logs"
        
    def setup_logging(self, logger_name: str = None, level: int = logging.INFO) -> logging.Logger:
        """Setup a logger with Redis handler"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Only add Redis handler if Redis is available
        if self.redis_client and self.redis_available:
            redis_handler = RedisLogHandler(self.redis_client, f"{self.log_key}:{logger_name or 'root'}")
            redis_handler.setLevel(level)
            redis_handler.setFormatter(formatter)
            logger.addHandler(redis_handler)
        
        # Always add console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def tail(self, 
             n: int = 50, 
             level: Optional[str] = None,
             module: Optional[str] = None,
             follow: bool = False) -> List[Dict[str, Any]]:
        """Tail the logs like 'tail -f'"""
        # Find all log keys
        all_keys = self.redis_client.keys(f"{self.log_key}:*")
        
        # Filter to get the main log keys (not level-specific)
        main_keys = [k for k in all_keys if k.count(':') == 2 and not any(lvl in k for lvl in ['debug', 'info', 'warning', 'error', 'critical'])]
        
        if not main_keys and all_keys:
            # Use any available keys
            main_keys = [k for k in all_keys if not any(lvl in k for lvl in ['debug', 'info', 'warning', 'error', 'critical'])]
        
        if level and main_keys:
            # Try to find level-specific key
            key = f"{main_keys[0]}:{level.lower()}"
        elif main_keys:
            key = main_keys[0]
        else:
            key = self.log_key
        
        # Get last n logs
        raw_logs = self.redis_client.lrange(key, 0, n - 1)
        logs = []
        
        for log_str in raw_logs:
            try:
                log = json.loads(log_str)
                # Filter by module if specified
                if module and log.get('module') != module:
                    continue
                logs.append(log)
            except:
                continue
        
        return logs
    
    def search(self, 
               pattern: str,
               level: Optional[str] = None,
               limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs for pattern"""
        if level:
            key = f"{self.log_key}:{level.lower()}"
        else:
            key = self.log_key
        
        raw_logs = self.redis_client.lrange(key, 0, limit - 1)
        matches = []
        
        for log_str in raw_logs:
            try:
                log = json.loads(log_str)
                if pattern.lower() in log.get('message', '').lower():
                    matches.append(log)
            except:
                continue
        
        return matches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'total_logs': self.redis_client.llen(self.log_key),
            'levels': {}
        }
        
        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            level_key = f"{self.log_key}:{level}"
            count = self.redis_client.llen(level_key)
            if count > 0:
                stats['levels'][level] = count
        
        # Get memory usage for logs
        total_size = 0
        all_keys = self.redis_client.keys(f"{self.log_key}*")
        for key in all_keys:
            total_size += self.redis_client.memory_usage(key) or 0
        
        stats['memory_bytes'] = total_size
        stats['memory_human'] = f"{total_size / 1024:.2f} KB"
        
        return stats
    
    def clear_logs(self, level: Optional[str] = None):
        """Clear logs"""
        if level:
            self.redis_client.delete(f"{self.log_key}:{level.lower()}")
        else:
            # Clear all log keys
            keys = self.redis_client.keys(f"{self.log_key}*")
            if keys:
                self.redis_client.delete(*keys)
    
    def format_log(self, log: Dict[str, Any], verbose: bool = False) -> str:
        """Format a log entry for display"""
        if verbose:
            return json.dumps(log, indent=2)
        
        # Compact format
        timestamp = log.get('timestamp', '')[:19]  # Remove microseconds
        level = log.get('level', 'INFO').ljust(8)
        module = log.get('module', 'unknown')[:15].ljust(15)
        message = log.get('message', '')[:100]
        
        return f"{timestamp} {level} {module} {message}"
    
    def print_tail(self, n: int = 20, level: Optional[str] = None):
        """Print recent logs nicely formatted"""
        logs = self.tail(n, level)
        
        print(f"\n=== Last {n} logs" + (f" (level: {level})" if level else "") + " ===")
        for log in reversed(logs):  # Show oldest first
            print(self.format_log(log))
        
        stats = self.get_stats()
        print(f"\nTotal logs: {stats['total_logs']} | Memory: {stats['memory_human']}")
        print(f"Levels: {stats['levels']}")


# Convenience functions
def get_redis_logger(logger_name: str = None) -> logging.Logger:
    """Get a logger configured for Redis (falls back to console if Redis unavailable)"""
    try:
        rl = RedisLogger()
        return rl.setup_logging(logger_name)
    except Exception as e:
        # Fallback to basic console logging
        print(f"Warning: Redis logging unavailable, using console only: {e}", file=sys.stderr)
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger


def tail_logs(n: int = 20, level: Optional[str] = None):
    """Quick function to tail logs"""
    rl = RedisLogger()
    rl.print_tail(n, level)


def search_logs(pattern: str, limit: int = 50):
    """Quick function to search logs"""
    rl = RedisLogger()
    results = rl.search(pattern, limit=limit)
    print(f"\n=== Found {len(results)} matches for '{pattern}' ===")
    for log in results:
        print(rl.format_log(log))


if __name__ == "__main__":
    # Test the logger
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "tail":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            tail_logs(n)
        
        elif command == "search":
            if len(sys.argv) > 2:
                search_logs(sys.argv[2])
            else:
                print("Usage: redis_logger.py search <pattern>")
        
        elif command == "stats":
            rl = RedisLogger()
            stats = rl.get_stats()
            print(json.dumps(stats, indent=2))
        
        elif command == "clear":
            rl = RedisLogger()
            rl.clear_logs()
            print("Logs cleared")
        
        elif command == "test":
            # Test logging
            logger = get_redis_logger("test")
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            print("Test logs written. Use 'tail' to view.")
        
        else:
            print("Commands: tail [n], search <pattern>, stats, clear, test")
    
    else:
        print("Redis Logger - Logs stored in Redis, not on SD card")
        print("Commands: tail [n], search <pattern>, stats, clear, test")