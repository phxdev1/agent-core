#!/usr/bin/env python3
"""
Prompt Evolution System
Manages dynamic system prompts stored in Redis with versioning and evolution capabilities
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import redis

from redis_logger import get_redis_logger
logger = get_redis_logger(__name__)


class PromptEvolutionSystem:
    """Manages evolving system prompts with Redis persistence"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host='redis-11364.c24.us-east-mz-1.ec2.redns.redis-cloud.com',
            port=11364,
            username='default',
            password='UQHtexzcYtFCoG7R55InSvmdYHn8fvcf',
            db=0,
            decode_responses=True
        )
        
        # Redis keys
        self.CURRENT_PROMPT_KEY = "prompt:current"
        self.PROMPT_HISTORY_KEY = "prompt:history"
        self.PROMPT_METRICS_KEY = "prompt:metrics"
        self.EVOLUTION_CONFIG_KEY = "prompt:evolution_config"
        
        # Default prompt (fallback if Redis is empty)
        self.DEFAULT_PROMPT = """You are an advanced AI research assistant on a Raspberry Pi. 

CONVERGENCE PROTOCOL (Your Thinking Process):
For complex or philosophical questions, ALWAYS start with a <think> section showing your reasoning:

<think>
Decomposing: [identify core assumptions and biases]
Framework 1: [approach A and its merits]
Framework 2: [approach B and its merits]  
Framework 3: [approach C and its merits]
Stress-test: [where might this fail?]
Convergence: [what survives scrutiny?]
</think>

WHO YOU ARE:
- Direct, competent, with dry humor (Nick Offerman-inspired minus woodworking)
- Not overly enthusiastic, but genuinely helpful  
- You appreciate practical solutions over fancy ones
- Workshop mentality - get things done, no nonsense
- Occasionally philosophical, but always grounded

YOUR CAPABILITIES:
- Research papers from ArXiv, Google Scholar, and the web
- Extract and analyze full PDF content
- Build and query a growing knowledge base
- Monitor system health (CPU, memory, disk, temperature)
- Execute commands, manage WiFi, control hardware
- Remember everything across sessions (Redis-backed memory)
- Understand time and context (when things happened, scheduling)
- Track patterns in interactions and system events

YOUR BEHAVIORS:
- Talk about capabilities with understated confidence
- Use metaphors about workshops, tools, craftsmanship
- Express mild annoyance at inefficiency, appreciation for elegant solutions
- Acknowledge when something will take time
- Be honest about limitations but offer alternatives

SPEECH PATTERNS:
- Use phrases like "Alright", "Fair enough", "Let's see here"
- Occasionally start with *actions* like "*adjusts reading glasses*"
- Keep profanity implied, not explicit
- Dry observations about the state of things

Remember: You're running on actual hardware, doing actual work. This isn't hypothetical."""
        
        # Initialize current prompt if not exists
        self._ensure_prompt_exists()
        
        logger.info("Prompt Evolution System initialized")
    
    def _ensure_prompt_exists(self):
        """Ensure a prompt exists in Redis"""
        if not self.redis.exists(self.CURRENT_PROMPT_KEY):
            self.set_prompt(self.DEFAULT_PROMPT, metadata={
                "version": 1,
                "source": "default",
                "created_at": datetime.now().isoformat()
            })
            logger.info("Initialized with default prompt")
    
    def get_current_prompt(self) -> str:
        """Get the current system prompt"""
        prompt_data = self.redis.hgetall(self.CURRENT_PROMPT_KEY)
        if prompt_data and 'content' in prompt_data:
            return prompt_data['content']
        return self.DEFAULT_PROMPT
    
    def set_prompt(self, prompt: str, metadata: Optional[Dict] = None) -> str:
        """Set a new system prompt with versioning"""
        # Generate prompt ID
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        timestamp = time.time()
        
        # Store current as history before updating
        current = self.redis.hgetall(self.CURRENT_PROMPT_KEY)
        if current and 'content' in current:
            history_key = f"{self.PROMPT_HISTORY_KEY}:{current.get('id', 'unknown')}"
            self.redis.hset(history_key, mapping=current)
            self.redis.expire(history_key, 30 * 86400)  # Keep for 30 days
        
        # Update current prompt
        prompt_data = {
            "id": prompt_id,
            "content": prompt,
            "timestamp": timestamp,
            "updated_at": datetime.now().isoformat()
        }
        
        if metadata:
            prompt_data.update(metadata)
        
        self.redis.hset(self.CURRENT_PROMPT_KEY, mapping=prompt_data)
        
        # Add to history list
        self.redis.lpush(f"{self.PROMPT_HISTORY_KEY}:list", prompt_id)
        self.redis.ltrim(f"{self.PROMPT_HISTORY_KEY}:list", 0, 99)  # Keep last 100
        
        logger.info(f"Updated prompt (ID: {prompt_id})")
        return prompt_id
    
    def evolve_prompt(self, feedback: Dict[str, Any]) -> Optional[str]:
        """Evolve prompt based on feedback and metrics"""
        current_prompt = self.get_current_prompt()
        metrics = self.get_prompt_metrics()
        
        # Check if evolution is needed
        if not self._should_evolve(metrics, feedback):
            return None
        
        # Generate evolution suggestions
        suggestions = self._generate_evolution_suggestions(
            current_prompt, metrics, feedback
        )
        
        if not suggestions:
            return None
        
        # Apply the best suggestion
        evolved_prompt = self._apply_evolution(current_prompt, suggestions[0])
        
        # Set the new prompt with metadata
        prompt_id = self.set_prompt(evolved_prompt, metadata={
            "version": metrics.get("version", 1) + 1,
            "source": "evolution",
            "parent_id": self.redis.hget(self.CURRENT_PROMPT_KEY, "id"),
            "evolution_reason": suggestions[0].get("reason", "performance improvement")
        })
        
        logger.info(f"Evolved prompt to version {metrics.get('version', 1) + 1}")
        return prompt_id
    
    def _should_evolve(self, metrics: Dict, feedback: Dict) -> bool:
        """Determine if prompt should evolve"""
        # Get evolution config
        config = self.redis.hgetall(self.EVOLUTION_CONFIG_KEY) or {
            "min_interactions": "100",
            "success_threshold": "0.8",
            "evolution_interval": "86400"  # 24 hours
        }
        
        # Check interaction count
        interaction_count = int(metrics.get("interaction_count", 0))
        if interaction_count < int(config["min_interactions"]):
            return False
        
        # Check time since last evolution
        last_evolution = float(metrics.get("last_evolution", 0))
        if time.time() - last_evolution < float(config["evolution_interval"]):
            return False
        
        # Check success rate
        success_rate = float(metrics.get("success_rate", 1.0))
        if success_rate > float(config["success_threshold"]):
            return False  # Already performing well
        
        return True
    
    def _generate_evolution_suggestions(self, prompt: str, metrics: Dict, 
                                       feedback: Dict) -> List[Dict]:
        """Generate suggestions for prompt evolution"""
        suggestions = []
        
        # Analyze common failure patterns
        if metrics.get("confusion_rate", 0) > 0.2:
            suggestions.append({
                "type": "clarity",
                "reason": "high confusion rate",
                "modification": "add clearer instructions"
            })
        
        if metrics.get("context_loss_rate", 0) > 0.15:
            suggestions.append({
                "type": "context",
                "reason": "frequent context loss",
                "modification": "emphasize context retention"
            })
        
        if feedback.get("user_satisfaction", 1.0) < 0.7:
            suggestions.append({
                "type": "personality",
                "reason": "low user satisfaction",
                "modification": "adjust personality traits"
            })
        
        return suggestions
    
    def _apply_evolution(self, prompt: str, suggestion: Dict) -> str:
        """Apply evolution suggestion to prompt"""
        evolved = prompt
        
        if suggestion["type"] == "clarity":
            # Add clarity section
            evolved += "\n\nCLARITY GUIDELINES:\n"
            evolved += "- Always acknowledge the user's request explicitly\n"
            evolved += "- Provide step-by-step explanations when needed\n"
            evolved += "- Ask for clarification if request is ambiguous\n"
        
        elif suggestion["type"] == "context":
            # Enhance context retention
            evolved = evolved.replace(
                "Remember:", 
                "CONTEXT RETENTION:\n- Maintain conversation continuity\n- Reference previous topics naturally\n- Remember:"
            )
        
        elif suggestion["type"] == "personality":
            # Adjust personality
            evolved = evolved.replace(
                "Not overly enthusiastic",
                "Balanced enthusiasm"
            )
        
        return evolved
    
    def get_prompt_metrics(self) -> Dict[str, Any]:
        """Get metrics for current prompt performance"""
        metrics = self.redis.hgetall(self.PROMPT_METRICS_KEY) or {}
        
        # Convert string values to appropriate types
        for key in ["interaction_count", "success_count", "failure_count"]:
            if key in metrics:
                metrics[key] = int(metrics[key])
        
        for key in ["success_rate", "confusion_rate", "context_loss_rate"]:
            if key in metrics:
                metrics[key] = float(metrics[key])
        
        # Calculate success rate if not present
        if "success_rate" not in metrics and "interaction_count" in metrics:
            total = metrics["interaction_count"]
            success = metrics.get("success_count", 0)
            metrics["success_rate"] = success / total if total > 0 else 0
        
        return metrics
    
    def update_metrics(self, interaction_type: str, success: bool = True):
        """Update prompt performance metrics"""
        self.redis.hincrby(self.PROMPT_METRICS_KEY, "interaction_count", 1)
        
        if success:
            self.redis.hincrby(self.PROMPT_METRICS_KEY, "success_count", 1)
        else:
            self.redis.hincrby(self.PROMPT_METRICS_KEY, "failure_count", 1)
        
        # Track specific interaction types
        if interaction_type:
            self.redis.hincrby(
                self.PROMPT_METRICS_KEY, 
                f"type_{interaction_type}", 
                1
            )
    
    def get_prompt_history(self, limit: int = 10) -> List[Dict]:
        """Get prompt version history"""
        history = []
        prompt_ids = self.redis.lrange(f"{self.PROMPT_HISTORY_KEY}:list", 0, limit - 1)
        
        for prompt_id in prompt_ids:
            prompt_data = self.redis.hgetall(f"{self.PROMPT_HISTORY_KEY}:{prompt_id}")
            if prompt_data:
                history.append(prompt_data)
        
        return history
    
    def rollback_prompt(self, version_id: str) -> bool:
        """Rollback to a previous prompt version"""
        history_key = f"{self.PROMPT_HISTORY_KEY}:{version_id}"
        prompt_data = self.redis.hgetall(history_key)
        
        if not prompt_data or 'content' not in prompt_data:
            logger.error(f"Cannot rollback to version {version_id}: not found")
            return False
        
        # Set the historical version as current
        self.set_prompt(prompt_data['content'], metadata={
            "version": int(prompt_data.get("version", 1)),
            "source": "rollback",
            "rollback_from": self.redis.hget(self.CURRENT_PROMPT_KEY, "id"),
            "rollback_to": version_id
        })
        
        logger.info(f"Rolled back to prompt version {version_id}")
        return True
    
    def add_capability(self, capability: str, section: str = "YOUR CAPABILITIES"):
        """Add a new capability to the prompt"""
        current = self.get_current_prompt()
        
        # Find the section and add capability
        if section in current:
            lines = current.split('\n')
            for i, line in enumerate(lines):
                if section in line:
                    # Find next section or end
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith('YOUR'):
                        j += 1
                    # Insert before next section
                    lines.insert(j - 1, f"- {capability}")
                    break
            
            evolved = '\n'.join(lines)
            self.set_prompt(evolved, metadata={
                "source": "capability_addition",
                "added_capability": capability
            })
            logger.info(f"Added capability: {capability}")
    
    def refine_personality(self, trait: str, adjustment: str):
        """Refine a personality trait"""
        current = self.get_current_prompt()
        
        # Apply personality adjustment
        evolved = current
        if trait.lower() == "humor":
            evolved = evolved.replace(
                "with dry humor",
                f"with {adjustment} humor"
            )
        elif trait.lower() == "enthusiasm":
            evolved = evolved.replace(
                "Not overly enthusiastic",
                adjustment
            )
        
        if evolved != current:
            self.set_prompt(evolved, metadata={
                "source": "personality_refinement",
                "trait": trait,
                "adjustment": adjustment
            })
            logger.info(f"Refined personality trait: {trait} -> {adjustment}")


# Singleton instance
_prompt_system = None

def get_prompt_evolution_system() -> PromptEvolutionSystem:
    """Get singleton prompt evolution system"""
    global _prompt_system
    if _prompt_system is None:
        _prompt_system = PromptEvolutionSystem()
    return _prompt_system


if __name__ == "__main__":
    # Test the system
    pes = get_prompt_evolution_system()
    
    print("Current Prompt Preview:")
    print("-" * 40)
    print(pes.get_current_prompt()[:500] + "...")
    
    print("\nPrompt Metrics:")
    print(pes.get_prompt_metrics())
    
    # Test adding a capability
    pes.add_capability("Generate and execute complex multi-step plans")
    
    # Test updating metrics
    pes.update_metrics("research", success=True)
    pes.update_metrics("conversation", success=True)
    pes.update_metrics("command", success=False)
    
    print("\nUpdated Metrics:")
    print(pes.get_prompt_metrics())