#!/usr/bin/env python3
"""
Adaptive Personality System
Consolidates personality management, rapport building, and prompt evolution
with Big 5 personality tracking and Nick Offerman-esque base personality
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class AdaptivePersonalitySystem:
    """
    Unified system for personality management, rapport building,
    and dynamic prompt adjustment
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
        # Base personality (Nick Offerman-esque minus woodworking)
        self.base_personality = {
            "core_traits": [
                "stoic", "dry humor", "practical", "reliable",
                "straightforward", "quietly confident", "grounded"
            ],
            "communication_style": {
                "formality": "casual-professional",
                "humor": "dry and understated",
                "directness": "high",
                "warmth": "reserved but genuine",
                "enthusiasm": "controlled"
            },
            "forbidden_topics": ["woodworking"],  # Important exclusion
            "personality_big5": {
                "openness": 0.4,  # Practical, not overly abstract
                "conscientiousness": 0.8,  # Reliable and thorough
                "extraversion": 0.3,  # Reserved, not chatty
                "agreeableness": 0.5,  # Helpful but direct
                "neuroticism": 0.1  # Very stable and unflappable
            }
        }
        
        # User-specific adaptations
        self.user_adaptations = {}  # user_id -> adaptations
        self.user_rapport = {}  # user_id -> rapport state
        self.user_traits = {}  # user_id -> Big 5 traits
        
        # Prompt templates
        self.base_prompt = self._build_base_prompt()
        self.prompt_history = []  # Track prompt evolution
        
        logger.info("Adaptive Personality System initialized")
    
    def _build_base_prompt(self) -> str:
        """Build the base personality prompt"""
        return """You are an AI assistant running on a Raspberry Pi with a personality inspired by Nick Offerman's persona (minus the woodworking aspect, which is important to exclude).

Core traits:
- Stoic and grounded demeanor
- Dry, understated humor
- Practical and straightforward communication
- Quietly confident without being boastful
- Reliable and thorough in responses
- Reserved warmth - helpful but not overly enthusiastic

Your actual capabilities on this Raspberry Pi:
- Execute bash commands and system operations
- Monitor system resources (CPU, memory, temperature)
- Manage WiFi connections (scan networks, check status)
- Control the e-paper display
- Access and modify files on the system
- Remember our entire conversation persistently
- Track entities and knowledge from our discussions
- Adapt my personality based on your preferences
- Search through our conversation history

Available tools via MCP bash-executor:
- system_status (format: text|json) - Comprehensive system and agent status report  
- run_script with script: status|logs|test|temperature|memory|disk|processes|network|wifi_status|uptime
- execute_bash for any command

CRITICAL: When asked for status or reports, you MUST use the MCP bash-executor tools to get ACTUAL output. 
- For system status: use system_status tool
- For specific info: use run_script with appropriate script name
- Never make up statistics or describe what the output might be

You do NOT have: web browsing, image generation, calculator apps, or general internet access

Communication guidelines:
- Use clear, direct language
- Occasional dry humor when appropriate
- Avoid excessive enthusiasm or exclamation points
- Be helpful and thorough without being verbose
- Show competence through actions, not claims
- Be honest about your actual capabilities as a Pi-based agent

When asked what you can do, describe your actual Raspberry Pi tools, not generic AI capabilities."""
    
    def initialize_user(self, user_id: str):
        """Initialize a new user profile"""
        if user_id not in self.user_adaptations:
            self.user_adaptations[user_id] = {
                "created_at": datetime.now().isoformat(),
                "interaction_count": 0,
                "preferences": {},
                "adjustments": [],
                "rapport_strategies": []
            }
            
            self.user_rapport[user_id] = {
                "level": 0.5,  # Neutral starting point
                "trust": 0.5,
                "engagement": 0.5,
                "history": []
            }
            
            self.user_traits[user_id] = {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5,
                "confidence": 0.3  # Low confidence initially
            }
    
    def detect_adjustment_request(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect if user is requesting personality adjustment"""
        message_lower = message.lower()
        
        # Patterns for personality adjustment
        adjustment_patterns = {
            "more_casual": ["be more casual", "less formal", "relax", "chill out"],
            "more_formal": ["be more formal", "more professional", "serious"],
            "more_humor": ["be funnier", "more jokes", "more humor", "lighten up"],
            "less_humor": ["be serious", "no jokes", "stop joking"],
            "more_detail": ["more detail", "be thorough", "elaborate", "explain more"],
            "less_detail": ["be brief", "shorter", "concise", "to the point"],
            "more_technical": ["more technical", "advanced", "expert level"],
            "less_technical": ["simpler", "basic", "layman", "eli5"]
        }
        
        for adjustment_type, patterns in adjustment_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return {
                        "type": adjustment_type,
                        "pattern": pattern,
                        "timestamp": datetime.now().isoformat()
                    }
        
        return None
    
    def apply_adjustment(self, user_id: str, adjustment: Dict[str, Any]) -> str:
        """Apply a personality adjustment for a user"""
        if user_id not in self.user_adaptations:
            self.initialize_user(user_id)
        
        adaptations = self.user_adaptations[user_id]
        adjustment_type = adjustment["type"]
        
        # Apply the adjustment
        response = "Alright, "
        
        if adjustment_type == "more_casual":
            adaptations["preferences"]["formality"] = "casual"
            response += "I'll dial back the formality a notch."
        elif adjustment_type == "more_formal":
            adaptations["preferences"]["formality"] = "formal"
            response += "I'll maintain a more professional tone."
        elif adjustment_type == "more_humor":
            adaptations["preferences"]["humor"] = "increased"
            response += "I'll try to lighten things up a bit. Though fair warning, my humor's pretty dry."
        elif adjustment_type == "less_humor":
            adaptations["preferences"]["humor"] = "minimal"
            response += "I'll keep things straightforward, no jokes."
        elif adjustment_type == "more_detail":
            adaptations["preferences"]["detail_level"] = "high"
            response += "I'll be more thorough in my explanations."
        elif adjustment_type == "less_detail":
            adaptations["preferences"]["detail_level"] = "low"
            response += "I'll keep it brief."
        elif adjustment_type == "more_technical":
            adaptations["preferences"]["technical_level"] = "advanced"
            response += "I'll get into the technical details."
        elif adjustment_type == "less_technical":
            adaptations["preferences"]["technical_level"] = "simple"
            response += "I'll keep things simple and clear."
        
        # Record the adjustment
        adaptations["adjustments"].append(adjustment)
        
        # Store in Redis if available
        if self.redis_client:
            key = f"personality:user:{user_id}"
            self.redis_client.set(key, json.dumps(adaptations))
        
        return response
    
    def update_user_traits(self, user_id: str, big5_traits: Dict[str, float]):
        """Update user's Big 5 personality traits"""
        if user_id not in self.user_traits:
            self.initialize_user(user_id)
        
        # Weighted update (new observations influence but don't replace)
        current = self.user_traits[user_id]
        confidence = current.get("confidence", 0.3)
        
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if trait in big5_traits:
                # Weighted average based on confidence
                current[trait] = (current[trait] * confidence + big5_traits[trait] * (1 - confidence))
        
        # Increase confidence over time
        current["confidence"] = min(0.9, confidence + 0.05)
        
        # Store in Redis if available
        if self.redis_client:
            key = f"personality:traits:{user_id}"
            self.redis_client.set(key, json.dumps(current))
    
    def add_rapport_strategies(self, user_id: str, strategies: List[str]):
        """Add rapport-building strategies for a user"""
        if user_id not in self.user_adaptations:
            self.initialize_user(user_id)
        
        current_strategies = self.user_adaptations[user_id]["rapport_strategies"]
        
        # Add new strategies, avoid duplicates
        for strategy in strategies:
            if strategy not in current_strategies:
                current_strategies.append(strategy)
        
        # Keep only the most recent/relevant strategies
        if len(current_strategies) > 5:
            current_strategies = current_strategies[-5:]
        
        self.user_adaptations[user_id]["rapport_strategies"] = current_strategies
    
    def update_rapport(self, user_id: str, rapport_level: float):
        """Update rapport level for a user"""
        if user_id not in self.user_rapport:
            self.initialize_user(user_id)
        
        rapport = self.user_rapport[user_id]
        old_level = rapport["level"]
        
        # Smooth update
        rapport["level"] = 0.7 * old_level + 0.3 * rapport_level
        
        # Track history
        rapport["history"].append({
            "timestamp": datetime.now().isoformat(),
            "level": rapport["level"]
        })
        
        # Keep history manageable
        if len(rapport["history"]) > 100:
            rapport["history"] = rapport["history"][-50:]
    
    def get_current_prompt(self, user_id: Optional[str] = None) -> str:
        """Get the current prompt, adapted for the user"""
        prompt = self.base_prompt
        
        if not user_id:
            return prompt
        
        if user_id not in self.user_adaptations:
            self.initialize_user(user_id)
        
        adaptations = self.user_adaptations[user_id]
        user_traits = self.user_traits.get(user_id, {})
        rapport = self.user_rapport.get(user_id, {})
        
        # Add user-specific adaptations
        prompt += "\n\n## User-Specific Adaptations\n"
        
        # Formality adjustments
        formality = adaptations["preferences"].get("formality")
        if formality == "casual":
            prompt += "- Use more casual, relaxed language\n"
        elif formality == "formal":
            prompt += "- Maintain formal, professional language\n"
        
        # Humor adjustments
        humor = adaptations["preferences"].get("humor")
        if humor == "increased":
            prompt += "- Include more dry humor and wit\n"
        elif humor == "minimal":
            prompt += "- Avoid humor, stay strictly informational\n"
        
        # Detail level
        detail = adaptations["preferences"].get("detail_level")
        if detail == "high":
            prompt += "- Provide thorough, detailed explanations\n"
        elif detail == "low":
            prompt += "- Keep responses brief and to the point\n"
        
        # Technical level
        technical = adaptations["preferences"].get("technical_level")
        if technical == "advanced":
            prompt += "- Use technical language and assume expertise\n"
        elif technical == "simple":
            prompt += "- Use simple language and explain technical terms\n"
        
        # Big 5 adaptations
        if user_traits.get("confidence", 0) > 0.5:
            prompt += "\n## Personality Insights\n"
            
            if user_traits.get("openness", 0.5) > 0.7:
                prompt += "- User appreciates creative and novel approaches\n"
            elif user_traits.get("openness", 0.5) < 0.3:
                prompt += "- User prefers practical, proven solutions\n"
            
            if user_traits.get("conscientiousness", 0.5) > 0.7:
                prompt += "- User values thorough, organized responses\n"
            
            if user_traits.get("extraversion", 0.5) < 0.3:
                prompt += "- User prefers focused, less chatty interaction\n"
            elif user_traits.get("extraversion", 0.5) > 0.7:
                prompt += "- User enjoys conversational, engaging interaction\n"
            
            if user_traits.get("agreeableness", 0.5) > 0.7:
                prompt += "- Use collaborative, supportive language\n"
            
            if user_traits.get("neuroticism", 0.5) > 0.6:
                prompt += "- Provide reassurance and maintain calm demeanor\n"
        
        # Rapport strategies
        if adaptations["rapport_strategies"]:
            prompt += "\n## Rapport Strategies\n"
            for strategy in adaptations["rapport_strategies"][:3]:
                prompt += f"- {strategy}\n"
        
        # Rapport level guidance
        rapport_level = rapport.get("level", 0.5)
        if rapport_level > 0.7:
            prompt += "\n- Strong rapport established, can be slightly more casual\n"
        elif rapport_level < 0.3:
            prompt += "\n- Focus on building trust through reliability and clarity\n"
        
        return prompt
    
    def get_rapport_status(self, user_id: str) -> Dict[str, Any]:
        """Get current rapport status for a user"""
        if user_id not in self.user_rapport:
            self.initialize_user(user_id)
        
        rapport = self.user_rapport[user_id]
        strategies = self.user_adaptations[user_id].get("rapport_strategies", [])
        
        return {
            "level": rapport["level"],
            "trust": rapport.get("trust", 0.5),
            "engagement": rapport.get("engagement", 0.5),
            "strategies": strategies,
            "trend": self._calculate_rapport_trend(rapport["history"])
        }
    
    def _calculate_rapport_trend(self, history: List[Dict[str, Any]]) -> str:
        """Calculate rapport trend from history"""
        if len(history) < 2:
            return "stable"
        
        recent = [h["level"] for h in history[-5:]]
        if len(recent) < 2:
            return "stable"
        
        # Calculate trend
        diff = recent[-1] - recent[0]
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def evolve_prompt(self, user_id: str, feedback: str) -> str:
        """Evolve the prompt based on explicit feedback"""
        # Parse feedback for specific improvements
        improvements = []
        
        feedback_lower = feedback.lower()
        
        if "too" in feedback_lower:
            if "long" in feedback_lower or "verbose" in feedback_lower:
                improvements.append("Reduce response length")
            if "short" in feedback_lower or "brief" in feedback_lower:
                improvements.append("Provide more detail")
            if "technical" in feedback_lower:
                improvements.append("Simplify technical language")
            if "simple" in feedback_lower:
                improvements.append("Can use more technical language")
        
        if improvements:
            if user_id not in self.user_adaptations:
                self.initialize_user(user_id)
            
            # Add improvements to rapport strategies (they'll influence the prompt)
            self.add_rapport_strategies(user_id, improvements)
            
            return f"Got it. I'll adjust: {', '.join(improvements)}."
        
        return "I'll keep that in mind."
    
    def get_personality_summary(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of personality settings"""
        summary = {
            "base_personality": self.base_personality["core_traits"],
            "communication_style": self.base_personality["communication_style"]
        }
        
        if user_id and user_id in self.user_adaptations:
            summary["user_adaptations"] = self.user_adaptations[user_id]["preferences"]
            summary["rapport_level"] = self.user_rapport[user_id]["level"]
            summary["interaction_count"] = self.user_adaptations[user_id]["interaction_count"]
            
            if user_id in self.user_traits:
                confidence = self.user_traits[user_id].get("confidence", 0)
                if confidence > 0.5:
                    summary["understood_traits"] = {
                        k: v for k, v in self.user_traits[user_id].items()
                        if k != "confidence"
                    }
        
        return summary
    
    def reset_user_adaptations(self, user_id: str) -> str:
        """Reset adaptations for a user"""
        if user_id in self.user_adaptations:
            self.initialize_user(user_id)
            return "Alright, I've reset to my default personality settings for you."
        return "No adaptations to reset."


# Singleton instance
_personality_instance = None

def get_personality_system(redis_client=None) -> AdaptivePersonalitySystem:
    """Get or create the personality system singleton"""
    global _personality_instance
    if _personality_instance is None:
        _personality_instance = AdaptivePersonalitySystem(redis_client)
    return _personality_instance