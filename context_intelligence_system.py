#!/usr/bin/env python3
"""
Context Intelligence System
Consolidates all context processing, analysis, and intelligent summarization
including psychological profiling, semantic analysis, and rapport tracking.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass 
class ConversationIntelligence:
    """Comprehensive conversation analysis"""
    summary: str
    semantic_themes: List[str]
    emotional_trajectory: Dict[str, Any]
    linguistic_patterns: Dict[str, Any]
    psychological_profile: Dict[str, Any]
    interaction_style: Dict[str, Any]
    key_entities: List[str]
    unresolved_topics: List[str]
    user_preferences: Dict[str, Any]
    rapport_level: float
    big5_profile: Optional[Dict[str, float]] = None
    context_prompts: List[str] = None
    conversation_state: Dict[str, Any] = None


class ContextIntelligenceSystem:
    """
    Unified system for all context processing and analysis
    Combines psychological analysis, semantic understanding, rapport building,
    and intelligent summarization
    """
    
    def __init__(self):
        # Build analysis lexicons
        self.emotion_lexicon = self._build_emotion_lexicon()
        self.psych_indicators = self._build_psych_indicators()
        self.big5_indicators = self._build_big5_indicators()
        self.rapport_indicators = self._build_rapport_indicators()
        
        # Track interaction patterns
        self.interaction_history = defaultdict(list)
        self.user_profiles = {}
        self.rapport_states = {}
        
        logger.info("Context Intelligence System initialized")
    
    def _build_emotion_lexicon(self) -> Dict[str, List[str]]:
        """Build comprehensive emotion detection lexicon"""
        return {
            "joy": ["happy", "excited", "great", "wonderful", "fantastic", "love", "amazing", 
                   "excellent", "perfect", "delighted", "thrilled", "awesome"],
            "frustration": ["annoying", "frustrated", "stuck", "confused", "difficult", "problem", 
                          "issue", "struggling", "can't", "won't work", "broken", "failing"],
            "curiosity": ["how", "why", "what", "wonder", "curious", "interesting", "tell me",
                         "explain", "understand", "learn", "know", "explore"],
            "satisfaction": ["good", "nice", "perfect", "works", "solved", "fixed", "thanks",
                           "appreciate", "helpful", "great job", "exactly", "yes"],
            "concern": ["worried", "concerned", "afraid", "nervous", "unsure", "hesitant",
                       "uncertain", "doubt", "risky", "dangerous", "careful"],
            "confidence": ["sure", "certain", "know", "understand", "clear", "obvious",
                         "definitely", "absolutely", "of course", "confident"],
            "impatience": ["hurry", "quick", "fast", "now", "immediately", "asap", "urgent",
                         "rush", "speed up", "come on", "finally"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "pissed",
                     "upset", "rage", "hate", "terrible", "awful"]
        }
    
    def _build_psych_indicators(self) -> Dict[str, List[str]]:
        """Build psychological trait indicators"""
        return {
            "analytical": ["because", "therefore", "specifically", "exactly", "precisely", 
                         "logically", "consequently", "thus", "hence", "data", "evidence"],
            "intuitive": ["feel", "sense", "seems", "appears", "maybe", "probably", 
                        "guess", "hunch", "instinct", "vibe", "impression"],
            "detail_oriented": ["exactly", "specifically", "precise", "particular", "every", 
                              "all", "each", "individual", "meticulous", "thorough"],
            "big_picture": ["overall", "generally", "basically", "mainly", "primarily", 
                          "broadly", "holistic", "comprehensive", "general", "summary"],
            "assertive": ["need", "must", "should", "will", "want", "require", 
                        "demand", "insist", "expect", "necessary"],
            "collaborative": ["we", "us", "together", "help", "could", "might", 
                            "perhaps", "team", "collaborate", "share", "joint"],
            "independent": ["I", "my", "myself", "alone", "solo", "independently",
                          "individual", "personal", "self", "own"],
            "social": ["people", "everyone", "others", "team", "group", "community",
                      "society", "friends", "colleagues", "network"]
        }
    
    def _build_big5_indicators(self) -> Dict[str, Dict[str, List[str]]]:
        """Build Big 5 personality trait indicators"""
        return {
            "openness": {
                "high": ["creative", "imagine", "curious", "explore", "novel", "unique",
                        "interesting", "wonder", "abstract", "theoretical", "innovative",
                        "experiment", "unconventional", "artistic", "philosophical"],
                "low": ["practical", "traditional", "routine", "conventional", "simple",
                       "straightforward", "concrete", "realistic", "familiar", "proven"]
            },
            "conscientiousness": {
                "high": ["organized", "plan", "careful", "thorough", "precise", "systematic",
                        "responsible", "reliable", "efficient", "detailed", "meticulous",
                        "prepared", "structured", "disciplined", "goal"],
                "low": ["spontaneous", "flexible", "casual", "relaxed", "carefree",
                       "improvise", "adaptable", "easygoing", "informal", "loose"]
            },
            "extraversion": {
                "high": ["social", "talk", "energetic", "enthusiastic", "excited", "active",
                        "outgoing", "friendly", "talkative", "assertive", "bold",
                        "people", "group", "party", "fun"],
                "low": ["quiet", "alone", "solitary", "reserved", "private", "independent",
                       "thoughtful", "introspective", "calm", "peaceful", "solitude"]
            },
            "agreeableness": {
                "high": ["help", "kind", "cooperate", "trust", "empathy", "compassion",
                        "understanding", "supportive", "gentle", "considerate", "warm",
                        "collaborative", "harmony", "together", "share"],
                "low": ["compete", "challenge", "debate", "argue", "critical", "skeptical",
                       "independent", "self", "direct", "frank", "honest", "blunt"]
            },
            "neuroticism": {
                "high": ["worried", "anxious", "stressed", "nervous", "tense", "emotional",
                        "sensitive", "moody", "upset", "frustrated", "concerned",
                        "overwhelmed", "insecure", "uncertain", "fear"],
                "low": ["calm", "relaxed", "stable", "confident", "secure", "composed",
                       "steady", "resilient", "unworried", "peaceful", "serene"]
            }
        }
    
    def _build_rapport_indicators(self) -> Dict[str, List[str]]:
        """Build rapport level indicators"""
        return {
            "positive": ["thanks", "perfect", "great", "exactly", "yes", "agree",
                        "right", "correct", "appreciate", "helpful", "wonderful"],
            "negative": ["no", "wrong", "confused", "frustrated", "annoying", "bad",
                        "incorrect", "misunderstood", "disagree", "problem"],
            "engagement": ["tell me more", "interesting", "go on", "what else", "continue",
                         "elaborate", "explain", "how", "why", "really"],
            "disengagement": ["nevermind", "forget it", "whatever", "doesn't matter",
                            "skip", "move on", "stop", "enough", "boring"]
        }
    
    def analyze_conversation(self, 
                            steps: List[Dict[str, Any]],
                            user_id: Optional[str] = None) -> ConversationIntelligence:
        """
        Perform comprehensive analysis on conversation steps
        """
        # Separate messages by role
        user_messages = []
        agent_messages = []
        
        for step in steps:
            step_type = step.get("step_type", "")
            content = str(step.get("content", ""))
            
            if "user" in step_type.lower():
                user_messages.append(content)
            elif "agent" in step_type.lower():
                agent_messages.append(content)
        
        # Perform multi-dimensional analysis
        semantic_themes = self._extract_semantic_themes(user_messages + agent_messages)
        emotional_trajectory = self._analyze_emotional_trajectory(user_messages)
        linguistic_patterns = self._analyze_linguistic_patterns(user_messages)
        psychological_profile = self._build_psychological_profile(user_messages)
        big5_profile = self._analyze_big5_traits(user_messages)
        interaction_style = self._analyze_interaction_style(user_messages, agent_messages)
        key_entities = self._extract_key_entities(user_messages + agent_messages)
        unresolved_topics = self._identify_unresolved_topics(steps)
        user_preferences = self._extract_user_preferences(user_messages)
        rapport_level = self._calculate_rapport_level(steps)
        conversation_state = self._analyze_conversation_state(steps)
        
        # Generate context-specific prompts
        context_prompts = self._generate_context_prompts(
            emotional_trajectory,
            psychological_profile,
            big5_profile,
            interaction_style,
            rapport_level,
            unresolved_topics
        )
        
        # Create intelligent summary
        summary = self._create_intelligent_summary(
            user_messages,
            agent_messages,
            semantic_themes,
            emotional_trajectory,
            rapport_level
        )
        
        # Update user profile if user_id provided
        if user_id:
            self._update_user_profile(user_id, {
                "psychological": psychological_profile,
                "big5": big5_profile,
                "preferences": user_preferences,
                "rapport": rapport_level
            })
        
        return ConversationIntelligence(
            summary=summary,
            semantic_themes=semantic_themes,
            emotional_trajectory=emotional_trajectory,
            linguistic_patterns=linguistic_patterns,
            psychological_profile=psychological_profile,
            interaction_style=interaction_style,
            key_entities=key_entities,
            unresolved_topics=unresolved_topics,
            user_preferences=user_preferences,
            rapport_level=rapport_level,
            big5_profile=big5_profile,
            context_prompts=context_prompts,
            conversation_state=conversation_state
        )
    
    def _extract_semantic_themes(self, messages: List[str]) -> List[str]:
        """Extract semantic themes using advanced NLP"""
        all_text = " ".join(messages).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        # Filter stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                     "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                     "been", "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall", "can"}
        
        significant_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(significant_words)
        
        # Theme detection
        themes = []
        theme_groups = {
            "technical": ["code", "system", "error", "debug", "program", "software", "hardware",
                         "server", "database", "api", "function", "algorithm", "data"],
            "communication": ["tell", "explain", "show", "help", "understand", "clarify",
                            "describe", "discuss", "mention", "share", "express"],
            "problem_solving": ["fix", "solve", "issue", "problem", "solution", "resolve",
                              "troubleshoot", "debug", "repair", "correct", "improve"],
            "learning": ["learn", "teach", "know", "understand", "explain", "study",
                        "research", "discover", "explore", "investigate", "analyze"],
            "configuration": ["setup", "configure", "install", "setting", "option", "preference",
                            "customize", "adjust", "modify", "change", "update"],
            "analysis": ["analyze", "examine", "evaluate", "assess", "review", "inspect",
                        "investigate", "study", "observe", "measure", "test"]
        }
        
        for theme, keywords in theme_groups.items():
            theme_score = sum(word_freq.get(word, 0) for word in keywords)
            if theme_score > 2:
                themes.append((theme, theme_score))
        
        # Sort by score and add high-frequency terms
        themes.sort(key=lambda x: x[1], reverse=True)
        theme_names = [t[0] for t in themes[:3]]
        
        # Add specific high-frequency terms
        for word, count in word_freq.most_common(5):
            if count > 3 and not any(word in theme_groups.get(t, []) for t in theme_names if t in theme_groups):
                theme_names.append(f"focus:{word}")
        
        return theme_names[:5]
    
    def _analyze_emotional_trajectory(self, messages: List[str]) -> Dict[str, Any]:
        """Track emotional changes throughout conversation"""
        if not messages:
            return {"dominant": "neutral", "trend": "stable", "intensity": 0, "emotions": {}}
        
        trajectory = []
        emotion_scores = defaultdict(int)
        
        for msg in messages:
            msg_lower = msg.lower()
            msg_emotions = {}
            
            for emotion, keywords in self.emotion_lexicon.items():
                score = sum(1 for keyword in keywords if keyword in msg_lower)
                if score > 0:
                    msg_emotions[emotion] = score
                    emotion_scores[emotion] += score
            
            trajectory.append(msg_emotions or {"neutral": 1})
        
        # Analyze trend
        if len(trajectory) > 2:
            first_half_score = sum(sum(t.values()) for t in trajectory[:len(trajectory)//2])
            second_half_score = sum(sum(t.values()) for t in trajectory[len(trajectory)//2:])
            
            if second_half_score > first_half_score * 1.2:
                trend = "escalating"
            elif second_half_score < first_half_score * 0.8:
                trend = "calming"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Determine dominant emotion
        if emotion_scores:
            dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
            intensity = sum(emotion_scores.values()) / len(messages)
        else:
            dominant = "neutral"
            intensity = 0
        
        return {
            "dominant": dominant,
            "trend": trend,
            "intensity": min(intensity, 10),  # Cap at 10
            "emotions": dict(emotion_scores),
            "trajectory": trajectory
        }
    
    def _analyze_linguistic_patterns(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze linguistic patterns in communication"""
        if not messages:
            return {}
        
        patterns = {
            "avg_message_length": np.mean([len(m.split()) for m in messages]),
            "question_ratio": sum(1 for m in messages if "?" in m) / len(messages),
            "imperative_ratio": sum(1 for m in messages if any(
                m.lower().startswith(cmd) for cmd in ["do", "make", "run", "execute", "show", "tell"]
            )) / len(messages),
            "technical_language": sum(1 for m in messages if any(
                term in m.lower() for term in ["api", "code", "function", "variable", "error", "debug", "system"]
            )) / len(messages),
            "formality_level": self._assess_formality(messages),
            "complexity_level": self._assess_complexity(messages),
            "communication_style": self._assess_communication_style(messages)
        }
        
        return patterns
    
    def _assess_formality(self, messages: List[str]) -> str:
        """Assess formality level"""
        informal_indicators = ["gonna", "wanna", "yeah", "yep", "nope", "hey", "lol", "btw", "ur", "thx"]
        formal_indicators = ["please", "thank you", "would", "could", "kindly", "appreciate", "regards", "sincerely"]
        
        informal_score = sum(1 for m in messages for ind in informal_indicators if ind in m.lower())
        formal_score = sum(1 for m in messages for ind in formal_indicators if ind in m.lower())
        
        if formal_score > informal_score * 2:
            return "formal"
        elif informal_score > formal_score * 2:
            return "casual"
        else:
            return "neutral"
    
    def _assess_complexity(self, messages: List[str]) -> str:
        """Assess language complexity"""
        if not messages:
            return "simple"
        
        words = [w for m in messages for w in m.split()]
        if not words:
            return "simple"
        
        avg_word_length = np.mean([len(w) for w in words])
        avg_sentence_length = np.mean([len(m.split()) for m in messages])
        
        if avg_word_length > 6 and avg_sentence_length > 15:
            return "complex"
        elif avg_word_length < 4 and avg_sentence_length < 8:
            return "simple"
        else:
            return "moderate"
    
    def _assess_communication_style(self, messages: List[str]) -> str:
        """Determine communication style"""
        styles = {
            "directive": ["do", "must", "need", "should", "have to", "required"],
            "inquisitive": ["?", "how", "why", "what", "when", "where", "who"],
            "expressive": ["!", "feel", "think", "believe", "love", "hate"],
            "analytical": ["because", "therefore", "data", "analysis", "evidence"]
        }
        
        style_scores = {}
        for style, indicators in styles.items():
            score = sum(1 for m in messages for ind in indicators if ind in m.lower())
            style_scores[style] = score
        
        if style_scores:
            return max(style_scores.items(), key=lambda x: x[1])[0]
        return "neutral"
    
    def _build_psychological_profile(self, messages: List[str]) -> Dict[str, Any]:
        """Build detailed psychological profile"""
        profile_scores = defaultdict(int)
        
        for msg in messages:
            msg_lower = msg.lower()
            for trait, indicators in self.psych_indicators.items():
                score = sum(1 for ind in indicators if ind in msg_lower)
                profile_scores[trait] += score
        
        total = sum(profile_scores.values())
        if total == 0:
            return {"dominant_traits": ["balanced"], "profile_scores": {}}
        
        normalized = {k: v/total for k, v in profile_scores.items()}
        dominant_traits = [k for k, v in sorted(normalized.items(), key=lambda x: x[1], reverse=True) if v > 0.1][:3]
        
        return {
            "dominant_traits": dominant_traits or ["balanced"],
            "profile_scores": dict(normalized),
            "thinking_style": "analytical" if normalized.get("analytical", 0) > normalized.get("intuitive", 0) else "intuitive",
            "focus_type": "detail" if normalized.get("detail_oriented", 0) > normalized.get("big_picture", 0) else "big_picture",
            "interaction_preference": "collaborative" if normalized.get("collaborative", 0) > normalized.get("independent", 0) else "independent"
        }
    
    def _analyze_big5_traits(self, messages: List[str]) -> Dict[str, float]:
        """Analyze Big 5 personality traits"""
        trait_scores = {}
        
        for trait, indicators in self.big5_indicators.items():
            high_score = 0
            low_score = 0
            
            for msg in messages:
                msg_lower = msg.lower()
                high_score += sum(1 for word in indicators["high"] if word in msg_lower)
                low_score += sum(1 for word in indicators["low"] if word in msg_lower)
            
            # Normalize to 0-1 scale
            total = high_score + low_score
            if total > 0:
                trait_scores[trait] = 0.5 + (high_score - low_score) / (2 * total)
            else:
                trait_scores[trait] = 0.5
            
            # Clamp between 0 and 1
            trait_scores[trait] = max(0, min(1, trait_scores[trait]))
        
        return trait_scores
    
    def _analyze_interaction_style(self, user_messages: List[str], agent_messages: List[str]) -> Dict[str, Any]:
        """Analyze interaction dynamics"""
        user_avg_len = np.mean([len(m.split()) for m in user_messages]) if user_messages else 0
        agent_avg_len = np.mean([len(m.split()) for m in agent_messages]) if agent_messages else 0
        
        # Determine length pattern
        if user_avg_len > agent_avg_len * 1.5:
            length_pattern = "user_verbose"
        elif agent_avg_len > user_avg_len * 1.5:
            length_pattern = "agent_verbose"
        else:
            length_pattern = "balanced"
        
        # Determine tempo
        total_messages = len(user_messages) + len(agent_messages)
        if total_messages > 20:
            tempo = "rapid"
        elif total_messages > 10:
            tempo = "moderate"
        else:
            tempo = "slow"
        
        return {
            "length_pattern": length_pattern,
            "tempo": tempo,
            "turn_taking": "alternating" if abs(len(user_messages) - len(agent_messages)) <= 1 else "asymmetric",
            "engagement_level": "high" if len(user_messages) > 10 else "moderate" if len(user_messages) > 5 else "low",
            "user_message_avg": user_avg_len,
            "agent_message_avg": agent_avg_len
        }
    
    def _extract_key_entities(self, messages: List[str]) -> List[str]:
        """Extract key entities from conversation"""
        entities = []
        
        for msg in messages:
            # Proper nouns (capitalized words)
            caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', msg)
            entities.extend(caps)
            
            # File paths
            paths = re.findall(r'[/\\][\w/\\.-]+', msg)
            entities.extend(paths)
            
            # Code snippets
            code = re.findall(r'`([^`]+)`', msg)
            entities.extend(code)
            
            # URLs
            urls = re.findall(r'https?://[^\s]+', msg)
            entities.extend(urls)
            
            # Email addresses
            emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', msg)
            entities.extend(emails)
        
        # Return unique entities
        return list(set(entities))[:20]
    
    def _identify_unresolved_topics(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Identify unresolved or pending topics"""
        unresolved = []
        
        for i, step in enumerate(steps):
            if step.get("step_type") == "user_input":
                content = str(step.get("content", ""))
                
                # Check if it's a question
                if "?" in content:
                    # Look for answer in next few steps
                    answered = False
                    for j in range(i+1, min(i+5, len(steps))):
                        if steps[j].get("step_type") == "agent_response":
                            response = str(steps[j].get("content", ""))
                            # Simple heuristic: adequate response length
                            if len(response) > 50 and "don't know" not in response.lower():
                                answered = True
                                break
                    
                    if not answered:
                        unresolved.append(content[:100])
                
                # Check for error indicators
                if any(word in content.lower() for word in ["error", "failed", "broken", "doesn't work"]):
                    # Check if resolved in subsequent steps
                    resolved = False
                    for j in range(i+1, min(i+10, len(steps))):
                        if "fixed" in str(steps[j].get("content", "")).lower() or \
                           "solved" in str(steps[j].get("content", "")).lower():
                            resolved = True
                            break
                    
                    if not resolved:
                        unresolved.append(f"Issue: {content[:50]}")
        
        return unresolved[:5]
    
    def _extract_user_preferences(self, messages: List[str]) -> Dict[str, Any]:
        """Extract user preferences from conversation"""
        preferences = {}
        
        # Response length preference
        if any(word in msg.lower() for msg in messages for word in ["brief", "short", "quick", "concise"]):
            preferences["response_length"] = "brief"
        elif any(word in msg.lower() for msg in messages for word in ["detail", "explain", "elaborate", "thorough"]):
            preferences["response_length"] = "detailed"
        
        # Technical level
        if any(word in msg.lower() for msg in messages for word in ["simple", "basic", "easy", "layman"]):
            preferences["technical_level"] = "simple"
        elif any(word in msg.lower() for msg in messages for word in ["technical", "advanced", "expert", "detailed"]):
            preferences["technical_level"] = "advanced"
        
        # Interaction style
        if any(word in msg.lower() for msg in messages for word in ["step by step", "guide", "walk through"]):
            preferences["guidance_style"] = "step_by_step"
        elif any(word in msg.lower() for msg in messages for word in ["overview", "summary", "general"]):
            preferences["guidance_style"] = "overview"
        
        return preferences
    
    def _calculate_rapport_level(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate rapport level from interaction"""
        rapport_score = 0.5  # Start neutral
        
        for step in steps:
            content = str(step.get("content", "")).lower()
            
            # Positive indicators
            positive_count = sum(1 for word in self.rapport_indicators["positive"] if word in content)
            rapport_score += positive_count * 0.02
            
            # Negative indicators
            negative_count = sum(1 for word in self.rapport_indicators["negative"] if word in content)
            rapport_score -= negative_count * 0.03
            
            # Engagement indicators
            engagement_count = sum(1 for word in self.rapport_indicators["engagement"] if word in content)
            rapport_score += engagement_count * 0.01
            
            # Disengagement indicators
            disengagement_count = sum(1 for word in self.rapport_indicators["disengagement"] if word in content)
            rapport_score -= disengagement_count * 0.04
        
        # Clamp between 0 and 1
        return max(0, min(1, rapport_score))
    
    def _analyze_conversation_state(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current conversation state"""
        if not steps:
            return {"phase": "initial", "momentum": "starting"}
        
        # Determine conversation phase
        step_count = len(steps)
        if step_count < 3:
            phase = "initial"
        elif step_count < 10:
            phase = "exploration"
        elif step_count < 30:
            phase = "development"
        else:
            phase = "extended"
        
        # Analyze momentum
        recent_steps = steps[-5:] if len(steps) > 5 else steps
        recent_user_steps = [s for s in recent_steps if "user" in s.get("step_type", "").lower()]
        
        if len(recent_user_steps) >= 3:
            momentum = "active"
        elif len(recent_user_steps) >= 1:
            momentum = "moderate"
        else:
            momentum = "low"
        
        # Check for patterns
        patterns = []
        if any("?" in str(s.get("content", "")) for s in recent_steps):
            patterns.append("questioning")
        if any("error" in str(s.get("content", "")).lower() for s in recent_steps):
            patterns.append("troubleshooting")
        if any("thank" in str(s.get("content", "")).lower() for s in recent_steps):
            patterns.append("concluding")
        
        return {
            "phase": phase,
            "momentum": momentum,
            "patterns": patterns,
            "step_count": step_count
        }
    
    def _generate_context_prompts(self,
                                 emotional_trajectory: Dict[str, Any],
                                 psychological_profile: Dict[str, Any],
                                 big5_profile: Dict[str, float],
                                 interaction_style: Dict[str, Any],
                                 rapport_level: float,
                                 unresolved_topics: List[str]) -> List[str]:
        """Generate context-aware behavioral prompts"""
        prompts = []
        
        # Emotional adaptation
        emotion = emotional_trajectory.get("dominant", "neutral")
        if emotion == "frustration":
            prompts.append("User is frustrated. Be extra clear, solution-focused, and patient.")
        elif emotion == "curiosity":
            prompts.append("User is curious. Provide rich, educational responses with examples.")
        elif emotion == "impatience":
            prompts.append("User is impatient. Get to the point quickly and be concise.")
        elif emotion == "joy":
            prompts.append("User is in good spirits. Maintain positive energy while being helpful.")
        
        # Psychological adaptation
        dominant_traits = psychological_profile.get("dominant_traits", [])
        if "analytical" in dominant_traits:
            prompts.append("User prefers logical, systematic explanations with clear reasoning.")
        elif "intuitive" in dominant_traits:
            prompts.append("User responds well to examples and conceptual explanations.")
        
        if psychological_profile.get("thinking_style") == "analytical":
            prompts.append("Include data, evidence, and logical structure in responses.")
        
        # Big 5 adaptations
        if big5_profile.get("openness", 0.5) > 0.7:
            prompts.append("User appreciates creative and novel approaches.")
        if big5_profile.get("conscientiousness", 0.5) > 0.7:
            prompts.append("User values thorough, well-organized responses.")
        if big5_profile.get("extraversion", 0.5) < 0.3:
            prompts.append("User prefers focused, less chatty interaction.")
        if big5_profile.get("agreeableness", 0.5) > 0.7:
            prompts.append("Use collaborative language and acknowledge user's input.")
        if big5_profile.get("neuroticism", 0.5) > 0.6:
            prompts.append("Provide reassurance and maintain calm, steady demeanor.")
        
        # Interaction style adaptation
        if interaction_style.get("length_pattern") == "user_verbose":
            prompts.append("User provides detailed input. Match their thoroughness.")
        elif interaction_style.get("length_pattern") == "agent_verbose":
            prompts.append("Keep responses concise - user prefers brevity.")
        
        # Rapport-based adjustments
        if rapport_level < 0.3:
            prompts.append("Focus on building trust through reliability and clarity.")
        elif rapport_level > 0.7:
            prompts.append("Strong rapport established. Can use subtle humor if appropriate.")
        
        # Unresolved topics
        if unresolved_topics:
            prompts.append(f"Consider addressing: {unresolved_topics[0][:50]}")
        
        return prompts
    
    def _create_intelligent_summary(self,
                                   user_messages: List[str],
                                   agent_messages: List[str],
                                   themes: List[str],
                                   emotional_trajectory: Dict[str, Any],
                                   rapport_level: float) -> str:
        """Create intelligent, context-aware summary"""
        parts = []
        
        # Conversation scope
        if user_messages and agent_messages:
            parts.append(f"{len(user_messages)} exchanges")
        
        # Main themes
        if themes:
            theme_str = ", ".join(themes[:3])
            parts.append(f"discussing {theme_str}")
        
        # Emotional context
        emotion = emotional_trajectory.get("dominant", "neutral")
        if emotion != "neutral":
            trend = emotional_trajectory.get("trend", "stable")
            parts.append(f"{emotion} tone ({trend})")
        
        # Rapport status
        if rapport_level > 0.7:
            parts.append("high engagement")
        elif rapport_level < 0.3:
            parts.append("building rapport")
        
        return " | ".join(parts) if parts else "Ongoing conversation"
    
    def _update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update stored user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        self.user_profiles[user_id].update(profile_data)
        self.user_profiles[user_id]["last_updated"] = datetime.now().isoformat()
    
    def format_for_prompt_injection(self, intelligence: ConversationIntelligence) -> str:
        """Format intelligence for LLM prompt injection"""
        prompt_parts = []
        
        # Core context
        prompt_parts.append("## Conversation Intelligence")
        prompt_parts.append(f"Summary: {intelligence.summary}")
        
        # Psychological insights
        if intelligence.psychological_profile.get("dominant_traits"):
            traits = ", ".join(intelligence.psychological_profile["dominant_traits"])
            prompt_parts.append(f"User traits: {traits}")
        
        # Big 5 personality
        if intelligence.big5_profile:
            high_traits = [k for k, v in intelligence.big5_profile.items() if v > 0.7]
            if high_traits:
                prompt_parts.append(f"Personality: High in {', '.join(high_traits)}")
        
        # Emotional state
        if intelligence.emotional_trajectory.get("dominant") != "neutral":
            emotion = intelligence.emotional_trajectory["dominant"]
            intensity = intelligence.emotional_trajectory.get("intensity", 0)
            prompt_parts.append(f"Emotional state: {emotion} (intensity: {intensity:.1f})")
        
        # Communication preferences
        formality = intelligence.linguistic_patterns.get("formality_level", "neutral")
        complexity = intelligence.linguistic_patterns.get("complexity_level", "moderate")
        prompt_parts.append(f"Communication: {formality}, {complexity} complexity")
        
        # Interaction dynamics
        engagement = intelligence.interaction_style.get("engagement_level", "moderate")
        prompt_parts.append(f"Engagement: {engagement}, Rapport: {intelligence.rapport_level:.0%}")
        
        # Key entities
        if intelligence.key_entities:
            entities = ", ".join(intelligence.key_entities[:5])
            prompt_parts.append(f"Key topics: {entities}")
        
        # Unresolved topics
        if intelligence.unresolved_topics:
            prompt_parts.append(f"Pending: {intelligence.unresolved_topics[0][:50]}")
        
        # Behavioral guidance
        if intelligence.context_prompts:
            prompt_parts.append("\n## Behavioral Adaptation")
            for prompt in intelligence.context_prompts[:5]:
                prompt_parts.append(f"- {prompt}")
        
        return "\n".join(prompt_parts)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get stored user profile"""
        return self.user_profiles.get(user_id)
    
    def get_rapport_strategies(self, 
                              user_id: str,
                              intelligence: ConversationIntelligence) -> List[str]:
        """Generate rapport-building strategies"""
        strategies = []
        
        # Based on Big 5 profile
        if intelligence.big5_profile:
            if intelligence.big5_profile.get("openness", 0.5) > 0.6:
                strategies.append("Share interesting technical insights and possibilities")
            if intelligence.big5_profile.get("conscientiousness", 0.5) > 0.6:
                strategies.append("Be precise and thorough in explanations")
            if intelligence.big5_profile.get("extraversion", 0.5) > 0.6:
                strategies.append("Engage in more conversational back-and-forth")
            elif intelligence.big5_profile.get("extraversion", 0.5) < 0.4:
                strategies.append("Respect need for focused, quiet interaction")
            if intelligence.big5_profile.get("agreeableness", 0.5) > 0.6:
                strategies.append("Use collaborative language ('we', 'together')")
            if intelligence.big5_profile.get("neuroticism", 0.5) > 0.6:
                strategies.append("Provide reassurance and maintain calm demeanor")
        
        # Based on rapport level
        if intelligence.rapport_level < 0.5:
            strategies.append("Focus on reliability and consistency")
            strategies.append("Follow through on all commitments")
        elif intelligence.rapport_level > 0.7:
            strategies.append("Can introduce personality-appropriate humor")
            strategies.append("Reference previous conversations to show memory")
        
        # Based on emotional state
        emotion = intelligence.emotional_trajectory.get("dominant", "neutral")
        if emotion == "frustration":
            strategies.append("Acknowledge challenges and provide clear solutions")
        elif emotion == "curiosity":
            strategies.append("Encourage exploration with detailed explanations")
        
        return strategies


# Singleton instance
_context_instance = None

def get_context_intelligence() -> ContextIntelligenceSystem:
    """Get or create the context intelligence singleton"""
    global _context_instance
    if _context_instance is None:
        _context_instance = ContextIntelligenceSystem()
    return _context_instance