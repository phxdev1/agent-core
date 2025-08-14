#!/usr/bin/env python3
"""
Enhanced Knowledge Graph System
Consolidates entity extraction, knowledge storage, and user profiling
with Big 5 personality integration
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class EnhancedKnowledgeGraph:
    """
    Unified knowledge graph system combining entity extraction,
    relationship management, and user profiling
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
        # In-memory graph storage (can be backed by Redis)
        self.entities = {}  # entity_id -> entity data
        self.relationships = {}  # relationship_id -> relationship data
        self.user_profiles = {}  # user_id -> profile data
        
        # Entity type definitions
        self.entity_types = {
            "person", "place", "organization", "concept", "technology",
            "project", "skill", "interest", "preference", "goal"
        }
        
        # Relationship types
        self.relationship_types = {
            "knows", "likes", "dislikes", "works_with", "located_at",
            "interested_in", "skilled_in", "prefers", "owns", "uses",
            "related_to", "part_of", "created_by", "mentioned"
        }
        
        logger.info("Enhanced Knowledge Graph initialized")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        
        # Extract person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        for match in re.finditer(person_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "person",
                "properties": {"source": "name_pattern"}
            })
        
        # Extract technologies/languages
        tech_keywords = ["python", "javascript", "react", "node", "docker", "kubernetes",
                        "aws", "azure", "gcp", "linux", "windows", "macos", "api",
                        "database", "sql", "nosql", "mongodb", "redis", "postgresql"]
        text_lower = text.lower()
        for tech in tech_keywords:
            if tech in text_lower:
                entities.append({
                    "name": tech,
                    "type": "technology",
                    "properties": {"mentioned": True}
                })
        
        # Extract organizations
        org_indicators = ["Inc", "LLC", "Corp", "Company", "University", "Institute"]
        for indicator in org_indicators:
            pattern = rf'\b[\w\s]+\s+{indicator}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "name": match.group(),
                    "type": "organization",
                    "properties": {"source": "org_pattern"}
                })
        
        # Extract skills and interests
        skill_patterns = [
            r"(?:I can|I know|skilled in|experienced with)\s+([\w\s]+)",
            r"(?:interested in|passionate about|love)\s+([\w\s]+)"
        ]
        for pattern in skill_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                skill = match.group(1).strip()
                if len(skill) < 50:  # Reasonable length
                    entities.append({
                        "name": skill,
                        "type": "skill" if "can" in pattern or "know" in pattern else "interest",
                        "properties": {"extracted_from": text[:100]}
                    })
        
        return entities
    
    def extract_user_info(self, text: str, user_id: str) -> Dict[str, Any]:
        """Extract user information from their messages"""
        info = {}
        
        # Extract name
        name_patterns = [
            r"(?:my name is|I'm|I am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["name"] = match.group(1)
                break
        
        # Extract location
        location_patterns = [
            r"(?:I live in|I'm from|located in|based in)\s+([\w\s,]+)",
            r"(?:in|from)\s+([A-Z][\w\s]+(?:,\s*[A-Z][\w\s]+)?)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(1).strip()
                if len(location) < 50:
                    info["location"] = location
                    break
        
        # Extract occupation
        occupation_patterns = [
            r"(?:I work as|I'm a|I am a|work as a|job is)\s+([\w\s]+)",
            r"(?:I'm|I am)\s+(?:a|an)\s+([\w\s]+)(?:\s+at|$)"
        ]
        for pattern in occupation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                occupation = match.group(1).strip()
                if len(occupation) < 30:
                    info["occupation"] = occupation
                    break
        
        # Extract preferences
        if "prefer" in text.lower() or "like" in text.lower():
            if not info.get("preferences"):
                info["preferences"] = []
            
            prefer_patterns = [
                r"(?:prefer|like|love|enjoy)\s+([\w\s]+)",
                r"(?:favorite|favourite)\s+([\w\s]+)\s+(?:is|are)\s+([\w\s]+)"
            ]
            for pattern in prefer_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    preference = match.group(1).strip()
                    if len(preference) < 30:
                        info["preferences"].append(preference)
        
        return info
    
    def add_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None) -> str:
        """Add an entity to the knowledge graph"""
        # Check if entity already exists
        for eid, entity in self.entities.items():
            if entity["name"].lower() == name.lower() and entity["type"] == entity_type:
                # Update properties
                if properties:
                    entity["properties"].update(properties)
                entity["last_updated"] = datetime.now().isoformat()
                return eid
        
        # Create new entity
        entity_id = str(uuid.uuid4())
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "properties": properties or {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Store in Redis if available
        if self.redis_client:
            key = f"entity:{entity_id}"
            self.redis_client.set(key, json.dumps(self.entities[entity_id]))
        
        return entity_id
    
    def add_relationship(self, 
                        source_id: str,
                        target_id: str,
                        relationship_type: str,
                        properties: Dict[str, Any] = None) -> str:
        """Add a relationship between entities"""
        # Check if relationship exists
        for rid, rel in self.relationships.items():
            if (rel["source"] == source_id and 
                rel["target"] == target_id and 
                rel["type"] == relationship_type):
                # Update properties
                if properties:
                    rel["properties"].update(properties)
                rel["last_updated"] = datetime.now().isoformat()
                return rid
        
        # Create new relationship
        relationship_id = str(uuid.uuid4())
        self.relationships[relationship_id] = {
            "id": relationship_id,
            "source": source_id,
            "target": target_id,
            "type": relationship_type,
            "properties": properties or {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Store in Redis if available
        if self.redis_client:
            key = f"relationship:{relationship_id}"
            self.redis_client.set(key, json.dumps(self.relationships[relationship_id]))
        
        return relationship_id
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
        """Update user profile with new information"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "id": user_id,
                "created_at": datetime.now().isoformat()
            }
        
        profile = self.user_profiles[user_id]
        
        # Handle different update types
        for key, value in updates.items():
            if key in ["name", "location", "occupation"]:
                profile[key] = value
            elif key in ["interests", "skills", "preferences"]:
                if key not in profile:
                    profile[key] = []
                if isinstance(value, list):
                    profile[key].extend(value)
                else:
                    profile[key].append(value)
                # Remove duplicates
                profile[key] = list(set(profile[key]))
            elif key == "big5_traits":
                profile["personality"] = value
            else:
                profile[key] = value
        
        profile["last_updated"] = datetime.now().isoformat()
        
        # Create entity for user if not exists
        user_entity_id = self.add_entity(
            name=profile.get("name", f"User_{user_id[:8]}"),
            entity_type="person",
            properties={"user_id": user_id, "profile": True}
        )
        profile["entity_id"] = user_entity_id
        
        # Create relationships for interests and skills
        if "interests" in profile:
            for interest in profile["interests"]:
                interest_id = self.add_entity(interest, "interest")
                self.add_relationship(user_entity_id, interest_id, "interested_in")
        
        if "skills" in profile:
            for skill in profile["skills"]:
                skill_id = self.add_entity(skill, "skill")
                self.add_relationship(user_entity_id, skill_id, "skilled_in")
        
        # Store in Redis if available
        if self.redis_client:
            key = f"user_profile:{user_id}"
            self.redis_client.set(key, json.dumps(profile))
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Try to load from Redis
        if self.redis_client:
            key = f"user_profile:{user_id}"
            data = self.redis_client.get(key)
            if data:
                profile = json.loads(data)
                self.user_profiles[user_id] = profile
                return profile
        
        return None
    
    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for entities"""
        query_lower = query.lower()
        results = []
        
        for entity_id, entity in self.entities.items():
            # Filter by type if specified
            if entity_type and entity["type"] != entity_type:
                continue
            
            # Search in name
            if query_lower in entity["name"].lower():
                results.append(entity)
                continue
            
            # Search in properties
            for key, value in entity.get("properties", {}).items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(entity)
                    break
        
        return results
    
    def get_related_entities(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get entities related to a given entity"""
        related = []
        
        for rel_id, rel in self.relationships.items():
            # Check if entity is source or target
            if rel["source"] == entity_id:
                # Filter by relationship type if specified
                if not relationship_type or rel["type"] == relationship_type:
                    target = self.entities.get(rel["target"])
                    if target:
                        related.append({
                            "entity": target,
                            "relationship": rel["type"],
                            "direction": "outgoing"
                        })
            elif rel["target"] == entity_id:
                if not relationship_type or rel["type"] == relationship_type:
                    source = self.entities.get(rel["source"])
                    if source:
                        related.append({
                            "entity": source,
                            "relationship": rel["type"],
                            "direction": "incoming"
                        })
        
        return related
    
    def get_user_connections(self, user_id: str) -> Dict[str, List[str]]:
        """Get all connections for a user"""
        profile = self.get_user_profile(user_id)
        if not profile or "entity_id" not in profile:
            return {}
        
        connections = defaultdict(list)
        related = self.get_related_entities(profile["entity_id"])
        
        for item in related:
            entity = item["entity"]
            rel_type = item["relationship"]
            connections[rel_type].append(entity["name"])
        
        return dict(connections)
    
    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """Merge two entities that represent the same thing"""
        if entity_id1 not in self.entities or entity_id2 not in self.entities:
            return entity_id1
        
        entity1 = self.entities[entity_id1]
        entity2 = self.entities[entity_id2]
        
        # Merge properties
        entity1["properties"].update(entity2["properties"])
        entity1["aliases"] = entity1.get("aliases", [])
        entity1["aliases"].append(entity2["name"])
        entity1["last_updated"] = datetime.now().isoformat()
        
        # Update relationships
        for rel_id, rel in self.relationships.items():
            if rel["source"] == entity_id2:
                rel["source"] = entity_id1
            if rel["target"] == entity_id2:
                rel["target"] = entity_id1
        
        # Remove entity2
        del self.entities[entity_id2]
        
        return entity_id1
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the entire graph"""
        return {
            "entities": list(self.entities.values()),
            "relationships": list(self.relationships.values()),
            "user_profiles": list(self.user_profiles.values()),
            "exported_at": datetime.now().isoformat()
        }
    
    def import_graph(self, data: Dict[str, Any]):
        """Import a graph"""
        # Import entities
        for entity in data.get("entities", []):
            self.entities[entity["id"]] = entity
        
        # Import relationships
        for rel in data.get("relationships", []):
            self.relationships[rel["id"]] = rel
        
        # Import user profiles
        for profile in data.get("user_profiles", []):
            self.user_profiles[profile["id"]] = profile
        
        logger.info(f"Imported {len(self.entities)} entities, "
                   f"{len(self.relationships)} relationships, "
                   f"{len(self.user_profiles)} user profiles")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        entity_type_counts = defaultdict(int)
        for entity in self.entities.values():
            entity_type_counts[entity["type"]] += 1
        
        relationship_type_counts = defaultdict(int)
        for rel in self.relationships.values():
            relationship_type_counts[rel["type"]] += 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_user_profiles": len(self.user_profiles),
            "entity_types": dict(entity_type_counts),
            "relationship_types": dict(relationship_type_counts)
        }


# Singleton instance
_knowledge_instance = None

def get_enhanced_knowledge_graph(redis_client=None) -> EnhancedKnowledgeGraph:
    """Get or create the enhanced knowledge graph singleton"""
    global _knowledge_instance
    if _knowledge_instance is None:
        _knowledge_instance = EnhancedKnowledgeGraph(redis_client)
    return _knowledge_instance