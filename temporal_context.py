#!/usr/bin/env python3
"""
Temporal Context System
Tracks time, schedules tasks, understands temporal relationships
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import croniter
from rq import Queue
from rq_scheduler import Scheduler
import redis

from redis_logger import get_redis_logger
logger = get_redis_logger(__name__)


class TimeGranularity(Enum):
    """Time granularities for context"""
    IMMEDIATE = "immediate"      # < 1 minute
    RECENT = "recent"            # < 1 hour  
    TODAY = "today"              # Same day
    THIS_WEEK = "this_week"      # Same week
    THIS_MONTH = "this_month"    # Same month
    HISTORICAL = "historical"    # Older


@dataclass
class TemporalEvent:
    """Represents an event in time"""
    event_id: str
    event_type: str
    description: str
    timestamp: float
    duration: Optional[float] = None
    recurrence: Optional[str] = None  # Cron expression
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)
    
    @property
    def relative_time(self) -> str:
        """Get human-readable relative time"""
        now = datetime.now()
        event_time = self.datetime
        delta = now - event_time
        
        if delta.total_seconds() < 0:
            # Future event
            delta = abs(delta)
            if delta.days > 0:
                return f"in {delta.days} days"
            elif delta.seconds > 3600:
                return f"in {delta.seconds // 3600} hours"
            elif delta.seconds > 60:
                return f"in {delta.seconds // 60} minutes"
            else:
                return "soon"
        else:
            # Past event
            if delta.days > 365:
                return f"{delta.days // 365} years ago"
            elif delta.days > 30:
                return f"{delta.days // 30} months ago"
            elif delta.days > 0:
                return f"{delta.days} days ago"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600} hours ago"
            elif delta.seconds > 60:
                return f"{delta.seconds // 60} minutes ago"
            else:
                return "just now"
    
    def get_granularity(self) -> TimeGranularity:
        """Get time granularity of event"""
        now = datetime.now()
        event_time = self.datetime
        delta = abs(now - event_time)
        
        if delta.total_seconds() < 60:
            return TimeGranularity.IMMEDIATE
        elif delta.total_seconds() < 3600:
            return TimeGranularity.RECENT
        elif delta.days == 0:
            return TimeGranularity.TODAY
        elif delta.days < 7:
            return TimeGranularity.THIS_WEEK
        elif delta.days < 30:
            return TimeGranularity.THIS_MONTH
        else:
            return TimeGranularity.HISTORICAL


class TemporalContext:
    """Manages temporal context and scheduling"""
    
    def __init__(self, redis_config: Optional[Dict] = None):
        self.redis = redis.Redis(
            host='redis-11364.c24.us-east-mz-1.ec2.redns.redis-cloud.com',
            port=11364,
            username='default',
            password='UQHtexzcYtFCoG7R55InSvmdYHn8fvcf',
            db=0,
            decode_responses=False
        )
        
        # RQ Scheduler for scheduled tasks
        try:
            # Quick timeout to avoid hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Scheduler initialization timeout")
            
            # Set 2 second timeout for scheduler init
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)
            
            try:
                self.scheduler = Scheduler(connection=self.redis)
                self.queue = Queue('scheduled', connection=self.redis)
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
                logger.info("RQ Scheduler initialized successfully")
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
                logger.warning("RQ Scheduler initialization timed out. Scheduled tasks disabled.")
                self.scheduler = None
                self.queue = None
        except Exception as e:
            logger.warning(f"RQ Scheduler initialization failed: {e}. Scheduled tasks disabled.")
            self.scheduler = None
            self.queue = None
        
        # Event storage keys
        self.events_key = "temporal:events"
        self.schedule_key = "temporal:schedule"
        self.patterns_key = "temporal:patterns"
        
        logger.info("Temporal context system initialized")
    
    def add_event(self, event_type: str, description: str, 
                  timestamp: Optional[float] = None,
                  duration: Optional[float] = None,
                  metadata: Optional[Dict] = None) -> str:
        """Add an event to temporal context"""
        event_id = f"event:{int(time.time() * 1000)}"
        
        event = TemporalEvent(
            event_id=event_id,
            event_type=event_type,
            description=description,
            timestamp=timestamp or time.time(),
            duration=duration,
            metadata=metadata or {}
        )
        
        # Store in Redis - ensure all values are strings/bytes
        event_data = {
            "event_id": str(event.event_id),
            "event_type": str(event.event_type),
            "description": str(event.description or ""),
            "timestamp": str(event.timestamp),
            "duration": str(event.duration or 0),
            "metadata": json.dumps(event.metadata) if event.metadata else "{}"
        }
        
        # Ensure all values are properly encoded
        encoded_data = {k: v.encode('utf-8') if isinstance(v, str) else v 
                       for k, v in event_data.items()}
        
        self.redis.hset(f"{self.events_key}:{event_id}", mapping=encoded_data)
        
        # Add to time-sorted set for efficient queries
        self.redis.zadd(self.events_key, {event_id: event.timestamp})
        
        logger.info(f"Added temporal event: {event_type} at {event.relative_time}")
        return event_id
    
    def get_events_by_time_range(self, start: float, end: float) -> List[TemporalEvent]:
        """Get events within a time range"""
        # Get event IDs in range
        event_ids = self.redis.zrangebyscore(self.events_key, start, end)
        
        events = []
        for event_id in event_ids:
            event_data = self.redis.hgetall(f"{self.events_key}:{event_id.decode()}")
            if event_data:
                events.append(self._parse_event(event_data))
        
        return events
    
    def get_events_by_granularity(self, granularity: TimeGranularity) -> List[TemporalEvent]:
        """Get events by time granularity"""
        now = datetime.now()
        
        if granularity == TimeGranularity.IMMEDIATE:
            start = now - timedelta(minutes=1)
        elif granularity == TimeGranularity.RECENT:
            start = now - timedelta(hours=1)
        elif granularity == TimeGranularity.TODAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == TimeGranularity.THIS_WEEK:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == TimeGranularity.THIS_MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now - timedelta(days=365)
        
        return self.get_events_by_time_range(start.timestamp(), now.timestamp())
    
    def schedule_task(self, task_func: str, run_at: datetime, 
                     args: Optional[tuple] = None,
                     kwargs: Optional[dict] = None,
                     description: Optional[str] = None) -> str:
        """Schedule a task for future execution"""
        if not self.scheduler:
            logger.warning("Scheduler not available - task not scheduled")
            return "scheduler_unavailable"
            
        job = self.scheduler.enqueue_at(
            run_at,
            task_func,
            *args if args else [],
            **kwargs if kwargs else {}
        )
        
        # Add as temporal event
        event_id = self.add_event(
            event_type="scheduled_task",
            description=description or f"Scheduled: {task_func}",
            timestamp=run_at.timestamp(),
            metadata={"job_id": job.id, "function": task_func}
        )
        
        logger.info(f"Scheduled task {task_func} for {run_at}")
        return job.id
    
    def schedule_recurring(self, task_func: str, cron_expression: str,
                          args: Optional[tuple] = None,
                          kwargs: Optional[dict] = None,
                          description: Optional[str] = None) -> str:
        """Schedule a recurring task with cron expression"""
        if not self.scheduler:
            logger.warning("Scheduler not available - recurring task not scheduled")
            return "scheduler_unavailable"
            
        cron = croniter.croniter(cron_expression, datetime.now())
        next_run = cron.get_next(datetime)
        
        # Schedule first occurrence
        job_id = self.schedule_task(
            task_func, next_run, args, kwargs, 
            description or f"Recurring: {task_func}"
        )
        
        # Store recurrence pattern
        self.redis.hset(self.schedule_key, job_id, cron_expression)
        
        return job_id
    
    def detect_patterns(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Detect temporal patterns in events"""
        # Get recent events
        now = time.time()
        week_ago = now - (7 * 24 * 3600)
        events = self.get_events_by_time_range(week_ago, now)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        patterns = {
            "total_events": len(events),
            "event_types": {},
            "time_distribution": {
                "morning": 0,    # 6-12
                "afternoon": 0,  # 12-18
                "evening": 0,    # 18-24
                "night": 0       # 0-6
            },
            "day_distribution": {
                "weekday": 0,
                "weekend": 0
            },
            "frequency": {}
        }
        
        for event in events:
            # Count by type
            patterns["event_types"][event.event_type] = \
                patterns["event_types"].get(event.event_type, 0) + 1
            
            # Time of day distribution
            hour = event.datetime.hour
            if 6 <= hour < 12:
                patterns["time_distribution"]["morning"] += 1
            elif 12 <= hour < 18:
                patterns["time_distribution"]["afternoon"] += 1
            elif 18 <= hour < 24:
                patterns["time_distribution"]["evening"] += 1
            else:
                patterns["time_distribution"]["night"] += 1
            
            # Weekday/weekend
            if event.datetime.weekday() < 5:
                patterns["day_distribution"]["weekday"] += 1
            else:
                patterns["day_distribution"]["weekend"] += 1
        
        # Calculate daily frequency
        if events:
            days_span = (events[-1].timestamp - events[0].timestamp) / 86400
            if days_span > 0:
                patterns["frequency"]["daily_average"] = len(events) / days_span
        
        return patterns
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current temporal context"""
        now = datetime.now()
        
        return {
            "current_time": now.isoformat(),
            "day_of_week": now.strftime("%A"),
            "time_of_day": self._get_time_of_day(now),
            "recent_events": [
                {
                    "type": e.event_type,
                    "description": e.description,
                    "when": e.relative_time
                }
                for e in self.get_events_by_granularity(TimeGranularity.RECENT)
            ],
            "todays_events": len(self.get_events_by_granularity(TimeGranularity.TODAY)),
            "upcoming_tasks": self._get_upcoming_tasks(),
            "patterns": self.detect_patterns()
        }
    
    def _get_time_of_day(self, dt: datetime) -> str:
        """Get human-readable time of day"""
        hour = dt.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _get_upcoming_tasks(self) -> List[Dict]:
        """Get upcoming scheduled tasks"""
        if not self.scheduler:
            return []
            
        jobs = list(self.scheduler.get_jobs())  # Convert generator to list
        upcoming = []
        
        for job in jobs[:5]:  # Next 5 tasks
            upcoming.append({
                "id": job.id,
                "function": job.func_name,
                "scheduled_for": job.enqueued_at.isoformat() if job.enqueued_at else None
            })
        
        return upcoming
    
    def _parse_event(self, event_data: Dict[bytes, bytes]) -> TemporalEvent:
        """Parse event from Redis data"""
        return TemporalEvent(
            event_id=event_data[b'event_id'].decode(),
            event_type=event_data[b'event_type'].decode(),
            description=event_data[b'description'].decode(),
            timestamp=float(event_data[b'timestamp']),
            duration=float(event_data[b'duration']) if event_data.get(b'duration') else None,
            metadata=json.loads(event_data[b'metadata']) if event_data.get(b'metadata') else {}
        )
    
    def understand_time_reference(self, text: str) -> Tuple[Optional[datetime], Optional[str]]:
        """Parse natural language time references"""
        text_lower = text.lower()
        now = datetime.now()
        
        # Relative time references
        if "now" in text_lower or "right now" in text_lower:
            return now, "immediate"
        elif "today" in text_lower:
            return now, "today"
        elif "tomorrow" in text_lower:
            return now + timedelta(days=1), "tomorrow"
        elif "yesterday" in text_lower:
            return now - timedelta(days=1), "yesterday"
        elif "next week" in text_lower:
            return now + timedelta(weeks=1), "next_week"
        elif "last week" in text_lower:
            return now - timedelta(weeks=1), "last_week"
        
        # Time of day
        if "morning" in text_lower:
            target = now.replace(hour=9, minute=0, second=0)
            if target < now:
                target += timedelta(days=1)
            return target, "morning"
        elif "afternoon" in text_lower:
            target = now.replace(hour=14, minute=0, second=0)
            if target < now:
                target += timedelta(days=1)
            return target, "afternoon"
        elif "evening" in text_lower:
            target = now.replace(hour=19, minute=0, second=0)
            if target < now:
                target += timedelta(days=1)
            return target, "evening"
        
        # Specific durations
        import re
        
        # "in X hours/minutes/days"
        in_pattern = r"in (\d+) (hour|minute|day|week)"
        match = re.search(in_pattern, text_lower)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            
            if unit == "minute":
                return now + timedelta(minutes=amount), f"in_{amount}_minutes"
            elif unit == "hour":
                return now + timedelta(hours=amount), f"in_{amount}_hours"
            elif unit == "day":
                return now + timedelta(days=amount), f"in_{amount}_days"
            elif unit == "week":
                return now + timedelta(weeks=amount), f"in_{amount}_weeks"
        
        # "X hours/minutes ago"
        ago_pattern = r"(\d+) (hour|minute|day|week)s? ago"
        match = re.search(ago_pattern, text_lower)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            
            if unit == "minute":
                return now - timedelta(minutes=amount), f"{amount}_minutes_ago"
            elif unit == "hour":
                return now - timedelta(hours=amount), f"{amount}_hours_ago"
            elif unit == "day":
                return now - timedelta(days=amount), f"{amount}_days_ago"
            elif unit == "week":
                return now - timedelta(weeks=amount), f"{amount}_weeks_ago"
        
        return None, None


# Singleton instance
_temporal_context = None

def get_temporal_context() -> TemporalContext:
    """Get singleton temporal context instance"""
    global _temporal_context
    if _temporal_context is None:
        _temporal_context = TemporalContext()
    return _temporal_context


async def test_temporal():
    """Test temporal context"""
    tc = get_temporal_context()
    
    # Add some events
    tc.add_event("chat", "User asked about research")
    tc.add_event("research", "Started research on quantum computing")
    tc.add_event("system", "CPU spike detected", metadata={"cpu": 85})
    
    # Parse time references
    refs = [
        "remind me tomorrow morning",
        "what happened yesterday",
        "schedule this in 2 hours",
        "check logs from 3 days ago"
    ]
    
    print("TIME REFERENCE PARSING:")
    print("-" * 40)
    for ref in refs:
        dt, desc = tc.understand_time_reference(ref)
        if dt:
            print(f'"{ref}" -> {dt.strftime("%Y-%m-%d %H:%M")} ({desc})')
    
    print("\nTEMPORAL CONTEXT SUMMARY:")
    print("-" * 40)
    summary = tc.get_context_summary()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_temporal())