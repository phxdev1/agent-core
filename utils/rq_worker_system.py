#!/usr/bin/env python3
"""
RQ-based Worker System with Hot Reload
Simple, reliable task queue with automatic code reloading
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# RQ imports
from rq import Queue, Worker, get_current_job
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry
from rq.decorators import job

import redis
from redis import Redis

# Our imports - use absolute imports with parent path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.research_system import ResearchSystem
from utils.pdf_extractor import PDFExtractor
from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)

# Redis connection for RQ (don't decode responses for RQ compatibility)
redis_conn = Redis(
    host='redis-11364.c24.us-east-mz-1.ec2.redns.redis-cloud.com',
    port=11364,
    username='default',
    password='UQHtexzcYtFCoG7R55InSvmdYHn8fvcf',
    db=0,
    decode_responses=False  # RQ needs binary mode
)

# Create queues
high_queue = Queue('high', connection=redis_conn)
default_queue = Queue('default', connection=redis_conn)
low_queue = Queue('low', connection=redis_conn)
research_queue = Queue('research', connection=redis_conn)


def send_notification(channel: str, message: str, data: Optional[Dict] = None):
    """Send notification via Redis pub/sub"""
    notification = {
        "channel": channel,
        "message": message,
        "timestamp": time.time(),
        "data": data or {}
    }
    # Encode to bytes for binary mode Redis
    redis_conn.publish(f"agent:{channel}", json.dumps(notification).encode('utf-8'))
    logger.info(f"Notification sent: {channel} - {message}")


def update_job_progress(percentage: int, message: str = ""):
    """Update current job progress"""
    job = get_current_job()
    if job:
        job.meta['progress'] = percentage
        job.meta['message'] = message
        job.save_meta()
        
        # Send progress notification
        send_notification(
            "progress",
            f"Job {job.id}: {message} ({percentage}%)",
            {"job_id": job.id, "progress": percentage}
        )


# === RESEARCH TASKS ===

def research_task(topic: str, max_documents: int = 10, sources: Optional[list] = None):
    """
    Research a topic - runs as RQ job
    This function is reloaded automatically when the file changes
    """
    logger.info(f"Starting research on: {topic}")
    update_job_progress(0, f"Starting research on {topic}")
    
    # Default sources
    if sources is None:
        sources = ["arxiv"]
        if os.getenv("SERPAPI_KEY"):
            sources.extend(["scholar", "web"])
    
    # Run async research in sync context
    async def _research():
        research = ResearchSystem()
        
        update_job_progress(20, "Querying sources")
        
        # Perform research
        results = await research.research_topic(
            topic=topic,
            sources=sources,
            max_documents=max_documents
        )
        
        update_job_progress(80, "Processing results")
        
        # Send completion notification
        send_notification(
            "research_complete",
            f"Research completed: {results['total_documents']} documents on {topic}",
            {
                "topic": topic,
                "documents": results['total_documents'],
                "sources": results['synthesis']['sources_breakdown']
            }
        )
        
        update_job_progress(100, "Research complete")
        return results
    
    # Run the async function
    return asyncio.run(_research())


def pdf_extract_task(url: str):
    """Extract PDF content"""
    logger.info(f"Extracting PDF: {url}")
    update_job_progress(0, "Starting PDF extraction")
    
    async def _extract():
        extractor = PDFExtractor()
        
        update_job_progress(30, "Downloading PDF")
        result = await extractor.process_arxiv_paper(url)
        
        if result['success']:
            update_job_progress(90, "Extraction complete")
            send_notification(
                "pdf_extracted",
                f"PDF extracted: {result.get('word_count', 0)} words",
                {"url": url, "word_count": result.get('word_count', 0)}
            )
        else:
            send_notification(
                "pdf_failed",
                f"PDF extraction failed: {result.get('error', 'Unknown error')}",
                {"url": url}
            )
        
        update_job_progress(100, "Done")
        return result
    
    return asyncio.run(_extract())


def query_knowledge_task(query: str, use_rag: bool = True, top_k: int = 5):
    """Query the knowledge base"""
    logger.info(f"Querying knowledge: {query}")
    update_job_progress(0, "Processing query")
    
    async def _query():
        research = ResearchSystem()
        
        update_job_progress(50, "Searching knowledge base")
        result = await research.query_knowledge(
            query=query,
            use_rag=use_rag,
            top_k=top_k
        )
        
        update_job_progress(100, "Query complete")
        return result
    
    return asyncio.run(_query())


# === UTILITY TASKS ===

def health_check_task():
    """Health check task"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "queues": {
            "high": len(high_queue),
            "default": len(default_queue),
            "low": len(low_queue),
            "research": len(research_queue)
        }
    }


# === CLIENT INTERFACE ===

class RQClient:
    """Client for submitting jobs to RQ"""
    
    def __init__(self):
        self.redis_conn = redis_conn
        self.pubsub = self.redis_conn.pubsub()
    
    def research(self, topic: str, max_documents: int = 10) -> str:
        """Submit research job"""
        job = research_queue.enqueue(
            research_task,
            topic=topic,
            max_documents=max_documents,
            job_timeout='30m',
            result_ttl=86400,
            failure_ttl=86400
        )
        
        print(f"\n✅ Research job submitted")
        print(f"   Job ID: {job.id}")
        print(f"   Topic: {topic}")
        print(f"   Status URL: http://localhost:9181/jobs/{job.id}")
        
        return job.id
    
    def extract_pdf(self, url: str) -> str:
        """Submit PDF extraction job"""
        job = research_queue.enqueue(
            pdf_extract_task,
            url=url,
            job_timeout='10m'
        )
        
        print(f"\n✅ PDF extraction job submitted")
        print(f"   Job ID: {job.id}")
        
        return job.id
    
    def query(self, question: str) -> str:
        """Submit knowledge query job"""
        job = default_queue.enqueue(
            query_knowledge_task,
            query=question,
            use_rag=True
        )
        
        print(f"\n✅ Query job submitted")
        print(f"   Job ID: {job.id}")
        
        return job.id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        job = Job.fetch(job_id, connection=self.redis_conn)
        
        return {
            "id": job.id,
            "status": job.get_status(),
            "progress": job.meta.get('progress', 0),
            "message": job.meta.get('message', ''),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "result": job.result if job.is_finished else None,
            "error": str(job.exc_info) if job.is_failed else None
        }
    
    def get_job_result(self, job_id: str) -> Any:
        """Get job result (blocking)"""
        job = Job.fetch(job_id, connection=self.redis_conn)
        
        # Wait for job to complete
        while not job.is_finished and not job.is_failed:
            time.sleep(1)
            job.refresh()
        
        if job.is_failed:
            raise Exception(f"Job failed: {job.exc_info}")
        
        return job.result
    
    def cancel_job(self, job_id: str):
        """Cancel a job"""
        job = Job.fetch(job_id, connection=self.redis_conn)
        job.cancel()
        print(f"Job {job_id} cancelled")
    
    def list_jobs(self, queue_name: str = 'default', status: str = 'all'):
        """List jobs in queue"""
        queue = Queue(queue_name, connection=self.redis_conn)
        
        jobs = []
        
        if status in ['all', 'queued']:
            for job_id in queue.job_ids:
                job = Job.fetch(job_id, connection=self.redis_conn)
                jobs.append({
                    "id": job.id,
                    "status": "queued",
                    "function": job.func_name
                })
        
        if status in ['all', 'started']:
            registry = StartedJobRegistry(queue_name, connection=self.redis_conn)
            for job_id in registry.get_job_ids():
                job = Job.fetch(job_id, connection=self.redis_conn)
                jobs.append({
                    "id": job.id,
                    "status": "started",
                    "function": job.func_name,
                    "progress": job.meta.get('progress', 0)
                })
        
        if status in ['all', 'finished']:
            registry = FinishedJobRegistry(queue_name, connection=self.redis_conn)
            for job_id in registry.get_job_ids()[:10]:  # Last 10
                job = Job.fetch(job_id, connection=self.redis_conn)
                jobs.append({
                    "id": job.id,
                    "status": "finished",
                    "function": job.func_name
                })
        
        return jobs
    
    def subscribe_notifications(self):
        """Subscribe to notifications"""
        self.pubsub.subscribe("agent:*")
        
        print("\n👂 Listening for notifications...")
        print("   Press Ctrl+C to stop\n")
        
        for message in self.pubsub.listen():
            if message['type'] == 'pmessage' or message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    timestamp = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')
                    
                    if data['channel'] == 'progress':
                        print(f"[{timestamp}] 📊 {data['message']}")
                    elif data['channel'] == 'research_complete':
                        print(f"[{timestamp}] ✅ {data['message']}")
                    elif data['channel'] == 'pdf_extracted':
                        print(f"[{timestamp}] 📄 {data['message']}")
                    else:
                        print(f"[{timestamp}] 🔔 {data['message']}")
                        
                except:
                    pass


def main():
    """Interactive client"""
    client = RQClient()
    
    print("\n" + "="*60)
    print("RQ AGENT CLIENT")
    print("="*60)
    print("\nCommands:")
    print("  research <topic>   - Research a topic")
    print("  pdf <url>         - Extract PDF")
    print("  query <question>  - Query knowledge base")
    print("  status <job_id>   - Check job status")
    print("  result <job_id>   - Get job result")
    print("  list [queue]      - List jobs")
    print("  listen            - Listen for notifications")
    print("  quit              - Exit")
    print("\n")
    
    while True:
        try:
            command = input("> ").strip()
            
            if not command:
                continue
            
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            
            if cmd == "quit":
                break
            
            elif cmd == "research" and len(parts) > 1:
                client.research(parts[1])
            
            elif cmd == "pdf" and len(parts) > 1:
                client.extract_pdf(parts[1])
            
            elif cmd == "query" and len(parts) > 1:
                client.query(parts[1])
            
            elif cmd == "status" and len(parts) > 1:
                status = client.get_job_status(parts[1])
                print(json.dumps(status, indent=2, default=str))
            
            elif cmd == "result" and len(parts) > 1:
                result = client.get_job_result(parts[1])
                print(f"Result: {result}")
            
            elif cmd == "list":
                queue_name = parts[1] if len(parts) > 1 else 'default'
                jobs = client.list_jobs(queue_name)
                for job in jobs:
                    print(f"  {job['id'][:8]}... | {job['status']} | {job['function']}")
            
            elif cmd == "listen":
                client.subscribe_notifications()
            
            else:
                print("Invalid command")
                
        except KeyboardInterrupt:
            print("\n")
            continue
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()