"""
Celery Application for Background Tasks
Handles asynchronous processing of extractions and other long-running tasks
"""

import os
import logging
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# Create Celery app
celery_app = Celery(
    "medtex",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["backend.app.tasks.celery_tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-extractions": {
        "task": "backend.app.tasks.celery_tasks.cleanup_old_extractions",
        "schedule": crontab(hour=2, minute=0),  # Run daily at 2 AM UTC
    },
    "cache-stats": {
        "task": "backend.app.tasks.celery_tasks.collect_cache_stats",
        "schedule": crontab(minute="*/30"),  # Run every 30 minutes
    },
}

logger.info(f"Celery app configured with broker: {CELERY_BROKER_URL}")
