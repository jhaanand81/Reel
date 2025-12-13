"""
Celery Tasks for Async Video Generation
Production-ready for 50-100 concurrent users

This module handles long-running video generation tasks asynchronously,
allowing the Flask backend to respond immediately while videos are
processed in the background.
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import os
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CELERY CONFIGURATION
# ============================================================

# Redis URL for broker and backend
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

# Create Celery app
celery_app = Celery(
    'video_tasks',
    broker=REDIS_URL,
    backend=CELERY_RESULT_BACKEND
)

# Celery configuration for production
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Concurrency settings (adjust based on GPU/CPU capacity)
    worker_concurrency=10,  # Number of concurrent tasks per worker
    worker_prefetch_multiplier=1,  # Don't prefetch too many tasks

    # Task execution settings
    task_acks_late=True,  # Acknowledge after task completes (reliability)
    task_reject_on_worker_lost=True,  # Re-queue if worker dies
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # Soft limit at 9 minutes

    # Result settings
    result_expires=86400,  # Results expire after 24 hours
    result_extended=True,  # Store additional task metadata

    # Rate limiting
    task_default_rate_limit='100/m',  # 100 tasks per minute max

    # Retry settings
    task_default_retry_delay=30,  # Wait 30 seconds before retry
    task_max_retries=3,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)


# ============================================================
# TASK SIGNALS (for monitoring)
# ============================================================

@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extras):
    """Log task start"""
    logger.info(f"Task starting: {task.name} [{task_id}]")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **extras):
    """Log task completion"""
    logger.info(f"Task completed: {task.name} [{task_id}] - State: {state}")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extras):
    """Log task failure"""
    logger.error(f"Task failed: {task_id} - {exception}")


# ============================================================
# VIDEO GENERATION TASKS
# ============================================================

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def generate_video_task(self, project_id: str, prompt: str, duration: int,
                        reel_type: str = "ai-generated",
                        aspect_ratio: str = "9:16") -> Dict[str, Any]:
    """
    Async video generation task

    This task handles the entire video generation pipeline:
    1. Generate video with Replicate API
    2. Poll for completion
    3. Download and save video
    4. Update project status

    Args:
        project_id: Unique project identifier
        prompt: Video generation prompt
        duration: Target video duration in seconds
        reel_type: Type of video (ai-generated, product-demo, etc.)
        aspect_ratio: Video aspect ratio (9:16 for reels)

    Returns:
        Dict with video URL and status
    """
    try:
        # Import services (lazy import to avoid circular dependencies)
        from main_hybrid import ReplicateService, get_media_duration, atomic_write_json, atomic_read_json

        logger.info(f"[TASK] Starting video generation: {project_id}")

        # Initialize service
        replicate = ReplicateService()
        if not replicate.is_available():
            raise Exception("Replicate API not available")

        # Start video generation
        result = replicate.generate_video(
            prompt=prompt,
            duration=duration,
            reel_type=reel_type,
            project_id=project_id,
            aspect_ratio=aspect_ratio
        )

        job_id = result['job_id']
        logger.info(f"[TASK] Video job started: {job_id}")

        # Update task state
        self.update_state(state='PROCESSING', meta={
            'job_id': job_id,
            'progress': 10,
            'message': 'Video generation started'
        })

        # Poll for completion (with timeout)
        import time
        max_wait = 600  # 10 minutes
        poll_interval = 5  # 5 seconds
        elapsed = 0

        while elapsed < max_wait:
            status = replicate.check_status(job_id)

            if status['status'] == 'completed' and status.get('videoUrl'):
                logger.info(f"[TASK] Video completed: {job_id}")

                # Download and save video
                video_url = status['videoUrl']
                video_path = _download_video(project_id, video_url)

                # Validate duration
                actual_duration = get_media_duration(Path(video_path))
                duration_valid = abs(actual_duration - duration) <= 2.0

                # Update project data
                _update_project_status(project_id, {
                    'status': 'completed',
                    'videoUrl': video_url,
                    'videoPath': str(video_path),
                    'actualDuration': actual_duration,
                    'targetDuration': duration,
                    'durationValid': duration_valid,
                    'completed': datetime.utcnow().isoformat()
                })

                return {
                    'status': 'completed',
                    'project_id': project_id,
                    'video_url': video_url,
                    'video_path': str(video_path),
                    'actual_duration': actual_duration,
                    'duration_valid': duration_valid
                }

            elif status['status'] == 'failed':
                error_msg = status.get('error', 'Video generation failed')
                raise Exception(error_msg)

            # Update progress
            progress = status.get('progress', 0)
            self.update_state(state='PROCESSING', meta={
                'job_id': job_id,
                'progress': min(progress, 95),
                'message': f'Generating video... {progress}%'
            })

            time.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout
        raise Exception(f"Video generation timed out after {max_wait} seconds")

    except Exception as exc:
        logger.error(f"[TASK] Video generation failed: {exc}")

        # Update project with error
        _update_project_status(project_id, {
            'status': 'failed',
            'error': str(exc),
            'failed': datetime.utcnow().isoformat()
        })

        # Retry if possible
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'status': 'failed',
            'project_id': project_id,
            'error': str(exc)
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def generate_voiceover_task(self, project_id: str, script: str,
                            voice_type: str = "male-professional",
                            target_duration: float = None) -> Dict[str, Any]:
    """
    Async voiceover generation task

    Generates voiceover with Edge-TTS and adjusts speed to match target duration.
    """
    try:
        from main_hybrid import EdgeTTSService, get_media_duration

        logger.info(f"[TASK] Starting voiceover generation: {project_id}")

        # Initialize Edge-TTS
        edge_tts = EdgeTTSService()
        if not edge_tts.is_available():
            raise Exception("Edge-TTS not available")

        # Generate voiceover
        result = edge_tts.generate_voiceover(
            text=script,
            voice_type=voice_type,
            project_id=project_id,
            target_duration=target_duration
        )

        # Update project
        _update_project_status(project_id, {
            'voiceover': {
                'status': 'completed',
                'path': result['audio_path'],
                'actualDuration': result['actual_duration'],
                'targetDuration': target_duration,
                'speedAdjusted': result['speed_adjusted'],
                'completed': datetime.utcnow().isoformat()
            }
        })

        return {
            'status': 'completed',
            'project_id': project_id,
            'audio_path': result['audio_path'],
            'actual_duration': result['actual_duration'],
            'speed_adjusted': result['speed_adjusted']
        }

    except Exception as exc:
        logger.error(f"[TASK] Voiceover generation failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'status': 'failed',
            'project_id': project_id,
            'error': str(exc)
        }


@celery_app.task(bind=True, max_retries=2)
def compose_video_task(self, project_id: str, target_duration: float = None) -> Dict[str, Any]:
    """
    Async video composition task

    Combines video and audio with exact duration control.
    """
    try:
        from main_hybrid import VideoComposer

        logger.info(f"[TASK] Starting video composition: {project_id}")

        composer = VideoComposer()
        if not composer.is_available():
            raise Exception("FFmpeg not available for composition")

        result = composer.compose(
            project_id=project_id,
            target_duration=target_duration
        )

        # Update project
        _update_project_status(project_id, {
            'composition': {
                'status': 'completed',
                'path': result['output_path'],
                'actualDuration': result['actual_duration'],
                'validated': result['validated'],
                'completed': datetime.utcnow().isoformat()
            }
        })

        return {
            'status': 'completed',
            'project_id': project_id,
            'output_path': result['output_path'],
            'actual_duration': result['actual_duration'],
            'validated': result['validated']
        }

    except Exception as exc:
        logger.error(f"[TASK] Video composition failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'status': 'failed',
            'project_id': project_id,
            'error': str(exc)
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _download_video(project_id: str, video_url: str) -> Path:
    """Download video from URL and save locally"""
    import requests

    video_dir = Path(f"outputs/videos/{project_id}")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "raw.mp4"

    response = requests.get(video_url, stream=True, timeout=120)
    response.raise_for_status()

    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"[TASK] Video downloaded: {video_path}")
    return video_path


def _update_project_status(project_id: str, updates: Dict[str, Any]):
    """Update project JSON file with status updates"""
    script_path = Path(f"outputs/scripts/{project_id}.json")

    if script_path.exists():
        try:
            with open(script_path, 'r') as f:
                project_data = json.load(f)
        except:
            project_data = {}
    else:
        project_data = {}

    project_data.update(updates)
    project_data['lastUpdated'] = datetime.utcnow().isoformat()

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, 'w') as f:
        json.dump(project_data, f, indent=2)


# ============================================================
# TASK CHAINS (for full pipeline)
# ============================================================

def create_video_pipeline(project_id: str, script: str, duration: int,
                         voice_type: str = "male-professional",
                         reel_type: str = "ai-generated") -> str:
    """
    Create a full video generation pipeline

    Chains tasks: generate_video -> generate_voiceover -> compose_video

    Returns: Celery group task ID for tracking
    """
    from celery import chain

    pipeline = chain(
        generate_video_task.s(project_id, script, duration, reel_type),
        generate_voiceover_task.s(project_id, script, voice_type, duration),
        compose_video_task.s(project_id, duration)
    )

    result = pipeline.apply_async()
    return result.id


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == '__main__':
    """
    Start Celery worker:
    celery -A celery_tasks worker --loglevel=info --concurrency=10

    Start Celery beat (for scheduled tasks):
    celery -A celery_tasks beat --loglevel=info

    Monitor with Flower:
    celery -A celery_tasks flower --port=5555
    """
    celery_app.start()
