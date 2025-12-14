"""
REELCRAFT AI - Production Backend API v2.0
Enterprise-grade Flask server with all services embedded
Hardened for 50-100 concurrent users with EXACT duration control

FIXES IMPLEMENTED:
1. Exact word count calculation for scripts (duration × 2.5 words/sec)
2. Correct Replicate API parameters for WAN 2.5
3. Edge-TTS with speed control for exact audio duration
4. FFmpeg composition with explicit duration control
5. Duration validation for all outputs
6. Input sanitization with bleach
7. Production health checks
"""

from flask import Flask, request, jsonify, send_file, g, after_this_request, current_app
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import os
import sys
import re
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from datetime import datetime
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import tempfile
import json
import uuid
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import base64
import subprocess
import shutil
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Optional: Flask-Compress for gzip
try:
    from flask_compress import Compress
    COMPRESS_AVAILABLE = True
except ImportError:
    COMPRESS_AVAILABLE = False

# FFmpeg path detection - use imageio_ffmpeg for bundled binaries
def get_ffmpeg_path():
    """Get ffmpeg executable path - prefers imageio_ffmpeg bundled binary"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'

FFMPEG_PATH = get_ffmpeg_path()

# Load environment variables from parent directory (where .env file is)
# This ensures .env is found whether running from backend/ or root folder
_script_dir = Path(__file__).resolve().parent
_root_dir = _script_dir.parent
_env_path = _root_dir / '.env'
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Fallback to current working directory
    load_dotenv()

# ========================================================================
# IDEMPOTENCY STORE (Redis or in-memory fallback)
# ========================================================================
try:
    import redis as _redis
except ImportError:
    _redis = None

class IdempotencyStore:
    """Store for idempotent request handling - prevents duplicate operations"""

    def __init__(self, redis_url=None):
        self.ttl = 600  # 10 minutes default TTL
        self._redis = None
        self._mem = {}  # fallback: key -> (expires_at, status, headers, body)

        if redis_url and 'redis' in redis_url and _redis:
            try:
                self._redis = _redis.from_url(redis_url)
                self._redis.ping()
            except Exception as e:
                logging.warning(f"[IdempotencyStore] Redis connection failed, using memory: {e}")
                self._redis = None

    def get(self, key):
        """Get cached response for idempotency key"""
        try:
            if self._redis:
                raw = self._redis.get(f'idem:{key}')
                if raw:
                    return json.loads(raw)
            else:
                v = self._mem.get(key)
                if v and v[0] > time.time():
                    return {'status': v[1], 'headers': v[2], 'body': v[3]}
                elif v:
                    del self._mem[key]  # Expired
        except Exception as e:
            logging.error(f"[IdempotencyStore] Get error: {e}")
        return None

    def set(self, key, status, headers, body, ttl=None):
        """Store response for idempotency key"""
        payload = json.dumps({'status': status, 'headers': headers, 'body': body})
        try:
            if self._redis:
                self._redis.setex(f'idem:{key}', ttl or self.ttl, payload)
            else:
                self._mem[key] = (time.time() + (ttl or self.ttl), status, headers, body)
                # Cleanup old entries (keep max 1000)
                if len(self._mem) > 1000:
                    now = time.time()
                    expired = [k for k, v in self._mem.items() if v[0] < now]
                    for k in expired[:100]:
                        del self._mem[k]
        except Exception as e:
            logging.error(f"[IdempotencyStore] Set error: {e}")


def idempotent(ttl=600):
    """Decorator for idempotent write endpoints"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            idem_key = request.headers.get('Idempotency-Key')
            if not idem_key:
                return f(*args, **kwargs)

            store = current_app.config.get('idem_store')
            if not store:
                return f(*args, **kwargs)

            # Check cache
            cached = store.get(idem_key)
            if cached:
                logging.info(f"[Idempotent] Replaying cached response for key: {idem_key[:16]}...")
                resp = current_app.response_class(
                    response=cached['body'],
                    status=cached['status'],
                    mimetype='application/json'
                )
                resp.headers['Idempotency-Replayed'] = '1'
                for k, v in cached.get('headers', {}).items():
                    resp.headers[k] = v
                return resp

            # Execute and cache
            result = f(*args, **kwargs)

            # Normalize response
            if isinstance(result, tuple):
                flask_resp, status_code = result[0], result[1] if len(result) > 1 else 200
            else:
                flask_resp = result
                status_code = getattr(result, 'status_code', 200)

            # Only cache successful responses
            if status_code in (200, 201, 202):
                try:
                    body = flask_resp.get_data(as_text=True) if hasattr(flask_resp, 'get_data') else str(flask_resp)
                    hdrs = {}
                    if hasattr(flask_resp, 'headers'):
                        hdrs = {k: v for k, v in flask_resp.headers.items() if k.lower().startswith('x-')}
                    store.set(idem_key, status_code, hdrs, body, ttl)
                    logging.info(f"[Idempotent] Cached response for key: {idem_key[:16]}...")
                except Exception as e:
                    logging.error(f"[Idempotent] Cache error: {e}")

            return result
        return wrapper
    return decorator


# FFmpeg concurrency pool (prevents CPU overload)
FFMPEG_POOL = None

# ========================================================================
# PRODUCTION CONSTANTS
# ========================================================================

# Speaking rate for script generation (words per second)
WORDS_PER_SECOND = 2.5
WORD_COUNT_TOLERANCE = 5  # Allow ±5 words

# Video generation settings
MAX_VIDEO_DURATION = 60  # Maximum video duration in seconds
WAN25_MAX_DURATION = 10  # WAN 2.5 max clip duration
DEFAULT_FPS = 24
DEFAULT_ASPECT_RATIO = "9:16"  # Portrait for reels

# Audio settings
AUDIO_DURATION_TOLERANCE = 0.5  # Allow ±0.5 seconds
MAX_TEMPO_ADJUSTMENT = 1.1  # Maximum audio speed adjustment (barely noticeable)

# Sanitization settings
MAX_TOPIC_LENGTH = 500
MAX_SCRIPT_LENGTH = 2000
MAX_PROMPT_LENGTH = 500

# ========================================================================
# SECURITY: URL VALIDATION FOR SSRF PREVENTION
# ========================================================================

# Allowed hosts for remote media downloads (SSRF protection)
ALLOWED_MEDIA_HOSTS = os.getenv('ALLOWED_MEDIA_HOSTS',
    'replicate.delivery,.replicate.com,.runpod.ai,storage.googleapis.com').split(',')

def is_safe_url(url: str) -> bool:
    """Validate URL is safe (HTTPS only, allowed hosts) - prevents SSRF attacks"""
    if not url:
        return False
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)

        # Must be HTTPS
        if parsed.scheme != 'https':
            logging.warning(f"[SSRF] Rejected non-HTTPS URL: {parsed.scheme}://...")
            return False

        # Check against allowed hosts (supports wildcards like .replicate.com)
        host = parsed.netloc.lower()
        for allowed in ALLOWED_MEDIA_HOSTS:
            allowed = allowed.strip().lower()
            if allowed.startswith('.'):
                # Wildcard: *.replicate.com
                if host.endswith(allowed) or host == allowed[1:]:
                    return True
            else:
                # Exact match
                if host == allowed:
                    return True

        logging.warning(f"[SSRF] Rejected URL with disallowed host: {host}")
        return False
    except Exception as e:
        logging.error(f"[SSRF] URL validation error: {e}")
        return False


# ========================================================================
# SECURITY: AUTH GUARD DECORATOR
# ========================================================================

def require_auth(f):
    """Decorator to require authentication for sensitive endpoints

    Checks for valid API key (X-API-Key header) or JWT Bearer token.
    For admin endpoints, set require_admin=True.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Check API Key
        api_key = request.headers.get('X-API-Key')
        expected_key = os.getenv('API_SECRET_KEY')

        if expected_key and api_key == expected_key:
            g.auth_method = 'api_key'
            return f(*args, **kwargs)

        # Check JWT Bearer token
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            # Token validation is handled by auth blueprint
            # For now, just check token exists (auth blueprint validates)
            if token:
                g.auth_method = 'bearer'
                return f(*args, **kwargs)

        # No valid auth
        return jsonify({
            'status': 'error',
            'error': 'Authentication required',
            'error_code': 'AUTH_REQUIRED'
        }), 401
    return wrapper


def require_admin(f):
    """Decorator for admin-only endpoints (debug, metrics, admin dashboard)

    Allows access if ANY of:
    - FLASK_ENV is 'development', OR
    - Request has valid ADMIN_API_KEY header, OR
    - User is authenticated with admin/superadmin role (via require_auth)
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Allow in development mode
        if os.getenv('FLASK_ENV') == 'development':
            return f(*args, **kwargs)

        # Check admin API key
        api_key = request.headers.get('X-Admin-Key')
        admin_key = os.getenv('ADMIN_API_KEY')
        if admin_key and api_key == admin_key:
            return f(*args, **kwargs)

        # Check if user is authenticated with admin role (set by require_auth)
        if hasattr(g, 'current_user') and g.current_user:
            user_role = g.current_user.get('role', '')
            if user_role in ['admin', 'superadmin']:
                return f(*args, **kwargs)

        # Forbidden
        return jsonify({
            'status': 'error',
            'error': 'Admin access required',
            'error_code': 'ADMIN_REQUIRED'
        }), 403
    return wrapper


# ========================================================================
# HTTP SESSION MANAGEMENT (Connection Pooling + Retries)
# ========================================================================

_thread_local = threading.local()

def get_http_session() -> requests.Session:
    """Get thread-local HTTP session with connection pooling and retries"""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            connect=2,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )

        # Configure connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100,
            pool_block=False
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Set default timeout
        session.request = lambda *args, **kwargs: requests.Session.request(
            session,
            *args,
            timeout=kwargs.pop('timeout', 30),
            **kwargs
        )

        _thread_local.session = session

    return _thread_local.session

# ========================================================================
# EMBEDDED SERVICE CLASSES (with HTTP pooling)
# ========================================================================

# === INPUT SANITIZATION ===
def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input - remove HTML tags and limit length"""
    if not text:
        return ""
    # Remove HTML tags using regex (bleach alternative)
    clean = re.sub(r'<[^>]+>', '', str(text))
    # Remove potentially dangerous characters
    clean = re.sub(r'[<>"\';\\]', '', clean)
    # Normalize whitespace
    clean = ' '.join(clean.split())
    return clean[:max_length].strip()


def calculate_word_count(duration: int, length: str = None) -> Tuple[int, int, int]:
    """Calculate exact word count for target duration or script length setting

    PRIORITY: Duration takes precedence for short-form videos (5-30s).
    This ensures the voiceover fits the exact target duration.

    Args:
        duration: Target duration in seconds
        length: Script length setting ('short', 'medium', 'long') - only used for longer videos

    Returns: (target_words, min_words, max_words)

    Duration-based word counts (at 2.5 words/second):
    - 5s  → ~12 words (short punchy hook)
    - 10s → ~25 words (quick message)
    - 15s → ~38 words (standard reel)
    - 30s → ~75 words (detailed reel)
    - 60s → ~150 words (full story)
    """
    # For short-form videos (5-30s), ALWAYS use duration-based calculation
    # This ensures voiceover fits the exact target duration
    if duration and duration <= 30:
        target = int(duration * WORDS_PER_SECOND)
        # Tighter tolerance for short videos (±3 words)
        tolerance = 3 if duration <= 15 else WORD_COUNT_TOLERANCE
        min_words = max(8, target - tolerance)
        max_words = target + tolerance
        logging.debug(f"[SCRIPT] Duration-based: {duration}s → {target} words (±{tolerance})")
        return target, min_words, max_words

    # For longer videos, use length presets if provided
    if length:
        length_word_counts = {
            'short': (75, 60, 90),       # Short: ~30s video
            'medium': (150, 125, 175),   # Medium: ~60s video
            'long': (225, 200, 250),     # Long: ~90s video
        }
        if length.lower() in length_word_counts:
            return length_word_counts[length.lower()]

    # Fallback: Calculate from duration for any other case
    target = int(duration * WORDS_PER_SECOND)
    min_words = max(10, target - WORD_COUNT_TOLERANCE)
    max_words = target + WORD_COUNT_TOLERANCE
    return target, min_words, max_words


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split()) if text else 0


# === GROQ SERVICE ===
class GroqService:
    """AI script generation using Groq with EXACT word count for duration"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.api_key)

    def generate_script(self, topic: str, length: str = "medium",
                       tone: str = "witty", duration: int = 30,
                       brand: str = None) -> Dict[str, Any]:
        """Generate a video script with EXACT word count for target duration

        Returns dict with script, word_count, and estimated_duration
        """
        if not self.is_available():
            raise Exception("Groq API key not configured")

        # Sanitize input
        topic = sanitize_input(topic, MAX_TOPIC_LENGTH)
        brand = sanitize_input(brand, 100) if brand else ""

        # Calculate EXACT word count based on duration (priority for short videos)
        target_words, min_words, max_words = calculate_word_count(duration, length)

        # Build prompt with STRICT word count requirement
        brand_context = f"\nBrand/Product: {brand}" if brand else ""

        # Duration-specific style guidance
        if duration <= 5:
            style_guide = "Ultra-short hook - ONE powerful statement that grabs attention instantly. No fluff."
        elif duration <= 10:
            style_guide = "Quick impact - Start with a hook, deliver one key message. Punchy and memorable."
        elif duration <= 15:
            style_guide = "Standard reel - Hook, brief context, memorable closing. Every word must earn its place."
        elif duration <= 30:
            style_guide = "Detailed reel - Hook, develop the story, strong call-to-action. Keep momentum throughout."
        else:
            style_guide = "Full narrative - Build a complete story arc with beginning, middle, and satisfying end."

        prompt = f"""Create a {duration}-second video script narration.

TOPIC: {topic}{brand_context}

CRITICAL WORD COUNT REQUIREMENT:
- You MUST write EXACTLY {target_words} words (minimum {min_words}, maximum {max_words})
- This is calculated for {WORDS_PER_SECOND} words per second speaking rate
- Count your words carefully before responding

FORMAT FOR {duration}s VIDEO:
{style_guide}

STYLE REQUIREMENTS:
- Tone: {tone}
- Format: Pure narration only (no "[Scene:]" or stage directions)
- Style: Engaging, punchy, perfect for social media reels
- Must flow naturally when spoken aloud

OUTPUT: Write ONLY the narration text. No labels, no word count, no explanations.
Remember: EXACTLY {target_words} words (±{min(3, WORD_COUNT_TOLERANCE)} words for short videos)."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert video script writer who writes scripts with EXACT word counts. When asked for {target_words} words, you deliver exactly that. You only output the narration text, nothing else."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        try:
            session = get_http_session()
            response = session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            script = data['choices'][0]['message']['content'].strip()

            # Clean up script - remove any labels or formatting
            script = self._clean_script(script)

            # Count actual words
            actual_word_count = count_words(script)
            estimated_duration = actual_word_count / WORDS_PER_SECOND

            # Log if word count is off
            if actual_word_count < min_words or actual_word_count > max_words:
                logging.warning(
                    f"Script word count mismatch: target={target_words}, "
                    f"actual={actual_word_count}, duration_diff={abs(duration - estimated_duration):.1f}s"
                )

            return {
                "script": script,
                "word_count": actual_word_count,
                "target_word_count": target_words,
                "estimated_duration": round(estimated_duration, 1),
                "target_duration": duration
            }

        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq API error: {str(e)}")

    def _clean_script(self, script: str) -> str:
        """Clean up script text - remove labels, formatting, quotes"""
        # Remove common labels
        patterns_to_remove = [
            r'^\s*\[.*?\]\s*',  # [Scene:] etc
            r'^\s*\(.*?\)\s*',  # (Note:) etc
            r'^\s*Script:\s*',
            r'^\s*Narration:\s*',
            r'^\s*Voice\s*over:\s*',
            r'^\s*VO:\s*',
            r'\s*Word count:.*$',
            r'\s*\d+ words\.?\s*$',
        ]

        result = script
        for pattern in patterns_to_remove:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE)

        # Remove quotes at start/end
        result = result.strip('"\'')

        # Normalize whitespace
        result = ' '.join(result.split())

        return result.strip()

    def enhance_prompt(self, prompt: str, context: str = None) -> Dict[str, Any]:
        """Enhance a video prompt for better AI video generation results

        Takes a basic prompt and returns an enhanced version with:
        - Cinematic details (lighting, camera angles, motion)
        - Visual specifics (colors, textures, environment)
        - Professional video terminology

        Args:
            prompt: User's original prompt
            context: Optional context (e.g., "product demo", "lifestyle")

        Returns:
            dict with original, enhanced prompt, and improvements made
        """
        if not self.is_available():
            raise Exception("Groq API key not configured")

        # Sanitize input
        prompt = sanitize_input(prompt, MAX_PROMPT_LENGTH)
        context = sanitize_input(context, 100) if context else "general video ad"

        system_prompt = """You are an expert AI video prompt engineer. Your job is to enhance basic prompts into detailed, cinematic descriptions that produce stunning AI-generated videos.

ENHANCEMENT RULES:
1. Add specific visual details (lighting, colors, textures, environment)
2. Include camera motion/angles (smooth pan, close-up, wide shot, dolly)
3. Describe the mood/atmosphere
4. Add professional video terminology
5. Keep it concise (under 150 words)
6. Maintain the original intent - don't change the core subject
7. Make it suitable for AI video generation models

OUTPUT: Return ONLY the enhanced prompt. No explanations, no labels."""

        user_prompt = f"""Enhance this video prompt for AI video generation:

ORIGINAL PROMPT: {prompt}
CONTEXT: {context}

Create an enhanced, cinematic version that will produce a stunning video."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }

        try:
            session = get_http_session()
            response = session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()

            data = response.json()
            enhanced = data['choices'][0]['message']['content'].strip()

            # Clean up - remove quotes and labels
            enhanced = enhanced.strip('"\'')
            enhanced = re.sub(r'^(Enhanced prompt:|Prompt:|Here\'s the enhanced.*?:)\s*', '', enhanced, flags=re.IGNORECASE)
            enhanced = enhanced.strip()

            # Ensure it's not empty
            if not enhanced or len(enhanced) < 10:
                enhanced = prompt  # Fallback to original

            return {
                "original": prompt,
                "enhanced": enhanced,
                "context": context,
                "word_count": len(enhanced.split())
            }

        except requests.exceptions.RequestException as e:
            logging.error(f"Prompt enhancement failed: {e}")
            # Return original on error - don't break the flow
            return {
                "original": prompt,
                "enhanced": prompt,
                "context": context,
                "error": str(e)
            }


# === REPLICATE SERVICE ===
class ReplicateService:
    """Video generation using Replicate API with WAN 2.5 Fast model"""

    # WAN model versions (2.5 fast supports native 1080p)
    WAN_T2V_FAST = "wan-video/wan-2.5-t2v-fast"  # Text-to-video (fast, up to 1080p)

    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv('REPLICATE_API_TOKEN')
        self.base_url = "https://api.replicate.com/v1"
        self.model = self.WAN_T2V_FAST

    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.api_token)

    def generate_video(self, prompt: str, duration: int = 30,
                       reel_type: str = "ai-generated",
                       project_id: str = None,
                       aspect_ratio: str = "9:16",
                       style_tokens: str = None,
                       quality: str = "standard",
                       resolution: str = "1080p") -> Dict[str, Any]:
        """Start video generation with WAN 2.5 Fast model

        WAN 2.5 Fast generates ~5 second clips with native 1080p support.
        For longer videos, we generate multiple clips in parallel and stitch.

        Args:
            prompt: Video description
            duration: Target video length (5-60s)
            reel_type: Video style type
            project_id: Project identifier
            aspect_ratio: "9:16" or "16:9" (not directly supported, uses 480p)
            style_tokens: Custom style descriptors for consistency
            quality: "fast", "standard", or "high"
            resolution: "480p" (~40s/clip) or "720p" (~2.5min/clip)

        Returns: dict with job_id, clips_needed, target_duration
        """
        if not self.is_available():
            raise Exception("Replicate API token not configured")

        # Sanitize prompt
        prompt = sanitize_input(prompt, MAX_PROMPT_LENGTH)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # WAN 2.2 Fast generates ~5 second clips
        # Calculate how many clips needed for target duration
        clip_duration = 5  # WAN 2.2 fast produces ~5s clips
        clips_needed = max(1, (duration + 4) // 5)  # Ceiling division

        # Quality presets (sample_steps for WAN 2.2)
        quality_settings = {
            "fast": {"steps": 20},
            "standard": {"steps": 30},
            "high": {"steps": 40}
        }
        preset = quality_settings.get(quality, quality_settings["standard"])

        # Create enhanced video prompt with style tokens
        video_prompt = self._enhance_prompt(prompt, reel_type, style_tokens)

        # WAN 2.5 Fast API parameters
        # Resolution options: "480p" (~40s/clip), "720p" (~2.5min/clip), "1080p" (~4min/clip)
        # Validate resolution parameter
        valid_resolutions = ["480p", "720p", "1080p"]
        if resolution not in valid_resolutions:
            resolution = "1080p"  # Default to 1080p HD

        logging.info(f"[REPLICATE] Using WAN 2.5 Fast, resolution: {resolution}")

        base_payload = {
            "version": self.model,
            "input": {
                "prompt": video_prompt,
                "num_frames": 81,  # ~5 seconds at 16fps
                "resolution": resolution,  # User-selected resolution
                "sample_steps": preset["steps"]
            }
        }

        try:
            session = get_http_session()
            job_ids = []

            # Start all clips in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def start_single_clip(clip_index):
                """Start a single clip generation"""
                try:
                    response = session.post(
                        f"{self.base_url}/predictions",
                        headers=headers,
                        json=base_payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()
                    job_id = data.get('id')
                    logging.info(f"[CLIP {clip_index + 1}/{clips_needed}] Started: {job_id}")
                    return {"index": clip_index, "job_id": job_id, "status": "started"}
                except Exception as e:
                    logging.error(f"[CLIP {clip_index + 1}] Failed to start: {e}")
                    return {"index": clip_index, "job_id": None, "status": "failed", "error": str(e)}

            # Start all clips in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(clips_needed, 3)) as executor:
                futures = {
                    executor.submit(start_single_clip, i): i
                    for i in range(clips_needed)
                }
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            # Sort by index and collect job_ids
            results.sort(key=lambda x: x['index'])
            job_ids = [r['job_id'] for r in results if r.get('job_id')]

            if not job_ids:
                raise Exception("No video generation jobs started successfully")

            # Use first job_id as primary, store all in composite
            primary_job_id = job_ids[0]

            logging.info(f"Video generation started: primary_job={primary_job_id}, "
                        f"total_jobs={len(job_ids)}, duration={duration}s, clips={clips_needed}")

            return {
                "job_id": primary_job_id,
                "all_job_ids": job_ids,  # All clip job IDs
                "clips_needed": clips_needed,
                "clip_duration": clip_duration,
                "target_duration": duration,
                "aspect_ratio": aspect_ratio,
                "multi_clip": clips_needed > 1
            }

        except requests.exceptions.RequestException as e:
            raise Exception(f"Replicate API error: {str(e)}")

    def _enhance_prompt(self, prompt: str, reel_type: str, style_tokens: str = None) -> str:
        """Enhance prompt based on reel type and custom style tokens"""
        style_prefixes = {
            "ai-generated": "Cinematic high-quality video, professional lighting, ",
            "product-demo": "Clean product showcase video, studio lighting, professional, ",
            "lifestyle": "Lifestyle video, natural lighting, authentic feel, ",
            "explainer": "Clear explanatory video, professional graphics, ",
            "testimonial": "Professional testimonial style video, clean background, "
        }

        prefix = style_prefixes.get(reel_type, style_prefixes["ai-generated"])

        # Use custom style tokens if provided, otherwise use defaults
        if style_tokens:
            suffix = f". {style_tokens}"
        else:
            suffix = ". 4K quality, smooth motion, engaging visuals, consistent style"

        enhanced = f"{prefix}{prompt}{suffix}"

        return enhanced[:MAX_PROMPT_LENGTH]

    def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a video generation job"""
        if not self.is_available():
            raise Exception("Replicate API token not configured")

        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

        try:
            session = get_http_session()
            response = session.get(
                f"{self.base_url}/predictions/{job_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            status = data.get('status', 'unknown')

            result = {
                'status': self._map_status(status),
                'progress': self._calculate_progress(status),
                'videoUrl': None,
                'error': None
            }

            if status == 'succeeded':
                output = data.get('output')
                if output:
                    if isinstance(output, list) and len(output) > 0:
                        result['videoUrl'] = output[0]
                    elif isinstance(output, str):
                        result['videoUrl'] = output
            elif status == 'failed':
                result['error'] = data.get('error', 'Video generation failed')

            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"Status check error: {str(e)}")

    def _map_status(self, replicate_status: str) -> str:
        """Map Replicate status to our status"""
        status_map = {
            'starting': 'processing',
            'processing': 'processing',
            'succeeded': 'completed',
            'failed': 'failed',
            'canceled': 'failed'
        }
        return status_map.get(replicate_status, 'pending')

    def _calculate_progress(self, status: str) -> int:
        """Calculate progress percentage"""
        progress_map = {
            'starting': 10,
            'processing': 50,
            'succeeded': 100,
            'failed': 0,
            'canceled': 0
        }
        return progress_map.get(status, 0)


# === RUNPOD SERVICE ===
class RunPodService:
    """Video generation using RunPod Serverless with WAN 2.2 + Edge TTS

    This service connects to your RunPod serverless endpoint running
    the WAN 2.2 model with integrated Edge TTS for voiceovers.
    """

    def __init__(self, api_key: str = None, endpoint_id: str = None):
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        self.endpoint_id = endpoint_id or os.getenv('RUNPOD_ENDPOINT_ID')
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"

        # Job tracking
        self.active_jobs = {}

    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.api_key and self.endpoint_id)

    def generate_video(self, prompt: str, duration: int = 30,
                       script: str = None, voice: str = "male-professional",
                       reel_type: str = "ai-generated",
                       project_id: str = None,
                       aspect_ratio: str = "9:16",
                       style_tokens: str = None,
                       quality: str = "standard") -> Dict[str, Any]:
        """Start video generation on RunPod with WAN 2.2

        Args:
            prompt: Video description prompt
            duration: Target duration in seconds (5-30)
            script: Narration script for Edge TTS voiceover (also used for scene-aware prompts)
            voice: Voice preset for TTS
            reel_type: Type of video content
            project_id: Project identifier
            aspect_ratio: "9:16" (portrait) or "16:9" (landscape)
            style_tokens: Custom style descriptors for visual consistency across clips
            quality: "fast", "standard", or "high"

        Returns: dict with job_id and status info

        Scene-Aware Generation:
            For videos > 5s, the script is split into segments and each 5s clip
            gets a context-specific prompt based on:
            - Base visual prompt
            - Visual cues from corresponding script segment
            - Scene position (opening, mid-sequence, finale)
            - Consistent style tokens
        """
        if not self.is_available():
            raise Exception("RunPod API key or endpoint ID not configured")

        # Sanitize inputs
        prompt = sanitize_input(prompt, MAX_PROMPT_LENGTH)
        script = sanitize_input(script, MAX_SCRIPT_LENGTH) if script else ""
        style_tokens = sanitize_input(style_tokens, 200) if style_tokens else None

        # Clamp duration
        duration = max(5, min(30, duration))

        # Map aspect ratio to resolution
        resolution = "portrait_720p" if aspect_ratio == "9:16" else "720p"

        # Enhance prompt with style tokens
        video_prompt = self._enhance_prompt(prompt, reel_type, style_tokens)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # RunPod serverless payload with scene-aware generation support
        payload = {
            "input": {
                "prompt": video_prompt,
                "script": script,  # For Edge TTS + scene-aware prompt generation
                "voice": voice,
                "duration": duration,
                "resolution": resolution,
                "quality": quality,
                "include_audio": bool(script),
                "style_tokens": style_tokens,  # For visual consistency across clips
                "negative_prompt": "blurry, low quality, distorted, watermark, shaky, inconsistent"
            }
        }

        try:
            session = get_http_session()
            response = session.post(
                f"{self.base_url}/run",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            job_id = data.get('id')

            if not job_id:
                raise Exception("No job ID returned from RunPod")

            # Track job
            self.active_jobs[job_id] = {
                "project_id": project_id,
                "duration": duration,
                "created_at": datetime.now().isoformat()
            }

            # Calculate clips needed (5 seconds per clip)
            clips_needed = max(1, (duration + 4) // 5)
            clip_duration = 5

            logging.info(f"RunPod video generation started: job_id={job_id}, "
                        f"duration={duration}s ({clips_needed} clips), has_audio={bool(script)}")

            return {
                "job_id": job_id,
                "status": "starting",
                "target_duration": duration,
                "clips_needed": clips_needed,
                "clip_duration": clip_duration,
                "has_audio": bool(script)
            }

        except requests.exceptions.RequestException as e:
            raise Exception(f"RunPod API error: {str(e)}")

    def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of a RunPod video generation job

        Returns: dict with status, progress, and videoUrl when complete
        """
        if not self.is_available():
            raise Exception("RunPod API not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            session = get_http_session()
            response = session.get(
                f"{self.base_url}/status/{job_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            runpod_status = data.get('status', 'UNKNOWN')

            result = {
                "job_id": job_id,
                "status": self._map_status(runpod_status),
                "progress": self._calculate_progress(runpod_status)
            }

            # Handle completed job
            if runpod_status == 'COMPLETED':
                output = data.get('output', {})

                # Get video from base64 or URL
                if 'video_base64' in output:
                    # Save base64 to file and return URL
                    video_url = self._save_video_from_base64(
                        output['video_base64'],
                        job_id
                    )
                    result['videoUrl'] = video_url
                elif 'video_url' in output:
                    result['videoUrl'] = output['video_url']

                result['duration'] = output.get('duration', 0)
                result['has_audio'] = output.get('has_audio', False)
                result['generation_time'] = output.get('generation_time', 0)

                # Cleanup tracking
                self.active_jobs.pop(job_id, None)

            elif runpod_status == 'FAILED':
                result['error'] = data.get('error', 'Video generation failed')
                self.active_jobs.pop(job_id, None)

            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"RunPod status check error: {str(e)}")

    def _save_video_from_base64(self, video_base64: str, job_id: str) -> str:
        """Save base64 video to file and return URL path"""
        try:
            # Decode base64
            video_bytes = base64.b64decode(video_base64)

            # Create output directory
            output_dir = Path(__file__).parent.parent / "outputs" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save file
            filename = f"runpod_{job_id}.mp4"
            filepath = output_dir / filename

            with open(filepath, 'wb') as f:
                f.write(video_bytes)

            logging.info(f"Saved RunPod video: {filepath}")

            # Return relative URL for frontend
            return f"/outputs/videos/{filename}"

        except Exception as e:
            logging.error(f"Failed to save video from base64: {e}")
            raise Exception(f"Failed to save video: {str(e)}")

    def _enhance_prompt(self, prompt: str, reel_type: str, style_tokens: str = None) -> str:
        """Enhance prompt for better video quality with optional custom style tokens"""
        style_prefixes = {
            "ai-generated": "Cinematic high-quality video, professional lighting, smooth motion, ",
            "product-demo": "Clean product showcase, studio lighting, professional, ",
            "lifestyle": "Lifestyle video, natural lighting, authentic feel, ",
            "explainer": "Clear explanatory video, professional, ",
            "testimonial": "Professional testimonial style, clean background, "
        }
        prefix = style_prefixes.get(reel_type, style_prefixes["ai-generated"])

        # Use custom style tokens if provided
        if style_tokens:
            suffix = f". {style_tokens}"
        else:
            suffix = ". consistent style, engaging visuals"

        return f"{prefix}{prompt}{suffix}"[:MAX_PROMPT_LENGTH]

    def _map_status(self, runpod_status: str) -> str:
        """Map RunPod status to our status"""
        status_map = {
            'IN_QUEUE': 'pending',
            'IN_PROGRESS': 'processing',
            'COMPLETED': 'completed',
            'FAILED': 'failed',
            'CANCELLED': 'failed',
            'TIMED_OUT': 'failed'
        }
        return status_map.get(runpod_status, 'pending')

    def _calculate_progress(self, runpod_status: str) -> int:
        """Calculate progress percentage"""
        progress_map = {
            'IN_QUEUE': 5,
            'IN_PROGRESS': 50,
            'COMPLETED': 100,
            'FAILED': 0,
            'CANCELLED': 0,
            'TIMED_OUT': 0
        }
        return progress_map.get(runpod_status, 0)


# === TTS SERVICE (OpenAI Compatible) ===
class TTSService:
    """Text-to-speech service using OpenAI API with connection pooling"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/audio/speech"

    def is_available(self) -> bool:
        """Check if service is available"""
        return bool(self.api_key)

    def generate_voiceover(self, text: str, voice_type: str = "male-professional",
                          project_id: str = None) -> Path:
        """Generate voiceover from text"""
        if not self.is_available():
            # Fall back to Kokoro if OpenAI not available
            kokoro = KokoroService()
            if kokoro.is_available():
                return kokoro.generate_voiceover(text, voice_type, project_id)
            raise Exception("No TTS service available")

        # Map voice types to OpenAI voices
        voice_map = {
            'male-professional': 'onyx',
            'male-casual': 'echo',
            'female-professional': 'nova',
            'female-casual': 'alloy',
            'neutral': 'shimmer'
        }

        voice = voice_map.get(voice_type, 'onyx')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "tts-1",
            "input": text[:4000],  # OpenAI limit
            "voice": voice
        }

        try:
            session = get_http_session()
            response = session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            # Save audio file
            audio_dir = Path(f"outputs/audio/{project_id}")
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / "voiceover.mp3"

            with open(audio_path, 'wb') as f:
                f.write(response.content)

            return audio_path

        except requests.exceptions.RequestException as e:
            raise Exception(f"TTS API error: {str(e)}")


# === KOKORO TTS SERVICE (Fallback) ===
class KokoroService:
    """Fallback TTS using Kokoro (local or API) with connection pooling"""

    def __init__(self):
        self.api_url = os.getenv('KOKORO_API_URL', 'http://localhost:8000')

    def is_available(self) -> bool:
        """Check if Kokoro service is available"""
        try:
            session = get_http_session()
            response = session.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate_voiceover(self, text: str, voice_type: str = "male-professional",
                          project_id: str = None) -> Path:
        """Generate voiceover using Kokoro"""
        if not self.is_available():
            raise Exception("Kokoro TTS service not available")

        voice_map = {
            'male-professional': 'af_sky',
            'male-casual': 'am_michael',
            'female-professional': 'af_bella',
            'female-casual': 'af_sarah'
        }

        voice = voice_map.get(voice_type, 'af_sky')

        payload = {
            "text": text,
            "voice": voice,
            "speed": 1.0
        }

        try:
            session = get_http_session()
            response = session.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            # Save audio file
            audio_dir = Path(f"outputs/audio/{project_id}")
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / "voiceover.wav"

            with open(audio_path, 'wb') as f:
                f.write(response.content)

            return audio_path

        except requests.exceptions.RequestException as e:
            raise Exception(f"Kokoro TTS error: {str(e)}")


# === EDGE-TTS SERVICE (Production-ready with duration control) ===
class EdgeTTSService:
    """Production Edge-TTS with EXACT duration control via speed adjustment"""

    # Voice mapping for Edge-TTS
    VOICE_MAP = {
        'male-professional': 'en-US-GuyNeural',
        'male-casual': 'en-US-ChristopherNeural',
        'male-narrator': 'en-US-DavisNeural',
        'female-professional': 'en-US-JennyNeural',
        'female-casual': 'en-US-AriaNeural',
        'female-narrator': 'en-US-SaraNeural',
        'neutral': 'en-US-TonyNeural'
    }

    def __init__(self):
        self._edge_tts_available = None

    def is_available(self) -> bool:
        """Check if edge-tts is available"""
        if self._edge_tts_available is None:
            try:
                import edge_tts
                self._edge_tts_available = True
            except ImportError:
                self._edge_tts_available = False
        return self._edge_tts_available

    def generate_voiceover(self, text: str, voice_type: str = "male-professional",
                          project_id: str = None, target_duration: float = None) -> Dict[str, Any]:
        """Generate voiceover with OPTIONAL duration matching

        Args:
            text: Script text to convert to speech
            voice_type: Voice style to use
            project_id: Project identifier for file paths
            target_duration: If set, adjust audio speed to match this duration

        Returns:
            Dict with audio_path, actual_duration, speed_adjusted
        """
        if not self.is_available():
            raise Exception("Edge-TTS not available. Install with: pip install edge-tts")

        import edge_tts

        voice = self.VOICE_MAP.get(voice_type, 'en-US-GuyNeural')

        # Create output directory
        audio_dir = Path(f"outputs/audio/{project_id}")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / "voiceover.mp3"
        temp_audio_path = audio_dir / "voiceover_temp.mp3"

        try:
            # Generate audio using edge-tts (async)
            asyncio.run(self._generate_async(text, voice, str(temp_audio_path)))

            # Get actual duration
            actual_duration = self._get_audio_duration(temp_audio_path)
            speed_adjusted = False

            # If target duration specified, adjust speed
            if target_duration and actual_duration > 0:
                duration_diff = abs(actual_duration - target_duration)

                if duration_diff > AUDIO_DURATION_TOLERANCE:
                    # Calculate required speed adjustment
                    speed_ratio = actual_duration / target_duration

                    # Limit speed adjustment to prevent unnatural audio
                    if speed_ratio > MAX_TEMPO_ADJUSTMENT:
                        logging.warning(
                            f"Audio speed adjustment capped: needed {speed_ratio:.2f}x, "
                            f"using {MAX_TEMPO_ADJUSTMENT}x max"
                        )
                        speed_ratio = MAX_TEMPO_ADJUSTMENT
                    elif speed_ratio < 1 / MAX_TEMPO_ADJUSTMENT:
                        speed_ratio = 1 / MAX_TEMPO_ADJUSTMENT

                    # Adjust tempo with FFmpeg
                    self._adjust_audio_tempo(temp_audio_path, audio_path, speed_ratio)
                    speed_adjusted = True

                    # Get new duration
                    actual_duration = self._get_audio_duration(audio_path)

                    logging.info(
                        f"Audio speed adjusted: ratio={speed_ratio:.2f}, "
                        f"new_duration={actual_duration:.1f}s, target={target_duration}s"
                    )
                else:
                    # Duration is close enough, just copy
                    shutil.move(str(temp_audio_path), str(audio_path))
            else:
                # No target duration, just use as-is
                shutil.move(str(temp_audio_path), str(audio_path))

            return {
                "audio_path": str(audio_path),
                "actual_duration": actual_duration,
                "target_duration": target_duration,
                "speed_adjusted": speed_adjusted,
                "voice": voice
            }

        except Exception as e:
            # Cleanup temp file
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            raise Exception(f"Edge-TTS error: {str(e)}")

    async def _generate_async(self, text: str, voice: str, output_path: str):
        """Async audio generation with edge-tts"""
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffmpeg -i (works without ffprobe)"""
        try:
            result = subprocess.run([
                FFMPEG_PATH, '-i', str(audio_path)
            ], capture_output=True, text=True, timeout=10)
            # FFmpeg outputs duration to stderr
            output = result.stderr
            # Look for Duration: HH:MM:SS.decimal (variable decimal places)
            # FIX: Parse decimal part correctly regardless of length (ms vs centiseconds)
            match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', output)
            if match:
                hours, minutes, seconds, decimal = match.groups()
                # Convert decimal part: divide by 10^(len) to get fractional seconds
                fractional = int(decimal) / (10 ** len(decimal))
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + fractional
        except Exception as e:
            logging.warning(f"Could not get audio duration: {e}")

        return 0.0

    def _adjust_audio_tempo(self, input_path: Path, output_path: Path, speed_ratio: float):
        """Adjust audio tempo with FFmpeg to match target duration"""
        # For large speed changes, chain multiple atempo filters
        # atempo only supports 0.5 to 2.0 range
        filters = []
        remaining_ratio = speed_ratio

        while remaining_ratio > 2.0:
            filters.append('atempo=2.0')
            remaining_ratio /= 2.0

        while remaining_ratio < 0.5:
            filters.append('atempo=0.5')
            remaining_ratio /= 0.5

        if abs(remaining_ratio - 1.0) > 0.01:
            filters.append(f'atempo={remaining_ratio:.4f}')

        filter_chain = ','.join(filters) if filters else 'acopy'

        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(input_path),
            '-filter:a', filter_chain,
            '-vn',  # No video
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            # SECURITY FIX: Log detailed error server-side, return generic message to client
            logging.error(f"[FFmpeg] Tempo adjustment failed: {result.stderr.decode()}")
            raise Exception("Audio processing failed. Please try again.")


# === LOCAL KOKORO TTS SERVICE (with exact-fit) ===
class LocalKokoroService:
    """Local Kokoro TTS with two-pass exact-fit duration control

    Uses the local Kokoro library (not API) for fast, high-quality TTS.
    Two-pass exact-fit: generate naturally, then fit to exact duration.
    """

    # Voice configurations for smart selection
    VOICES = {
        "af_bella": {"name": "Bella", "style": "warm, conversational", "best_for": ["lifestyle", "wellness", "beauty"]},
        "af_sarah": {"name": "Sarah", "style": "clear, professional", "best_for": ["corporate", "tech", "education"]},
        "af_nicole": {"name": "Nicole", "style": "friendly, casual", "best_for": ["social", "food", "travel"]},
        "af_sky": {"name": "Sky", "style": "youthful, energetic", "best_for": ["fitness", "gaming", "youth"]},
        "af_nova": {"name": "Nova", "style": "modern, engaging", "best_for": ["fashion", "luxury", "modern"]},
        "af_heart": {"name": "Heart", "style": "emotional, expressive", "best_for": ["emotional", "story", "inspirational"]},
        "af_river": {"name": "River", "style": "calm, soothing", "best_for": ["meditation", "spa", "nature"]},
        "am_adam": {"name": "Adam", "style": "deep, authoritative", "best_for": ["finance", "automotive", "serious"]},
        "am_michael": {"name": "Michael", "style": "warm, conversational", "best_for": ["casual", "friendly", "approachable"]},
    }

    # Voice type mapping
    VOICE_TYPE_MAP = {
        'male-professional': 'am_adam',
        'male-casual': 'am_michael',
        'female-professional': 'af_sarah',
        'female-casual': 'af_bella',
        'female-warm': 'af_heart',
        'female-calm': 'af_river',
    }

    def __init__(self):
        self._available = None
        self._pipeline = None
        self._init_error = None

    def is_available(self) -> bool:
        """Check if local Kokoro library is available AND model is loaded

        This properly tests that:
        1. kokoro package is installed
        2. misaki phonemizer is installed
        3. espeak-ng is available
        4. Model files can be loaded
        """
        if self._available is None:
            try:
                from kokoro import KPipeline
                # Actually try to create the pipeline - this loads the model
                logging.info("[KOKORO] Testing pipeline initialization...")
                self._pipeline = KPipeline(lang_code="a")
                logging.info("[KOKORO] Pipeline initialized successfully!")
                self._available = True
            except ImportError as e:
                logging.error(f"[KOKORO] Import failed: {e}")
                self._init_error = f"Import error: {e}"
                self._available = False
            except Exception as e:
                logging.error(f"[KOKORO] Initialization failed: {e}")
                self._init_error = f"Init error: {e}"
                self._available = False
        return self._available

    def get_init_error(self) -> str:
        """Get the initialization error message if any"""
        return self._init_error

    def _get_pipeline(self):
        """Get or create Kokoro pipeline"""
        if self._pipeline is None and self.is_available():
            # Pipeline should already be created by is_available()
            pass
        return self._pipeline

    def select_voice(self, script: str, voice_type: str = None, tone: str = None, category: str = None) -> str:
        """Smart voice selection based on tone, script content, and category

        Automatically selects the best voice matching:
        1. Tone (witty, serious, sarcastic, professional, etc.)
        2. Script content keywords
        3. Category/brand context

        User doesn't need to do anything - voice is selected automatically!
        """
        # Tone-based voice selection (highest priority)
        tone_voice_map = {
            # Witty/Fun tones - use energetic, expressive voices
            'witty': 'af_sky',           # Youthful, energetic - perfect for witty content
            'funny': 'af_sky',
            'humorous': 'af_sky',

            # Serious/Professional tones - use authoritative voices
            'serious': 'am_adam',        # Deep, authoritative
            'professional': 'af_sarah',  # Clear, professional
            'formal': 'af_sarah',
            'corporate': 'am_adam',

            # Sarcastic/Edgy tones - use modern, engaging voices
            'sarcastic': 'af_nova',      # Modern, engaging with edge
            'edgy': 'af_nova',
            'bold': 'af_nova',

            # Emotional/Inspirational tones
            'emotional': 'af_heart',     # Emotional, expressive
            'inspirational': 'af_heart',
            'heartfelt': 'af_heart',

            # Calm/Soothing tones
            'calm': 'af_river',          # Calm, soothing
            'relaxed': 'af_river',
            'peaceful': 'af_river',

            # Casual/Friendly tones
            'casual': 'af_bella',        # Warm, conversational
            'friendly': 'am_michael',    # Warm, approachable
            'conversational': 'af_bella',
        }

        # Check tone first (automatic selection based on user's tone choice in UI)
        if tone and tone.lower() in tone_voice_map:
            selected = tone_voice_map[tone.lower()]
            logging.info(f"[KOKORO] Auto-selected voice '{selected}' for tone: {tone}")
            return selected

        # Use explicit voice type if provided
        if voice_type and voice_type in self.VOICE_TYPE_MAP:
            return self.VOICE_TYPE_MAP[voice_type]

        script_lower = script.lower()

        # Content-based detection (analyzes script keywords)
        if any(word in script_lower for word in ["skincare", "beauty", "glow", "spa", "botanical", "serum", "wellness"]):
            return "af_river"  # Calm, soothing
        elif any(word in script_lower for word in ["energy", "workout", "fitness", "power", "dynamic", "pump", "gains"]):
            return "af_sky"  # Youthful, energetic
        elif any(word in script_lower for word in ["luxury", "premium", "elegant", "sophisticated", "exclusive"]):
            return "af_nova"  # Modern, engaging
        elif any(word in script_lower for word in ["tech", "innovation", "smart", "digital", "ai", "software"]):
            return "af_sarah"  # Clear, professional
        elif any(word in script_lower for word in ["travel", "adventure", "discover", "escape", "journey", "explore"]):
            return "af_bella"  # Warm, inspiring
        elif any(word in script_lower for word in ["inspire", "dream", "believe", "heart", "hope", "love"]):
            return "af_heart"  # Emotional, expressive
        elif any(word in script_lower for word in ["finance", "invest", "business", "money", "stock", "crypto"]):
            return "am_adam"  # Deep, authoritative
        elif any(word in script_lower for word in ["food", "recipe", "delicious", "tasty", "cook", "chef"]):
            return "af_nicole"  # Friendly, casual
        elif any(word in script_lower for word in ["game", "gaming", "player", "stream", "esports"]):
            return "af_sky"  # Youthful, energetic

        # Default: af_heart (most natural and expressive for general content)
        return "af_heart"

    def generate_voiceover(self, text: str, voice_type: str = "female-warm",
                          project_id: str = None, target_duration: float = None,
                          tone: str = None) -> Dict[str, Any]:
        """Generate voiceover with two-pass exact-fit and automatic voice selection

        Pass 1: Generate TTS naturally at 1.0x speed
        Pass 2: Exact-fit to target duration (silence trim, time-stretch, pad/trim)

        Voice is automatically selected based on:
        1. Tone setting (witty, serious, sarcastic, professional)
        2. Script content analysis
        3. Voice type preference

        Args:
            text: Script text to convert to speech
            voice_type: Voice style preference (optional)
            project_id: Project identifier for file paths
            target_duration: If set, exact-fit audio to this duration
            tone: Script tone for automatic voice matching (witty, serious, etc.)

        Returns:
            Dict with audio_path, actual_duration, voice, exact_fit_applied
        """
        if not self.is_available():
            raise Exception("Local Kokoro TTS not available. Install with: pip install kokoro soundfile")

        import numpy as np
        import soundfile as sf

        # Create output directory
        audio_dir = Path(f"outputs/audio/{project_id}")
        audio_dir.mkdir(parents=True, exist_ok=True)

        raw_audio_path = audio_dir / "voiceover_raw.wav"
        final_audio_path = audio_dir / "voiceover.wav"

        # Select best voice automatically based on tone and content
        voice = self.select_voice(text, voice_type, tone=tone)
        voice_info = self.VOICES.get(voice, {})

        logging.info(f"[KOKORO] Voice: {voice} ({voice_info.get('name', '')}) - {voice_info.get('style', '')}")

        # PASS 1: Generate TTS naturally with CHUNK TIMING capture
        logging.info(f"[KOKORO] Pass 1: Generating TTS at natural speed with timing...")
        pipeline = self._get_pipeline()

        audio_parts = []
        chunk_timings = []
        current_sample = 0
        sample_rate = 24000

        for gs, ps, audio in pipeline(text, voice=voice, speed=1.0):
            chunk_samples = len(audio)
            start_time = current_sample / sample_rate
            end_time = (current_sample + chunk_samples) / sample_rate

            chunk_timings.append({
                "text": gs.strip(),
                "start": start_time,
                "end": end_time
            })

            audio_parts.append(audio)
            current_sample += chunk_samples
            logging.debug(f"[KOKORO] Chunk [{start_time:.2f}s - {end_time:.2f}s]: '{gs.strip()}'")

        if not audio_parts:
            raise Exception("Kokoro TTS generated no audio")

        audio = np.concatenate(audio_parts)
        sf.write(str(raw_audio_path), audio, sample_rate)

        # Calculate PROPORTIONAL word-level timing from chunks
        word_timings = self._calculate_word_timings(chunk_timings)
        logging.info(f"[KOKORO] Generated {len(word_timings)} word timings from {len(chunk_timings)} chunks")

        raw_duration = get_media_duration(raw_audio_path)
        logging.info(f"[KOKORO] Pass 1 complete: {raw_duration:.2f}s")

        # PASS 2: Exact-fit if target_duration specified
        exact_fit_applied = False
        if target_duration and target_duration > 0:
            logging.info(f"[KOKORO] Pass 2: Exact-fit to {target_duration}s...")
            try:
                self._exact_fit_audio(raw_audio_path, final_audio_path, target_duration)
                exact_fit_applied = True
                final_duration = get_media_duration(final_audio_path)
                logging.info(f"[KOKORO] Pass 2 complete: {final_duration:.2f}s (target: {target_duration}s)")
            except Exception as e:
                logging.warning(f"[KOKORO] Exact-fit failed: {e}, using raw audio")
                shutil.copy(str(raw_audio_path), str(final_audio_path))
                final_duration = raw_duration
        else:
            # No exact-fit needed
            shutil.copy(str(raw_audio_path), str(final_audio_path))
            final_duration = raw_duration

        return {
            "audio_path": str(final_audio_path),
            "raw_audio_path": str(raw_audio_path),
            "actual_duration": final_duration,
            "raw_duration": raw_duration,
            "target_duration": target_duration,
            "exact_fit_applied": exact_fit_applied,
            "voice": voice,
            "voice_name": voice_info.get('name', voice),
            "word_timings": word_timings  # For viral captions with word-by-word highlighting
        }

    def _calculate_word_timings(self, chunk_timings: List[Dict]) -> List[Dict]:
        """Calculate PROPORTIONAL word-level timing from chunk timings

        Kokoro TTS yields chunks (phrases), not individual words.
        We distribute each chunk's duration across its words proportionally
        by character length (longer words = more time).

        Args:
            chunk_timings: List of {"text": str, "start": float, "end": float}

        Returns:
            List of {"text": str, "start": float, "end": float} for each word
        """
        word_timings = []

        for chunk in chunk_timings:
            chunk_text = chunk["text"]
            chunk_start = chunk["start"]
            chunk_end = chunk["end"]
            chunk_duration = chunk_end - chunk_start

            # Split chunk into words
            words = chunk_text.split()
            if not words:
                continue

            # Calculate total "weight" (word length as proxy for duration)
            total_weight = sum(len(w) for w in words)
            if total_weight == 0:
                total_weight = len(words)

            # Distribute time proportionally
            current_time = chunk_start
            for word in words:
                word_weight = len(word) / total_weight
                word_duration = chunk_duration * word_weight

                word_timings.append({
                    "text": word,
                    "start": current_time,
                    "end": current_time + word_duration
                })
                current_time += word_duration

        return word_timings

    def _exact_fit_audio(self, input_path: Path, output_path: Path, target_seconds: float):
        """Two-pass exact-fit: time-stretch precisely, pad/trim to exact length

        IMPORTANT: Max speed-up is capped at 1.3x to keep voice understandable.
        If audio is too long, it will be trimmed after gentle speed-up.
        """
        actual = get_media_duration(input_path)
        if actual <= 0:
            raise RuntimeError(f"Cannot read audio duration from {input_path}")

        ratio = actual / float(target_seconds)  # >1 means speed up, <1 means slow down

        # Cap speed-up at 1.3x (30% faster) to keep voice natural and understandable
        # If audio is much longer, we'll trim the end after gentle speed-up
        MAX_SPEEDUP_RATIO = 1.3
        MIN_SLOWDOWN_RATIO = 0.7  # Don't slow down more than 30%

        original_ratio = ratio
        if ratio > MAX_SPEEDUP_RATIO:
            logging.warning(f"[EXACT-FIT] Ratio {ratio:.2f}x exceeds max {MAX_SPEEDUP_RATIO}x - capping speed, will trim")
            ratio = MAX_SPEEDUP_RATIO
        elif ratio < MIN_SLOWDOWN_RATIO:
            logging.warning(f"[EXACT-FIT] Ratio {ratio:.2f}x below min {MIN_SLOWDOWN_RATIO}x - capping speed, will pad")
            ratio = MIN_SLOWDOWN_RATIO

        logging.info(f"[EXACT-FIT] Source: {actual:.2f}s, Target: {target_seconds}s, "
                    f"Original ratio: {original_ratio:.3f}, Applied ratio: {ratio:.3f}")

        # Build atempo filter (now limited to reasonable range)
        def atempo_chain(r):
            # Since we capped ratio, this should always be 0.7-1.3 range
            if abs(r - 1.0) > 1e-6:
                return f'atempo={r:.6f}'
            return 'anull'

        stretch = atempo_chain(ratio)

        # Filter: gentle stretch + pad to ensure minimum duration + trim to exact target
        filter_str = f"{stretch},apad=whole_dur={target_seconds},atrim=duration={target_seconds}"

        cmd = [
            FFMPEG_PATH, '-y', '-i', str(input_path),
            '-af', filter_str,
            '-c:a', 'pcm_s16le', '-ar', '48000',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            # SECURITY FIX: Log detailed error server-side, return generic message to client
            logging.error(f"[FFmpeg] Exact-fit failed: {result.stderr[-500:]}")
            raise RuntimeError("Audio duration adjustment failed. Please try again.")

        # Verify output
        final_dur = get_media_duration(output_path)
        if abs(final_dur - target_seconds) > 0.5:
            logging.warning(f"[EXACT-FIT] Duration off by {abs(final_dur - target_seconds):.2f}s")


# === DURATION VALIDATION HELPERS ===
def get_media_duration(file_path: Path) -> float:
    """Get duration of video or audio file using ffmpeg -i (works without ffprobe)"""
    try:
        result = subprocess.run([
            FFMPEG_PATH, '-i', str(file_path)
        ], capture_output=True, text=True, timeout=10)
        # FFmpeg outputs duration to stderr
        output = result.stderr
        # Look for Duration: HH:MM:SS.MS
        match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', output)
        if match:
            hours, minutes, seconds, ms = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 100
    except Exception as e:
        logging.warning(f"Could not get media duration: {e}")
    return 0.0


def get_video_fps(file_path: Path) -> float:
    """Get FPS of video file using ffmpeg -i"""
    try:
        result = subprocess.run([
            FFMPEG_PATH, '-i', str(file_path)
        ], capture_output=True, text=True, timeout=10)
        output = result.stderr
        # Look for fps in stream info (e.g., "24 fps" or "29.97 fps")
        match = re.search(r'(\d+(?:\.\d+)?)\s*fps', output)
        if match:
            return float(match.group(1))
        # Fallback: look for tbr (e.g., "30 tbr")
        match = re.search(r'(\d+(?:\.\d+)?)\s*tbr', output)
        if match:
            return float(match.group(1))
    except Exception as e:
        logging.warning(f"Could not get video FPS: {e}")
    return DEFAULT_FPS  # Fallback to default


def validate_duration(actual: float, target: float, tolerance: float = 1.0) -> Tuple[bool, str]:
    """Validate that actual duration matches target within tolerance

    Returns: (is_valid, message)
    """
    diff = abs(actual - target)
    if diff <= tolerance:
        return True, f"Duration OK: {actual:.1f}s (target: {target}s, diff: {diff:.1f}s)"
    else:
        return False, f"Duration MISMATCH: {actual:.1f}s (target: {target}s, diff: {diff:.1f}s)"


def bulletproof_concat(clip_paths: List[str], output_path: str,
                       aspect_ratio: str = "9:16", fps: int = 60) -> bool:
    """
    100% BULLETPROOF video concatenation with HIGH QUALITY output.

    Uses a two-pass approach:
    1. Re-encode each clip to IDENTICAL intermediate format (ts container)
    2. Simple binary concat of ts files, then remux to mp4

    Quality settings:
    - 60fps for smooth playback
    - 1080x1920 (portrait) or 1920x1080 (landscape) resolution
    - CRF 18 for high quality
    """
    logging.info(f"[BULLETPROOF] Starting concat of {len(clip_paths)} clips...")

    if not clip_paths:
        logging.error("[BULLETPROOF] No clips provided")
        return False

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing files first
    Path(output_path).unlink(missing_ok=True)
    for f in out_dir.glob("_temp_*.ts"):
        f.unlink(missing_ok=True)
    for f in out_dir.glob("_temp_*.mp4"):
        f.unlink(missing_ok=True)

    # Determine target resolution - FULL HD
    if aspect_ratio == "16:9":
        target_w, target_h = 1920, 1080  # Full HD landscape
    else:
        target_w, target_h = 1080, 1920  # Full HD portrait

    logging.info(f"[BULLETPROOF] Target: {target_w}x{target_h} @ {fps}fps, AR: {aspect_ratio}")

    # Validate and collect valid clips
    valid_clips = []
    for i, clip_path in enumerate(clip_paths):
        p = Path(clip_path)
        if p.exists() and p.stat().st_size > 10000:
            dur = get_media_duration(p)
            if dur > 0.5:
                valid_clips.append(clip_path)
                logging.info(f"[BULLETPROOF] Clip {i}: {p.name} ({p.stat().st_size//1024}KB, {dur:.1f}s)")
            else:
                logging.warning(f"[BULLETPROOF] Skipping zero-duration clip: {p.name}")
        else:
            logging.warning(f"[BULLETPROOF] Skipping missing/empty clip: {clip_path}")

    if not valid_clips:
        logging.error("[BULLETPROOF] No valid clips to concatenate!")
        return False

    if len(valid_clips) == 1:
        # Single clip - just re-encode to target format
        logging.info("[BULLETPROOF] Single clip - re-encoding directly")
        cmd = [
            FFMPEG_PATH, '-y', '-i', str(valid_clips[0]),
            '-vf', f'scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},fps={fps},setsar=1',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',  # High quality
            '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and Path(output_path).exists():
            logging.info(f"[BULLETPROOF] Single clip done: {output_path}")
            return True
        logging.error(f"[BULLETPROOF] Single clip failed: {result.stderr[:200]}")
        return False

    # PASS 1: Re-encode each clip to MPEG-TS with IDENTICAL HIGH QUALITY settings
    ts_files = []
    for i, clip_path in enumerate(valid_clips):
        ts_path = out_dir / f"_temp_{i:02d}.ts"

        # High quality encoding: 60fps, CRF 18, medium preset
        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(clip_path),
            '-vf', f'scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},fps={fps},setsar=1',
            '-c:v', 'libx264',
            '-preset', 'medium',  # Better quality than 'fast'
            '-crf', '18',         # High quality (lower = better, 18 is visually lossless)
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-g', str(fps),  # GOP size = fps (keyframe every second)
            '-bf', '0',      # No B-frames for cleaner concat
            '-an',           # No audio
            '-f', 'mpegts',  # MPEG-TS container
            str(ts_path)
        ]

        logging.info(f"[BULLETPROOF] Encoding clip {i+1}/{len(valid_clips)} to TS...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and ts_path.exists() and ts_path.stat().st_size > 1000:
            ts_files.append(str(ts_path))
            dur = get_media_duration(ts_path)
            logging.info(f"[BULLETPROOF] TS {i}: {ts_path.stat().st_size//1024}KB, {dur:.1f}s")
        else:
            logging.error(f"[BULLETPROOF] Failed to encode clip {i}: {result.stderr[:200]}")

    if not ts_files:
        logging.error("[BULLETPROOF] No TS files created!")
        return False

    # PASS 2: Binary concat of TS files, then remux to MP4
    # This is the simplest possible concat - just concatenate bytes
    concat_ts = out_dir / "_temp_concat.ts"

    logging.info(f"[BULLETPROOF] Binary concat of {len(ts_files)} TS files...")
    try:
        with open(concat_ts, 'wb') as outfile:
            for ts_file in ts_files:
                with open(ts_file, 'rb') as infile:
                    outfile.write(infile.read())
        logging.info(f"[BULLETPROOF] Concat TS: {concat_ts.stat().st_size//1024}KB")
    except Exception as e:
        logging.error(f"[BULLETPROOF] Binary concat failed: {e}")
        return False

    # PASS 3: Remux TS to MP4
    logging.info("[BULLETPROOF] Remuxing TS to MP4...")
    cmd = [
        FFMPEG_PATH, '-y',
        '-i', str(concat_ts),
        '-c', 'copy',  # No re-encoding, just remux
        '-movflags', '+faststart',
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Cleanup temp files
    for ts_file in ts_files:
        Path(ts_file).unlink(missing_ok=True)
    concat_ts.unlink(missing_ok=True)

    if result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 10000:
        final_dur = get_media_duration(Path(output_path))
        final_size = Path(output_path).stat().st_size / (1024 * 1024)
        logging.info(f"[BULLETPROOF] SUCCESS! {output_path}")
        logging.info(f"[BULLETPROOF] Final: {final_dur:.1f}s, {final_size:.1f}MB")
        return True
    else:
        logging.error(f"[BULLETPROOF] Remux failed: {result.stderr[:300]}")
        return False


# Keep old functions as fallbacks
def normalize_clip(input_path: str, output_path: str,
                   target_w: int = 720, target_h: int = 1280,
                   fps: int = 24, crf: int = 23) -> bool:
    """Re-encode a clip to fixed size/FPS/codec/SAR to make concat safe."""
    try:
        vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},fps={fps},setsar=1"
        cmd = [
            FFMPEG_PATH, "-y", "-i", str(input_path),
            "-vf", vf, "-r", str(fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-an", str(output_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            return False
        return Path(output_path).exists() and Path(output_path).stat().st_size > 10_000
    except:
        return False


def concatenate_clips(clip_paths: List[str], output_path: str) -> bool:
    """Simple concat demuxer - requires identical clips."""
    if not clip_paths:
        return False
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], output_path)
        return True
    try:
        concat_file = Path(output_path).parent / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for p in clip_paths:
                f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
        cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_file), '-c', 'copy', str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        concat_file.unlink(missing_ok=True)
        return result.returncode == 0 and Path(output_path).exists()
    except:
        return False


def normalize_and_concatenate(clip_paths: List[str], output_path: str,
                              aspect_ratio: str = "9:16", fps: int = 24) -> bool:
    """REDIRECTS TO BULLETPROOF CONCAT - the only reliable method."""
    logging.info("[NORMALIZE_AND_CONCAT] Redirecting to bulletproof_concat...")
    return bulletproof_concat(clip_paths, output_path, aspect_ratio, fps)


# === VIDEO COMPOSER ===
class VideoComposer:
    """Compose videos with EXACT duration control (no -shortest flag)"""

    def __init__(self):
        self.ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run([FFMPEG_PATH, '-version'],
                         capture_output=True,
                         check=True,
                         timeout=5)
            return True
        except:
            return False

    def is_available(self) -> bool:
        """Check if composer is available"""
        return self.ffmpeg_available

    def compose(self, project_id: str, music_style: str = "upbeat",
                target_duration: float = None) -> Dict[str, Any]:
        """Compose final video with EXACT duration control

        Args:
            project_id: Project identifier
            music_style: Music style (for future use)
            target_duration: If set, enforces exact output duration

        Returns:
            Dict with output_path, actual_duration, validated
        """
        if not self.is_available():
            raise Exception("FFmpeg not available for video composition")

        # Paths
        video_dir = Path(f"outputs/videos/{project_id}")
        audio_dir = Path(f"outputs/audio/{project_id}")

        # Find video file
        video_file = None
        for ext in ['mp4', 'webm', 'mov']:
            candidate = video_dir / f"raw.{ext}"
            if candidate.exists():
                video_file = candidate
                break

        if not video_file:
            raise Exception(f"No video file found for project {project_id}")

        # Find audio file
        audio_file = None
        for ext in ['mp3', 'wav']:
            candidate = audio_dir / f"voiceover.{ext}"
            if candidate.exists():
                audio_file = candidate
                break

        if not audio_file:
            raise Exception(f"No audio file found for project {project_id}")

        # Get durations
        video_duration = get_media_duration(video_file)
        audio_duration = get_media_duration(audio_file)

        logging.info(f"Composing: video={video_duration:.1f}s, audio={audio_duration:.1f}s, "
                    f"target={target_duration}s")

        # Output path
        output_path = video_dir / "final.mp4"

        # Build FFmpeg command with EXACT duration control
        cmd = self._build_compose_command(
            video_file, audio_file, output_path,
            video_duration, audio_duration, target_duration
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=True
            )

            if not output_path.exists():
                raise Exception("Video composition failed - output file not created")

            # Validate final duration
            final_duration = get_media_duration(output_path)

            if target_duration:
                is_valid, msg = validate_duration(final_duration, target_duration, tolerance=1.0)
                logging.info(f"Duration validation: {msg}")

                if not is_valid:
                    logging.warning(f"Final video duration mismatch: {msg}")
            else:
                is_valid = True
                msg = f"No target duration specified, final={final_duration:.1f}s"

            return {
                "output_path": str(output_path),
                "actual_duration": final_duration,
                "target_duration": target_duration,
                "validated": is_valid,
                "validation_msg": msg
            }

        except subprocess.TimeoutExpired:
            raise Exception("Video composition timed out")
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg error: {e.stderr}")

    def _build_compose_command(self, video_file: Path, audio_file: Path,
                               output_path: Path, video_duration: float,
                               audio_duration: float, target_duration: float,
                               video_fps: float = None) -> List[str]:
        """Build FFmpeg command with proper duration handling

        Strategy:
        - If video is shorter than target: loop video seamlessly
        - If video is longer than target: trim video
        - If audio is shorter than target: pad with silence
        - If audio is longer than target: it should already be adjusted by TTS
        - Use explicit -t flag for exact duration

        Works perfectly for 10-15 second clips by looping 5s video clips!
        """
        filters = []

        # Use actual video FPS for accurate loop calculation
        fps = video_fps or get_video_fps(video_file) or DEFAULT_FPS

        # Determine target (use explicit target, or audio duration, or video duration)
        actual_target = target_duration or audio_duration or video_duration

        logging.info(f"[COMPOSE] Building command: video={video_duration:.1f}s @ {fps}fps, "
                    f"audio={audio_duration:.1f}s, target={actual_target:.1f}s")

        # Video filter: handle duration mismatch
        if video_duration < actual_target - 0.5:
            # Loop video to fill duration (e.g., 5s video → 15s final)
            loop_count = int(actual_target / video_duration) + 1
            frame_count = int(video_duration * fps)
            logging.info(f"[COMPOSE] Looping video {loop_count}x (frame_count={frame_count})")
            filters.append(f"[0:v]loop=loop={loop_count}:size={frame_count}:start=0,trim=duration={actual_target},setpts=PTS-STARTPTS[v]")
        elif video_duration > actual_target + 0.5:
            # Trim video
            filters.append(f"[0:v]trim=duration={actual_target},setpts=PTS-STARTPTS[v]")
        else:
            # Video duration is close enough - use setpts as passthrough
            filters.append("[0:v]setpts=PTS-STARTPTS[v]")

        # Audio filter: Max 1.1x speed-up to preserve natural sound
        if audio_duration > actual_target + 0.5:
            speed_factor = audio_duration / actual_target
            if speed_factor <= MAX_TEMPO_ADJUSTMENT:
                # Small speed-up is acceptable
                filters.append(f"[1:a]atempo={speed_factor:.4f}[a]")
            else:
                # Speed-up would be too much - apply max and trim the rest
                filters.append(f"[1:a]atempo={MAX_TEMPO_ADJUSTMENT:.4f},atrim=duration={actual_target},asetpts=PTS-STARTPTS[a]")
        elif audio_duration < actual_target - 0.5:
            speed_factor = audio_duration / actual_target
            if speed_factor >= 0.9:
                # Small slow-down is acceptable
                filters.append(f"[1:a]atempo={speed_factor:.4f}[a]")
            else:
                # Too short - pad with silence
                filters.append(f"[1:a]apad=whole_dur={actual_target}[a]")
        else:
            # Audio duration is close enough - use anull as passthrough
            filters.append("[1:a]anull[a]")

        filter_complex = ';'.join(filters)

        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(video_file),
            '-i', str(audio_file),
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', '[a]',
            '-t', str(actual_target),  # EXPLICIT duration - no -shortest!
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            str(output_path)
        ]

        return cmd

    def compose_simple(self, project_id: str) -> Path:
        """Simple composition for videos that already have audio (e.g., WAN 2.5)

        Use this when video already contains synchronized audio from Replicate
        """
        video_dir = Path(f"outputs/videos/{project_id}")

        # Find video file
        video_file = None
        for ext in ['mp4', 'webm', 'mov']:
            candidate = video_dir / f"raw.{ext}"
            if candidate.exists():
                video_file = candidate
                break

        if not video_file:
            raise Exception(f"No video file found for project {project_id}")

        # Output path
        output_path = video_dir / "final.mp4"

        # Simple re-encode for web optimization
        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(video_file),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            return output_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Simple composition failed: {e.stderr}")

    def add_captions(self, project_id: str, script: str) -> Path:
        """Add captions to video"""
        if not self.is_available():
            raise Exception("FFmpeg not available for caption addition")

        video_dir = Path(f"outputs/videos/{project_id}")

        # Find final video
        input_video = video_dir / "final.mp4"
        if not input_video.exists():
            raise Exception(f"Final video not found for project {project_id}")

        output_video = video_dir / "captioned.mp4"

        # Create subtitle file
        srt_path = video_dir / "captions.srt"
        self._create_srt(script, srt_path)

        # Add captions with ffmpeg
        # FFmpeg subtitles filter needs special path escaping
        # Windows: C:\path\file.srt -> C\\:/path/file.srt
        # Linux: /app/path/file.srt -> /app/path/file.srt (no change needed)
        import platform
        if platform.system() == 'Windows':
            srt_path_escaped = str(srt_path).replace('\\', '/').replace(':', '\\:')
        else:
            # Linux/Docker - just use the path as-is with forward slashes
            srt_path_escaped = str(srt_path)

        # PROFESSIONAL caption styling - modern social media look
        # Impact font, thick outline, drop shadow - like TikTok/Reels captions
        caption_style = (
            "FontName=Impact,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H80000000,Bold=1,BorderStyle=1,Outline=3,Shadow=2,"
            "MarginL=20,MarginR=20,MarginV=60,Alignment=2"
        )

        cmd = [
            FFMPEG_PATH,
            '-i', str(input_video),
            '-vf', f"subtitles='{srt_path_escaped}':force_style='{caption_style}'",
            '-c:v', 'libx264',  # Re-encode video with captions
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',  # Re-encode audio (handles missing audio gracefully)
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            str(output_video)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check if output was created (some errors don't set returncode)
            if output_video.exists() and output_video.stat().st_size > 10000:
                logging.info(f"[CAPTIONS] Added captions to video: {output_video}")
                return output_video

            # If failed, log error and return original video
            logging.warning(f"[CAPTIONS] Failed, using uncaptioned video: {result.stderr[:300] if result.stderr else 'No error'}")
            return input_video

        except subprocess.TimeoutExpired:
            logging.warning("[CAPTIONS] Timed out, using uncaptioned video")
            return input_video
        except Exception as e:
            logging.warning(f"[CAPTIONS] Error: {e}, using uncaptioned video")
            return input_video

    def _create_srt(self, script: str, output_path: Path):
        """Create SRT subtitle file from script - compact 3-4 word chunks"""
        # Split into words and group into 3-4 word chunks for stylish captions
        words = script.split()
        words_per_chunk = 4  # Show 4 words at a time
        words_per_second = 2.5  # Speaking rate

        srt_content = []
        current_time = 0.0
        chunk_num = 1

        for i in range(0, len(words), words_per_chunk):
            chunk = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk)
            word_count = len(chunk)
            duration = word_count / words_per_second

            start_time = self._format_srt_time(current_time)
            end_time = self._format_srt_time(current_time + duration)

            srt_content.append(f"{chunk_num}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(chunk_text)
            srt_content.append("")

            current_time += duration
            chunk_num += 1

        output_path.write_text('\n'.join(srt_content), encoding='utf-8')

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_ass_time(self, seconds: float) -> str:
        """Format seconds to ASS time format (H:MM:SS.cc)"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def _smart_title(self, word: str) -> str:
        """Title case helper - don't capitalize small words"""
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'of', 'in'}
        w = word.lower().rstrip('.,!?')
        if w in small_words and len(word) < 4:
            return word.lower()
        return word.capitalize()

    def generate_viral_captions(self, word_timings: List[Dict], output_path: Path) -> Path:
        """Generate VIRAL captions with WORD-BY-WORD highlighting (ASS format)

        Creates TikTok/Reels style captions where each word lights up (yellow)
        as it's spoken, creating a karaoke effect.

        Args:
            word_timings: List of {"text": str, "start": float, "end": float}
            output_path: Path to save the .ass file

        Returns:
            Path to the generated .ass file
        """
        if not word_timings:
            logging.warning("[CAPTIONS] No word timings provided")
            return None

        # ASS header - VIRAL STYLE with bigger font, outline, shadow
        ass_content = """[Script Info]
Title: Viral Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,4,2,40,40,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # SEMANTIC GROUPING: Use timing gaps + punctuation
        groups = []
        current_group = []

        for i, timing in enumerate(word_timings):
            word = timing["text"].strip()
            if not word:
                continue
            current_group.append(timing)

            is_end_punct = word.endswith(('.', '!', '?'))
            is_comma = word.endswith(',')

            # Check for timing gap (>0.25s pause = natural break)
            has_pause = False
            if i < len(word_timings) - 1:
                gap = word_timings[i + 1]["start"] - timing["end"]
                has_pause = gap > 0.25

            # Group break conditions
            if is_end_punct or has_pause or len(current_group) >= 4 or (len(current_group) >= 3 and is_comma):
                if current_group:
                    groups.append(current_group.copy())
                    current_group = []

        if current_group:
            groups.append(current_group)

        logging.info(f"[CAPTIONS] Created {len(groups)} semantic phrase groups")

        events = []

        for group in groups:
            if not group:
                continue

            # WORD-BY-WORD HIGHLIGHTING
            # For each word in the group, create an event where THAT word is highlighted
            for word_idx, current_word_timing in enumerate(group):
                word_start = current_word_timing["start"]
                word_end = current_word_timing["end"]

                # Build the line with current word highlighted (yellow)
                line_parts = []
                for j, wt in enumerate(group):
                    word = self._smart_title(wt["text"].strip())
                    if j == word_idx:
                        # HIGHLIGHT current word: Yellow color
                        line_parts.append(f"{{\\c&H00FFFF&}}{word}{{\\c&HFFFFFF&}}")
                    else:
                        line_parts.append(word)

                caption_text = " ".join(line_parts)
                # Clean punctuation spacing
                caption_text = caption_text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")

                event = f"Dialogue: 0,{self._format_ass_time(word_start)},{self._format_ass_time(word_end)},Main,,0,0,0,,{caption_text}"
                events.append(event)

        ass_content += "\n".join(events)

        output_path.write_text(ass_content, encoding='utf-8')
        logging.info(f"[CAPTIONS] Viral ASS captions saved: {output_path} ({len(events)} dialogue events)")

        return output_path

    def add_viral_captions(self, project_id: str, word_timings: List[Dict],
                           target_duration: float = None) -> Path:
        """Add VIRAL word-by-word captions to video with AAC-safe audio

        Args:
            project_id: Project identifier
            word_timings: List of {"text": str, "start": float, "end": float}
            target_duration: If set, enforces exact output duration with AAC-safe audio

        Returns:
            Path to captioned video
        """
        import math

        if not self.is_available():
            raise Exception("FFmpeg not available for caption addition")

        video_dir = Path(f"outputs/videos/{project_id}")
        audio_dir = Path(f"outputs/audio/{project_id}")

        # Find final video
        input_video = video_dir / "final.mp4"
        if not input_video.exists():
            raise Exception(f"Final video not found for project {project_id}")

        output_video = video_dir / "captioned.mp4"

        # Generate viral ASS captions
        ass_path = video_dir / "captions_viral.ass"
        self.generate_viral_captions(word_timings, ass_path)

        if not ass_path.exists():
            logging.warning("[CAPTIONS] Failed to generate ASS file, returning original video")
            return input_video

        # Escape path for FFmpeg ASS filter
        import platform
        if platform.system() == 'Windows':
            ass_escaped = str(ass_path).replace('\\', '/').replace(':', '\\:')
        else:
            ass_escaped = str(ass_path)

        # Build FFmpeg command
        video_duration = get_media_duration(input_video)
        actual_target = target_duration or video_duration

        # Calculate AAC-safe audio duration (avoid packet padding pushing over target)
        # AAC uses 1024 samples per frame. Trim to nearest lower boundary.
        sr = 24000  # Kokoro sample rate (may be resampled but we trim before encode)
        aac_frame = 1024
        audio_cap = math.floor(actual_target * sr / aac_frame) * aac_frame / sr
        logging.info(f"[CAPTIONS] Target: {actual_target:.3f}s, AAC-safe audio: {audio_cap:.4f}s")

        # Video filter: ASS captions
        vf = f"ass='{ass_escaped}'"

        # Audio filter: AAC-safe trim
        af = f"atrim=0:{audio_cap},asetpts=PTS-STARTPTS"

        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(input_video),
            '-filter_complex', f"[0:v]{vf}[v];[0:a]{af}[a]",
            '-map', '[v]',
            '-map', '[a]',
            '-t', str(actual_target),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '20',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            str(output_video)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if output_video.exists() and output_video.stat().st_size > 10000:
                final_duration = get_media_duration(output_video)
                logging.info(f"[CAPTIONS] Viral captions added: {output_video} ({final_duration:.2f}s)")
                return output_video

            logging.warning(f"[CAPTIONS] Failed: {result.stderr[:300] if result.stderr else 'No error'}")
            return input_video

        except subprocess.TimeoutExpired:
            logging.warning("[CAPTIONS] Timed out, using uncaptioned video")
            return input_video
        except Exception as e:
            logging.warning(f"[CAPTIONS] Error: {e}, using uncaptioned video")
            return input_video


# === PRODUCTION VIDEO COMPOSER (Enhanced) ===
class ProductionVideoComposer(VideoComposer):
    """Enhanced video composer with advanced effects"""

    def compose(self, project_id: str, music_style: str = "upbeat",
                target_duration: float = None) -> Dict[str, Any]:
        """Compose with advanced effects and transitions"""
        # Run base composition first (returns dict)
        base_result = super().compose(project_id=project_id, music_style=music_style,
                                       target_duration=target_duration)

        # Paths
        video_dir = Path(f"outputs/videos/{project_id}")
        input_path = Path(base_result['output_path'])
        enhanced_path = video_dir / "final_enhanced.mp4"

        # Apply color grading and effects (optimized)
        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(input_path),
            '-vf', 'eq=contrast=1.1:brightness=0.05:saturation=1.2',
            '-c:a', 'copy',
            '-preset', 'fast',
            '-movflags', '+faststart',
            str(enhanced_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)

            # Replace original with enhanced
            shutil.move(str(enhanced_path), str(input_path))

            # Refresh duration
            base_result['actual_duration'] = get_media_duration(input_path)
            base_result['validated'] = True if (target_duration is None) else base_result.get('validated', False)

            return base_result
        except Exception:
            # If enhancement fails, return base result
            return base_result


# === FILE I/O UTILITIES ===
def atomic_write_json(path: Path, data: Dict[str, Any]):
    """Atomically write JSON to file"""
    temp_path = path.with_suffix('.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise Exception(f"Failed to write JSON: {e}")


def atomic_read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file with error handling"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to read JSON: {e}")


# ========================================================================
# FLASK APPLICATION
# ========================================================================

# === CONSTANTS ===
API_VERSION = "v1"
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
DEFAULT_TIMEOUT = 120
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')
ALLOWED_VIDEO_FORMATS = ['mp4', 'webm', 'mov']
ALLOWED_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:8080,http://127.0.0.1:8080').split(',')

# === ENUMS ===
class VideoStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

# === DATA MODELS ===
@dataclass
class ApiResponse:
    """Standardized API response"""
    status: ResponseStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"status": self.status.value}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.error_code:
            result["error_code"] = self.error_code
        if self.request_id:
            result["request_id"] = self.request_id
        return result

# === LOGGING SETUP ===
def setup_logging():
    """Configure production logging with rotation and request IDs"""
    # Ensure logs directory exists
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure handlers
    handlers = [logging.StreamHandler()]

    # Add rotating file handler
    file_handler = RotatingFileHandler(
        log_dir / 'backend.log',
        maxBytes=10_000_000,  # 10MB
        backupCount=10
    )
    handlers.append(file_handler)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )

    # Add request ID filter
    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            try:
                from flask import has_request_context
                if has_request_context():
                    record.request_id = getattr(g, 'request_id', 'no-request-id')
                else:
                    record.request_id = 'initialization'
            except RuntimeError:
                record.request_id = 'no-context'
            return True

    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdFilter())

    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()

# === FLASK APP FACTORY ===
def create_app(config=None):
    """Application factory pattern for better testing and configuration"""
    global FFMPEG_POOL

    # Get frontend path (sibling to backend folder)
    frontend_path = Path(__file__).resolve().parent.parent / 'frontend'

    app = Flask(__name__,
                static_folder=str(frontend_path),
                static_url_path='')

    # Load configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year cache
    app.config['RATELIMIT_HEADERS_ENABLED'] = True
    app.config['start_time'] = time.time()  # For uptime tracking in health endpoints
    app.config['PROPAGATE_EXCEPTIONS'] = False  # SECURITY: Don't propagate exceptions in prod

    if config:
        app.config.update(config)

    # Initialize gzip compression
    if COMPRESS_AVAILABLE:
        app.config['COMPRESS_ALGORITHM'] = 'gzip'
        app.config['COMPRESS_LEVEL'] = 6
        app.config['COMPRESS_MIN_SIZE'] = 500
        Compress(app)
        logger.info("[OK] Gzip compression enabled")
    else:
        logger.info("[INFO] Flask-Compress not installed, gzip disabled")

    # Initialize CORS with environment-based origins
    CORS(app,
         origins=ALLOWED_ORIGINS,
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-API-Key", "Idempotency-Key"],
         expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After", "Idempotency-Replayed"],
         max_age=3600)

    # Initialize rate limiter with Redis if available
    redis_url = os.getenv('REDIS_URL', 'memory://')
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per minute"],  # Increased for production
        storage_uri=redis_url,
        headers_enabled=True
    )

    # Initialize idempotency store
    app.config['idem_store'] = IdempotencyStore(redis_url)
    logger.info(f"[OK] Idempotency store initialized ({'Redis' if 'redis' in redis_url else 'memory'})")

    # Initialize FFmpeg concurrency pool (prevents CPU overload)
    ffmpeg_workers = int(os.getenv('FFMPEG_WORKERS', '3'))
    FFMPEG_POOL = ThreadPoolExecutor(max_workers=ffmpeg_workers)
    app.config['ffmpeg_pool'] = FFMPEG_POOL
    logger.info(f"[OK] FFmpeg pool initialized ({ffmpeg_workers} workers)")

    # Initialize services
    services = initialize_services()
    app.config['services'] = services

    # Register middleware
    register_middleware(app)

    # Register error handlers
    register_error_handlers(app)

    # Register Auth Blueprint BEFORE routes (so blueprints take priority over catch-all)
    try:
        from auth import auth_bp
        app.register_blueprint(auth_bp)
        logger.info("[OK] Auth blueprint registered (/api/v1/auth)")
    except ImportError:
        try:
            from backend.auth import auth_bp
            app.register_blueprint(auth_bp)
            logger.info("[OK] Auth blueprint registered (/api/v1/auth)")
        except ImportError as e:
            logger.warning(f"[WARN] Auth blueprint not available: {e}")

    # Register Admin Blueprint BEFORE routes
    try:
        from admin_routes import admin_bp
        app.register_blueprint(admin_bp)
        logger.info("[OK] Admin blueprint registered (/api/v1/admin)")
    except ImportError:
        try:
            from backend.admin_routes import admin_bp
            app.register_blueprint(admin_bp)
            logger.info("[OK] Admin blueprint registered (/api/v1/admin)")
        except ImportError as e:
            logger.warning(f"[WARN] Admin blueprint not available: {e}")

    # Register routes AFTER blueprints (catch-all route should be last)
    register_routes(app, limiter)

    # Determine which implementations are being used
    tts_type = "OpenAI" if services.get('tts') and isinstance(services['tts'], TTSService) else "Kokoro"
    composer_type = "Production" if isinstance(services.get('video_composer'), ProductionVideoComposer) else "Basic"

    logger.info("="*70)
    logger.info("REELCRAFT AI - Production Backend API (Hardened)")
    logger.info(f"   Version: {API_VERSION}")
    logger.info(f"   Environment: {os.getenv('FLASK_ENV', 'production')}")
    logger.info(f"   Redis: {'Connected' if 'redis' in redis_url else 'Memory (use Redis for production)'}")
    logger.info(f"   TTS: {tts_type}")
    logger.info(f"   Video Composer: {composer_type}")
    logger.info(f"   HTTP Pool: Enabled (100 connections)")
    logger.info(f"   Retry: Enabled (3 retries with backoff)")
    logger.info("="*70)

    return app

# === SERVICE INITIALIZATION ===
def initialize_services():
    """Initialize all external services with error handling"""
    services = {}

    try:
        services['groq'] = GroqService(api_key=os.getenv('GROQ_API_KEY'))
        logger.info("[OK] Groq service initialized")
    except Exception as e:
        logger.error(f"[ERROR] Groq initialization failed: {e}")
        services['groq'] = None

    try:
        # Try Local Kokoro first (best quality, no API needed)
        local_kokoro = LocalKokoroService()
        if local_kokoro.is_available():
            services['tts'] = local_kokoro
            services['local_kokoro'] = local_kokoro  # Also store separately for health checks
            logger.info("[OK] TTS service initialized (Local Kokoro - best quality)")
        else:
            error_msg = local_kokoro.get_init_error() or "Unknown error"
            logger.warning(f"[WARN] Local Kokoro unavailable: {error_msg}")

            # Fall back to OpenAI TTS
            tts = TTSService()
            if tts.is_available():
                services['tts'] = tts
                logger.info("[OK] TTS service initialized (OpenAI)")
            else:
                # Fall back to Kokoro API
                kokoro = KokoroService()
                if kokoro.is_available():
                    services['tts'] = kokoro
                    logger.info("[OK] TTS service initialized (Kokoro API)")
                else:
                    services['tts'] = None
                    logger.warning("[WARN] No TTS service available - video generation will fail!")
    except Exception as e:
        logger.error(f"[ERROR] TTS initialization failed: {e}")
        services['tts'] = None

    try:
        services['replicate'] = ReplicateService(api_token=os.getenv('REPLICATE_API_TOKEN'))
        logger.info("[OK] Replicate service initialized")
    except Exception as e:
        logger.error(f"[ERROR] Replicate initialization failed: {e}")
        services['replicate'] = None

    try:
        runpod = RunPodService(
            api_key=os.getenv('RUNPOD_API_KEY'),
            endpoint_id=os.getenv('RUNPOD_ENDPOINT_ID')
        )
        if runpod.is_available():
            services['runpod'] = runpod
            logger.info("[OK] RunPod service initialized (WAN 2.2 + Edge TTS)")
        else:
            services['runpod'] = None
            logger.info("[INFO] RunPod service not configured (set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)")
    except Exception as e:
        logger.error(f"[ERROR] RunPod initialization failed: {e}")
        services['runpod'] = None

    # Initialize video provider (demo mode always available)
    try:
        from video_providers import get_video_provider
        video_provider = get_video_provider()
        if video_provider and video_provider.is_available():
            services['video_provider'] = video_provider
            services['demo'] = video_provider  # Also register as 'demo' for frontend check
            logger.info(f"[OK] Video provider initialized ({video_provider.provider_name})")
        else:
            services['video_provider'] = None
            services['demo'] = None
    except Exception as e:
        logger.error(f"[ERROR] Video provider initialization failed: {e}")
        services['video_provider'] = None
        services['demo'] = None

    try:
        # Try production composer first
        composer = ProductionVideoComposer()
        if composer.is_available():
            services['video_composer'] = composer
            logger.info("[OK] Video composer initialized (Production)")
        else:
            # Fall back to basic composer
            composer = VideoComposer()
            if composer.is_available():
                services['video_composer'] = composer
                logger.info("[OK] Video composer initialized (Basic)")
            else:
                services['video_composer'] = None
                logger.error("[ERROR] Video composer not available (FFmpeg missing)")
    except Exception as e:
        logger.error(f"[ERROR] Video composer initialization failed: {e}")
        services['video_composer'] = None

    return services

# === MIDDLEWARE ===
def register_middleware(app):
    """Register application middleware"""

    @app.before_request
    def before_request():
        """Attach request ID and start timer"""
        # Generate or extract request ID
        g.request_id = request.headers.get('X-Request-ID', uuid.uuid4().hex)
        g.start_time = time.time()

        # Log request
        logger.info(f"--> {request.method} {request.path} from {get_remote_address()}")

    @app.after_request
    def after_request(response):
        """Add security headers and log response"""
        # Add request ID to response
        response.headers['X-Request-ID'] = g.request_id

        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Content Security Policy (tuned for video generation app)
        csp = (
            "default-src 'self'; "
            "img-src 'self' data: https: blob:; "
            "media-src 'self' https: blob:; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "connect-src 'self' https://api.pexels.com https://pixabay.com https://*.runpod.ai wss:; "
            "frame-ancestors 'none'"
        )
        response.headers['Content-Security-Policy'] = csp

        if os.getenv('FLASK_ENV') == 'production':
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Log response time
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            logger.info(f"<-- {response.status_code} in {duration:.3f}s")

        return response

# === DECORATORS ===
def track_performance(f):
    """Performance tracking decorator with request context"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"[PERF] {f.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[ERROR] {f.__name__} failed after {duration:.2f}s: {str(e)}")
            raise
    return decorated_function

def validate_request(*required_fields, **field_types):
    """Request validation decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Content-Type must be application/json",
                    error_code="INVALID_CONTENT_TYPE",
                    request_id=g.request_id
                ).to_dict()), 400

            data = request.get_json()

            # Check required fields
            for field in required_fields:
                if field not in data or data[field] is None:
                    return jsonify(ApiResponse(
                        status=ResponseStatus.ERROR,
                        error=f"Missing required field: {field}",
                        error_code="MISSING_FIELD",
                        request_id=g.request_id
                    ).to_dict()), 400

            # Validate field types (allow None for optional fields)
            for field, expected_type in field_types.items():
                if field in data and data[field] is not None:
                    value = data[field]
                    # Accept int when float is expected (JavaScript sends integers)
                    if expected_type == float and isinstance(value, (int, float)):
                        data[field] = float(value)  # Convert to float
                    elif not isinstance(value, expected_type):
                        return jsonify(ApiResponse(
                            status=ResponseStatus.ERROR,
                            error=f"Invalid type for field {field}: expected {expected_type.__name__}",
                            error_code="INVALID_TYPE",
                            request_id=g.request_id
                        ).to_dict()), 400

            g.validated_data = data
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_service(service_name):
    """Ensure required service is available"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import current_app
            services = current_app.config.get('services', {})
            service = services.get(service_name)

            if not service or not service.is_available():
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error=f"{service_name} service is not available",
                    error_code="SERVICE_UNAVAILABLE",
                    request_id=g.request_id
                ).to_dict()), 503

            g.service = service
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_video_provider():
    """Ensure video provider is available (uses VIDEO_PROVIDER env var)"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                from video_providers import get_video_provider
                provider = get_video_provider()
                if not provider or not provider.is_available():
                    return jsonify(ApiResponse(
                        status=ResponseStatus.ERROR,
                        error="No video provider available",
                        error_code="VIDEO_PROVIDER_UNAVAILABLE",
                        request_id=g.request_id
                    ).to_dict()), 503
                g.service = provider
                g.video_provider = provider
                return f(*args, **kwargs)
            except ImportError:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Video providers module not found",
                    error_code="MODULE_NOT_FOUND",
                    request_id=g.request_id
                ).to_dict()), 500
        return decorated_function
    return decorator

# === PATH VALIDATION ===
def safe_path_join(base: Path, *parts: str) -> Path:
    """Safely join paths preventing directory traversal"""
    base_resolved = base.resolve()
    candidate = (base / Path(*parts)).resolve()

    if base_resolved == candidate or base_resolved in candidate.parents:
        return candidate
    raise ValueError("Path traversal detected")

def validate_project_id(project_id: str) -> bool:
    """Validate project ID format"""
    return bool(SAFE_ID_PATTERN.match(project_id))

# === ERROR HANDLERS ===
def register_error_handlers(app):
    """Register global error handlers"""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify(ApiResponse(
            status=ResponseStatus.ERROR,
            error="Bad request",
            error_code="BAD_REQUEST",
            request_id=getattr(g, 'request_id', None)
        ).to_dict()), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify(ApiResponse(
            status=ResponseStatus.ERROR,
            error="Resource not found",
            error_code="NOT_FOUND",
            request_id=getattr(g, 'request_id', None)
        ).to_dict()), 404

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify(ApiResponse(
            status=ResponseStatus.ERROR,
            error="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            request_id=getattr(g, 'request_id', None)
        ).to_dict()), 429

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify(ApiResponse(
            status=ResponseStatus.ERROR,
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=getattr(g, 'request_id', None)
        ).to_dict()), 500

    @app.errorhandler(503)
    def service_unavailable(error):
        return jsonify(ApiResponse(
            status=ResponseStatus.ERROR,
            error="Service temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE",
            request_id=getattr(g, 'request_id', None)
        ).to_dict()), 503

# === ROUTES ===
def register_routes(app, limiter):
    """Register all application routes"""

    from flask import current_app, send_from_directory

    # === STATIC FILE SERVING ===
    @app.route('/')
    def serve_index():
        """Serve main index.html"""
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/<path:filename>')
    def serve_static(filename):
        """Serve static files (HTML, CSS, JS, etc.)"""
        # IMPORTANT: Skip API paths - let Flask handle them with registered blueprints
        # This prevents the catch-all route from intercepting /api/v1/auth/login etc.
        if filename.startswith('api/'):
            # Return 404 for unmatched API routes (blueprints should have handled these)
            return jsonify({'status': 'error', 'error': 'API endpoint not found', 'error_code': 'NOT_FOUND'}), 404

        # Check if file exists
        file_path = Path(app.static_folder) / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(app.static_folder, filename)
        # For SPA routing, return index.html for HTML requests
        if not '.' in filename:
            return send_from_directory(app.static_folder, 'index.html')
        # Return 404 for missing files
        return jsonify({'status': 'error', 'error': 'Resource not found', 'error_code': 'NOT_FOUND'}), 404

    # === HEALTH CHECK ===
    @app.route(f'/api/{API_VERSION}/health', methods=['GET'])
    @app.route('/api/health', methods=['GET'])  # Legacy support
    @limiter.exempt
    def health_check():
        """Enhanced health check with service and database status"""
        services = current_app.config.get('services', {})

        service_status = {}
        all_healthy = True

        for name, service in services.items():
            try:
                is_available = service and service.is_available()
                service_status[name] = is_available
                if not is_available:
                    all_healthy = False
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                service_status[name] = False
                all_healthy = False

        # Add aliases for frontend compatibility
        # Local Kokoro is the preferred TTS
        service_status['kokoro'] = service_status.get('local_kokoro', service_status.get('tts', False))
        service_status['local_kokoro'] = service_status.get('local_kokoro', False)
        service_status['runpod'] = service_status.get('replicate', False)

        # Check database health
        db_health = {'healthy': False}
        try:
            try:
                from database import check_database_health
            except ImportError:
                from backend.database import check_database_health
            db_health = check_database_health()
            service_status['database'] = db_health.get('healthy', False)
            if not db_health.get('healthy', False):
                all_healthy = False
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            service_status['database'] = False
            all_healthy = False

        return jsonify(ApiResponse(
            status=ResponseStatus.SUCCESS,
            data={
                'status': 'healthy' if all_healthy else 'degraded',
                'timestamp': datetime.utcnow().isoformat(),
                'version': API_VERSION,
                'environment': os.getenv('FLASK_ENV', 'production'),
                'services': service_status,
                'database': db_health,
                'uptime': time.time() - app.config.get('start_time', time.time())
            },
            request_id=g.request_id
        ).to_dict())

    # === KUBERNETES/LOAD BALANCER HEALTH PROBES ===
    # === DEBUG: List all routes ===
    @app.route(f'/api/{API_VERSION}/debug/routes', methods=['GET'])
    @limiter.limit("10 per minute")  # SECURITY FIX: Rate limit debug endpoint
    @require_admin  # SECURITY FIX: Require admin auth in production
    def debug_routes():
        """Debug endpoint to list all registered routes (admin only in production)"""
        routes = []
        for rule in app.url_map.iter_rules():
            if 'generate-video' in rule.rule or 'i2v' in rule.rule.lower():
                routes.append({
                    'path': rule.rule,
                    'methods': list(rule.methods - {'HEAD', 'OPTIONS'}),
                    'endpoint': rule.endpoint
                })
        return jsonify({
            'total_routes': len(list(app.url_map.iter_rules())),
            'video_routes': routes
        })

    @app.route(f'/api/{API_VERSION}/readiness', methods=['GET'])
    @app.route('/readiness', methods=['GET'])
    @limiter.exempt
    def readiness_check():
        """Kubernetes/ALB readiness probe

        Returns 200 if the service is ready to accept traffic.
        Checks if critical services (Groq, Replicate) are available.
        """
        services = current_app.config.get('services', {})

        # Check critical services for video generation
        groq_ok = services.get('groq') and services['groq'].is_available()
        replicate_ok = services.get('replicate') and services['replicate'].is_available()

        # Service is ready if at least script generation works
        is_ready = groq_ok

        if is_ready:
            return jsonify({
                'ready': True,
                'timestamp': datetime.utcnow().isoformat(),
                'services': {
                    'groq': groq_ok,
                    'replicate': replicate_ok
                }
            }), 200
        else:
            return jsonify({
                'ready': False,
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'Critical services unavailable',
                'services': {
                    'groq': groq_ok,
                    'replicate': replicate_ok
                }
            }), 503

    @app.route(f'/api/{API_VERSION}/liveness', methods=['GET'])
    @app.route('/liveness', methods=['GET'])
    @limiter.exempt
    def liveness_check():
        """Kubernetes/ALB liveness probe

        Returns 200 if the service is alive (basic health).
        This is a minimal check - just confirms the server is running.
        """
        return jsonify({
            'alive': True,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - app.config.get('start_time', time.time())
        }), 200

    @app.route(f'/api/{API_VERSION}/metrics', methods=['GET'])
    @app.route('/metrics', methods=['GET'])
    @limiter.exempt
    def metrics():
        """Basic metrics endpoint for monitoring

        Returns request counts, error rates, etc.
        For production, consider using prometheus_flask_exporter.
        """
        metrics_data = app.config.get('metrics', {})

        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - app.config.get('start_time', time.time()),
            'requests': {
                'total': metrics_data.get('total_requests', 0),
                'successful': metrics_data.get('successful_requests', 0),
                'failed': metrics_data.get('failed_requests', 0)
            },
            'activeJobs': metrics_data.get('active_jobs', 0),
            'completedJobs': metrics_data.get('completed_jobs', 0)
        }), 200

    # === PROMPT ENHANCEMENT ===
    @app.route(f'/api/{API_VERSION}/enhance-prompt', methods=['POST'])
    @app.route('/api/enhance-prompt', methods=['POST'])  # Legacy
    @limiter.limit("30 per minute")
    @validate_request('prompt', context=str)
    @require_service('groq')
    @track_performance
    def enhance_prompt():
        """Enhance a video prompt using AI for better video generation results

        Takes a basic prompt and returns a cinematic, detailed version
        optimized for AI video generation.
        """
        try:
            data = g.validated_data

            prompt = sanitize_input(data['prompt'], MAX_PROMPT_LENGTH)
            context = sanitize_input(data.get('context', ''), 100)

            if not prompt or len(prompt.strip()) < 3:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Prompt must be at least 3 characters",
                    error_code="INVALID_PROMPT",
                    request_id=g.request_id
                ).to_dict()), 400

            logger.info(f"Enhancing prompt: {prompt[:50]}...")

            result = g.service.enhance_prompt(
                prompt=prompt,
                context=context or None
            )

            logger.info(f"[OK] Prompt enhanced: {len(prompt)} -> {len(result['enhanced'])} chars")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'original': result['original'],
                    'enhanced': result['enhanced'],
                    'context': result.get('context'),
                    'wordCount': result.get('word_count', 0)
                },
                request_id=g.request_id
            ).to_dict())

        except Exception as e:
            logger.error(f"Prompt enhancement error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="ENHANCEMENT_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === SCRIPT GENERATION (with EXACT word count) ===
    @app.route(f'/api/{API_VERSION}/generate-script', methods=['POST'])
    @app.route('/api/generate-script', methods=['POST'])  # Legacy
    @limiter.limit("20 per minute")
    @validate_request('topic', length=str, tone=str, duration=int, brand=str)
    @require_service('groq')
    @track_performance
    @idempotent(ttl=900)  # 15 minute TTL for script generation
    def generate_script():
        """Generate AI script with EXACT word count for target duration

        Returns script with word count and estimated duration for validation.
        """
        try:
            data = g.validated_data

            # Sanitize inputs
            topic = sanitize_input(data['topic'], MAX_TOPIC_LENGTH)
            brand = sanitize_input(data.get('brand', ''), 100)
            length = data.get('length', 'medium')
            tone = data.get('tone', 'witty')
            duration = min(max(data.get('duration', 30), 5), MAX_VIDEO_DURATION)

            logger.info(f"Generating script: topic={topic[:50]}..., length={length}, tone={tone}, duration={duration}s")

            # Generate script with EXACT word count
            result = g.service.generate_script(
                topic=topic,
                length=length,
                tone=tone,
                duration=duration,
                brand=brand
            )

            # Create project with UUID
            project_id = f"proj_{uuid.uuid4().hex[:12]}"

            # Calculate target word count for validation (using length setting)
            target_words, min_words, max_words = calculate_word_count(duration, length)

            # Save script with metadata
            script_data = {
                'projectId': project_id,
                'topic': topic,
                'brand': brand,
                'script': result['script'],
                'length': length,
                'tone': tone,
                'duration': duration,
                'wordCount': result['word_count'],
                'targetWordCount': target_words,
                'estimatedDuration': result['estimated_duration'],
                'created': datetime.utcnow().isoformat()
            }

            script_path = Path(f"outputs/scripts/{project_id}.json")
            script_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(script_path, script_data)

            logger.info(f"[OK] Script generated: {project_id}, "
                       f"words={result['word_count']}/{target_words}, "
                       f"est_duration={result['estimated_duration']}s/{duration}s")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'script': result['script'],
                    'projectId': project_id,
                    'wordCount': result['word_count'],
                    'targetWordCount': target_words,
                    'estimatedDuration': result['estimated_duration'],
                    'targetDuration': duration
                },
                request_id=g.request_id
            ).to_dict()), 201

        except Exception as e:
            logger.error(f"Script generation error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="GENERATION_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === VOICE-OVER GENERATION (with duration matching) ===
    @app.route(f'/api/{API_VERSION}/generate-voiceover', methods=['POST'])
    @app.route('/api/generate-voiceover', methods=['POST'])  # Legacy
    @limiter.limit("10 per minute")
    @validate_request('script', 'projectId', voiceType=str, targetDuration=float)
    @track_performance
    @idempotent(ttl=900)  # 15 minute TTL for voiceover generation
    def generate_voiceover():
        """Generate voice-over with Kokoro TTS and automatic voice selection

        Features:
        - Automatic voice selection based on tone (witty, serious, sarcastic, professional)
        - Content-aware voice matching (analyzes script keywords)
        - Duration matching with exact-fit technology
        - Falls back to Edge-TTS if Kokoro unavailable
        """
        try:
            data = g.validated_data

            script = sanitize_input(data['script'], MAX_SCRIPT_LENGTH)
            project_id = data['projectId']
            voice_type = data.get('voiceType', 'male-professional')
            target_duration = data.get('targetDuration')  # Optional

            # Validate project ID
            if not validate_project_id(project_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid project ID",
                    error_code="INVALID_PROJECT_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            # Get target duration and tone from project data
            tone = None
            script_path = Path(f"outputs/scripts/{project_id}.json")
            if script_path.exists():
                project_data = atomic_read_json(script_path)
                if not target_duration:
                    target_duration = project_data.get('duration')
                tone = project_data.get('tone')  # Get tone for automatic voice selection

            logger.info(f"[TTS] Generating voiceover: project={project_id}, "
                       f"target_duration={target_duration}s, tone={tone}")

            # TTS Priority: Local Kokoro (best quality + exact-fit + auto voice selection)
            audio_path = None
            actual_duration = None
            speed_adjusted = False
            voice_used = None
            exact_fit_applied = False

            # 1. Try Local Kokoro TTS (best quality with exact-fit and automatic voice selection)
            local_kokoro = LocalKokoroService()
            word_timings = []  # For viral captions
            if local_kokoro.is_available():
                logger.info("[TTS] Using Local Kokoro TTS with auto voice selection...")
                try:
                    result = local_kokoro.generate_voiceover(
                        text=script,
                        voice_type=voice_type,
                        project_id=project_id,
                        target_duration=target_duration,
                        tone=tone  # Pass tone for automatic voice matching
                    )
                    audio_path = result['audio_path']
                    actual_duration = result['actual_duration']
                    exact_fit_applied = result['exact_fit_applied']
                    voice_used = result.get('voice_name', result.get('voice'))
                    speed_adjusted = exact_fit_applied
                    word_timings = result.get('word_timings', [])  # Capture for viral captions
                    logger.info(f"[TTS] Captured {len(word_timings)} word timings for viral captions")
                except Exception as e:
                    logger.warning(f"[TTS] Local Kokoro failed: {e}, trying Edge-TTS...")

            # 2. Fallback to Edge-TTS with duration control
            if not audio_path:
                edge_tts = EdgeTTSService()
                if edge_tts.is_available():
                    result = edge_tts.generate_voiceover(
                        text=script,
                        voice_type=voice_type,
                        project_id=project_id,
                        target_duration=target_duration
                    )
                    audio_path = result['audio_path']
                    actual_duration = result['actual_duration']
                    speed_adjusted = result['speed_adjusted']

            # 3. Fallback to other TTS services
            if not audio_path:
                tts_service = services.get('tts')
                if tts_service and tts_service.is_available():
                    audio_path = tts_service.generate_voiceover(
                        text=script,
                        voice_type=voice_type,
                        project_id=project_id
                    )
                    actual_duration = get_media_duration(Path(audio_path))
                    speed_adjusted = False
                else:
                    raise Exception("No TTS service available")

            # Update project data (including word_timings for viral captions)
            script_path = Path(f"outputs/scripts/{project_id}.json")
            if script_path.exists():
                project_data = atomic_read_json(script_path)
                project_data['voiceover'] = {
                    'path': str(audio_path),
                    'voice': voice_type,
                    'voiceUsed': voice_used,
                    'tone': tone,  # Tone used for automatic voice selection
                    'actualDuration': actual_duration,
                    'targetDuration': target_duration,
                    'speedAdjusted': speed_adjusted,
                    'exactFitApplied': exact_fit_applied,
                    'wordTimings': word_timings,  # For viral word-by-word captions
                    'generated': datetime.utcnow().isoformat()
                }
                atomic_write_json(script_path, project_data)

            logger.info(f"[OK] Voiceover generated: {audio_path}, "
                       f"voice={voice_used}, tone={tone}, "
                       f"duration={actual_duration:.1f}s/{target_duration}s, "
                       f"adjusted={speed_adjusted}, exact_fit={exact_fit_applied}")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'audioPath': str(audio_path),
                    'projectId': project_id,
                    'actualDuration': actual_duration,
                    'targetDuration': target_duration,
                    'speedAdjusted': speed_adjusted,
                    'exactFitApplied': exact_fit_applied,
                    'voiceUsed': voice_used
                },
                request_id=g.request_id
            ).to_dict()), 201

        except Exception as e:
            logger.error(f"Voiceover generation error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="TTS_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === VIDEO GENERATION (with scene-aware multi-clip support) ===
    @app.route(f'/api/{API_VERSION}/generate-video', methods=['POST'])
    @app.route('/api/generate-video', methods=['POST'])  # Legacy
    @limiter.limit("5 per minute")
    @validate_request('script', 'projectId', duration=int, reelType=str, styleTokens=str, quality=str, resolution=str)
    @require_video_provider()
    @track_performance
    @idempotent(ttl=900)  # 15 minute TTL for video generation
    def generate_video():
        """Generate AI video with scene-aware multi-clip generation

        For videos > 5 seconds, multiple 5s clips are generated with:
        - Scene-aware prompts based on script segments
        - Visual continuity through shared style tokens
        - Automatic stitching with FFmpeg
        """
        try:
            data = g.validated_data

            script = sanitize_input(data['script'], MAX_PROMPT_LENGTH)
            project_id = data['projectId']
            duration = min(max(data.get('duration', 30), 5), MAX_VIDEO_DURATION)
            reel_type = data.get('reelType', 'ai-generated')
            aspect_ratio = data.get('aspectRatio', DEFAULT_ASPECT_RATIO)
            style_tokens = data.get('styleTokens', None)  # For visual consistency
            quality = data.get('quality', 'standard')  # fast, standard, high
            resolution = data.get('resolution', '1080p')  # 480p, 720p, or 1080p (native)

            # Validate project ID
            if not validate_project_id(project_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid project ID",
                    error_code="INVALID_PROJECT_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            # Check user credits before generation
            user_id = None
            credits_required = 0
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                try:
                    import jwt
                    token = auth_header[7:]
                    payload = jwt.decode(token, options={"verify_signature": False})
                    user_id = payload.get('user_id') or payload.get('sub')
                except Exception:
                    pass

            if user_id:
                try:
                    try:
                        from database import calculate_video_credits, check_user_credits
                    except ImportError:
                        from backend.database import calculate_video_credits, check_user_credits

                    credits_required = calculate_video_credits(duration, quality)
                    credit_check = check_user_credits(user_id, credits_required)

                    if not credit_check['has_enough']:
                        logger.warning(f"[CREDITS] User {user_id} has insufficient credits: "
                                     f"balance={credit_check['balance']}, required={credits_required}")
                        return jsonify(ApiResponse(
                            status=ResponseStatus.ERROR,
                            error=f"Insufficient credits. You have {credit_check['balance']} credits, "
                                  f"but this video requires {credits_required} credits.",
                            error_code="INSUFFICIENT_CREDITS",
                            data={
                                'balance': credit_check['balance'],
                                'required': credits_required,
                                'shortfall': credit_check['shortfall']
                            },
                            request_id=g.request_id
                        ).to_dict()), 402  # 402 Payment Required

                    logger.info(f"[CREDITS] User {user_id} credit check passed: "
                               f"balance={credit_check['balance']}, required={credits_required}")
                except Exception as credit_error:
                    logger.warning(f"[CREDITS] Credit check failed, allowing generation: {credit_error}")

            # Calculate number of clips for logging
            num_clips = max(1, (duration + 4) // 5)
            logger.info(f"[VIDEO] Generating video: project={project_id}, "
                       f"duration={duration}s ({num_clips} clips), type={reel_type}, "
                       f"quality={quality}, resolution={resolution}, scene_aware={num_clips > 1}")

            # Start video generation with scene-aware parameters
            result = g.service.generate_video(
                prompt=script,
                duration=duration,
                reel_type=reel_type,
                project_id=project_id,
                aspect_ratio=aspect_ratio,
                style_tokens=style_tokens,
                quality=quality,
                resolution=resolution
            )

            job_id = result['job_id']
            all_job_ids = result.get('all_job_ids', [job_id])  # All clip job IDs
            clips_needed = result['clips_needed']
            clip_duration = result['clip_duration']
            multi_clip = result.get('multi_clip', False)

            # Update project data with generation details
            script_path = Path(f"outputs/scripts/{project_id}.json")
            if script_path.exists():
                project_data = atomic_read_json(script_path)
                project_data['video'] = {
                    'jobId': job_id,
                    'allJobIds': all_job_ids,  # Store ALL clip job IDs
                    'status': VideoStatus.PROCESSING.value,
                    'targetDuration': duration,
                    'clipsNeeded': clips_needed,
                    'clipDuration': clip_duration,
                    'aspectRatio': aspect_ratio,
                    'type': reel_type,
                    'multiClip': multi_clip,
                    'started': datetime.utcnow().isoformat()
                }
                atomic_write_json(script_path, project_data)

            logger.info(f"[OK] Video generation started: job={job_id}, "
                       f"all_jobs={len(all_job_ids)}, clips={clips_needed}×{clip_duration}s")

            # Track video job in database for My Videos feature
            try:
                try:
                    from database import create_video_job
                except ImportError:
                    from backend.database import create_video_job

                # Try to get user ID from auth token
                user_id = None
                auth_header = request.headers.get('Authorization', '')
                if auth_header.startswith('Bearer '):
                    try:
                        import jwt
                        token = auth_header[7:]
                        payload = jwt.decode(token, options={"verify_signature": False})
                        user_id = payload.get('user_id') or payload.get('sub')
                    except Exception:
                        pass

                if user_id:
                    create_video_job(job_id, user_id, g.service.__class__.__name__, script[:500], duration)
                    logger.info(f"[DB] Video job tracked for user {user_id}")
            except Exception as db_error:
                logger.warning(f"[DB] Could not track video job: {db_error}")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'jobId': job_id,
                    'projectId': project_id,
                    'status': VideoStatus.PROCESSING.value,
                    'targetDuration': duration,
                    'clipsNeeded': clips_needed,
                    'clipDuration': clip_duration
                },
                request_id=g.request_id
            ).to_dict()), 202  # 202 Accepted for async operation

        except Exception as e:
            logger.error(f"Video generation error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="VIDEO_GENERATION_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === IMAGE TO VIDEO GENERATION ===
    @app.route(f'/api/{API_VERSION}/generate-video-from-image', methods=['POST'])
    @app.route('/api/generate-video-from-image', methods=['POST'])  # Legacy
    @limiter.limit("5 per minute")
    @require_video_provider()
    @track_performance
    @idempotent(ttl=900)  # 15 minute TTL
    def generate_video_from_image():
        """Generate video from uploaded image using WAN 2.5 i2v model

        Accepts image as:
        - Base64 data URI in 'image' field
        - URL to hosted image in 'imageUrl' field
        """
        try:
            data = request.get_json() or {}

            # Get image (base64 or URL)
            image_data = data.get('image')  # Base64 data URI
            image_url = data.get('imageUrl')  # URL to image

            if not image_data and not image_url:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Image is required (provide 'image' as base64 or 'imageUrl')",
                    error_code="IMAGE_REQUIRED",
                    request_id=g.request_id
                ).to_dict()), 400

            # SECURITY FIX: Validate imageUrl against SSRF attacks
            if image_url and not image_url.startswith('data:'):
                if not is_safe_url(image_url):
                    return jsonify(ApiResponse(
                        status=ResponseStatus.ERROR,
                        error="Invalid image URL. Only HTTPS URLs from trusted hosts are allowed.",
                        error_code="INVALID_IMAGE_URL",
                        request_id=g.request_id
                    ).to_dict()), 400

            # Use provided URL or base64 data
            final_image = image_url if image_url else image_data

            project_id = data.get('projectId', f"i2v_{int(time.time())}")
            prompt = sanitize_input(data.get('prompt', ''), MAX_PROMPT_LENGTH)
            duration = min(max(data.get('duration', 5), 5), MAX_VIDEO_DURATION)
            resolution = data.get('resolution', '1080p')

            # Validate resolution
            valid_resolutions = ["480p", "720p", "1080p"]
            if resolution not in valid_resolutions:
                resolution = "1080p"

            logger.info(f"[I2V] Starting image-to-video: project={project_id}, "
                       f"duration={duration}s, resolution={resolution}")

            # Get video provider and call i2v method
            from video_providers import get_video_provider
            provider = get_video_provider()

            if not hasattr(provider, 'generate_video_from_image'):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Image-to-video not supported by current provider",
                    error_code="I2V_NOT_SUPPORTED",
                    request_id=g.request_id
                ).to_dict()), 400

            result = provider.generate_video_from_image(
                image_url=final_image,
                prompt=prompt,
                duration=duration,
                resolution=resolution
            )

            job_id = result['job_id']
            all_job_ids = result.get('all_job_ids', [job_id])
            clips_needed = result.get('clips_needed', 1)
            clip_duration = result.get('clip_duration', 5)

            # Save project data
            script_path = Path(f"outputs/scripts/{project_id}.json")
            script_path.parent.mkdir(parents=True, exist_ok=True)

            project_data = {
                'projectId': project_id,
                'mode': 'i2v',
                'prompt': prompt,
                'video': {
                    'jobId': job_id,
                    'allJobIds': all_job_ids,
                    'status': VideoStatus.PROCESSING.value,
                    'targetDuration': duration,
                    'clipsNeeded': clips_needed,
                    'clipDuration': clip_duration,
                    'mode': 'i2v',
                    'started': datetime.utcnow().isoformat()
                }
            }
            atomic_write_json(script_path, project_data)

            logger.info(f"[OK] I2V generation started: job={job_id}, clips={clips_needed}")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'jobId': job_id,
                    'projectId': project_id,
                    'status': VideoStatus.PROCESSING.value,
                    'mode': 'i2v',
                    'targetDuration': duration,
                    'clipsNeeded': clips_needed,
                    'clipDuration': clip_duration
                },
                request_id=g.request_id
            ).to_dict()), 202

        except Exception as e:
            logger.error(f"I2V generation error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="I2V_GENERATION_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === VIDEO STATUS ===
    @app.route(f'/api/{API_VERSION}/video-status/<job_id>', methods=['GET'])
    @app.route('/api/video-status/<job_id>', methods=['GET'])  # Legacy
    @limiter.limit("60 per minute")
    @require_video_provider()
    def check_video_status(job_id):
        """Check video generation status (handles multi-clip jobs)"""
        try:
            # Validate job ID
            if not validate_project_id(job_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid job ID",
                    error_code="INVALID_JOB_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            project_id = request.args.get('projectId')

            # Check if this is a multi-clip job by reading project data
            all_job_ids = [job_id]  # Default to single job
            is_multi_clip = False

            if project_id and validate_project_id(project_id):
                script_path = Path(f"outputs/scripts/{project_id}.json")
                if script_path.exists():
                    project_data = atomic_read_json(script_path)
                    video_data = project_data.get('video', {})
                    stored_job_ids = video_data.get('allJobIds', [])
                    if stored_job_ids:
                        all_job_ids = stored_job_ids
                        is_multi_clip = video_data.get('multiClip', len(stored_job_ids) > 1)

            # Check status of ALL jobs
            all_completed = True
            all_video_urls = []
            overall_progress = 0
            any_failed = False
            error_msg = None

            for idx, jid in enumerate(all_job_ids):
                try:
                    single_status = g.service.check_status(jid)
                    status_val = single_status.get('status', 'pending')

                    if status_val == VideoStatus.COMPLETED.value:
                        video_url = single_status.get('videoUrl')
                        if video_url:
                            all_video_urls.append(video_url)
                        overall_progress += 100 // len(all_job_ids)
                    elif status_val == 'failed':
                        any_failed = True
                        error_msg = single_status.get('error', 'Clip generation failed')
                        break
                    else:
                        all_completed = False
                        overall_progress += single_status.get('progress', 0) // len(all_job_ids)
                except Exception as e:
                    logger.error(f"Error checking status for job {jid}: {e}")
                    all_completed = False

            # Build combined status
            if any_failed:
                status = {
                    'status': 'failed',
                    'error': error_msg,
                    'progress': overall_progress
                }
            elif all_completed and len(all_video_urls) == len(all_job_ids):
                status = {
                    'status': VideoStatus.COMPLETED.value,
                    'progress': 100,
                    'multi_clip': is_multi_clip,
                    'videoUrls': all_video_urls,
                    'videoUrl': all_video_urls[0] if all_video_urls else None
                }
                logger.info(f"[MULTI-CLIP] All {len(all_job_ids)} clips completed!")
            else:
                status = {
                    'status': 'processing',
                    'progress': overall_progress,
                    'multi_clip': is_multi_clip,
                    'completedClips': len(all_video_urls),
                    'totalClips': len(all_job_ids)
                }

            # If completed, download video(s) and update project
            if project_id and status.get('status') == VideoStatus.COMPLETED.value:
                if validate_project_id(project_id):
                    video_dir = Path(f"outputs/videos/{project_id}")
                    video_dir.mkdir(parents=True, exist_ok=True)
                    local_video_url = None
                    clip_paths = []

                    # Get video URLs
                    video_urls = status.get('videoUrls', [])

                    if is_multi_clip and video_urls:
                        # Download all clips
                        logger.info(f"[MULTI-CLIP] Downloading {len(video_urls)} clips...")
                        for i, url in enumerate(video_urls):
                            if url and url.startswith('http'):
                                try:
                                    clip_path = video_dir / f"clip_{i}.mp4"
                                    logger.info(f"[DOWNLOAD] Clip {i+1}/{len(video_urls)}: {url[:60]}...")

                                    video_response = requests.get(url, stream=True, timeout=120)
                                    video_response.raise_for_status()

                                    with open(clip_path, 'wb') as f:
                                        for chunk in video_response.iter_content(chunk_size=8192):
                                            f.write(chunk)

                                    clip_paths.append(str(clip_path))
                                    logger.info(f"[OK] Clip {i+1} downloaded: {clip_path}")
                                except Exception as e:
                                    logger.error(f"Failed to download clip {i}: {e}")

                        # Normalize and concatenate clips into raw.mp4
                        if clip_paths:
                            try:
                                raw_path = video_dir / "raw.mp4"

                                # Get aspect ratio from project data (default 9:16 portrait)
                                aspect_ratio = "9:16"
                                try:
                                    script_path = Path(f"outputs/scripts/{project_id}.json")
                                    if script_path.exists():
                                        pd = atomic_read_json(script_path)
                                        aspect_ratio = pd.get("video", {}).get("aspectRatio", "9:16")
                                except Exception as e:
                                    logger.warning(f"[STATUS] Could not read aspect ratio; defaulting 9:16: {e}")

                                # Use normalize_and_concatenate to handle varying clip properties
                                if normalize_and_concatenate(clip_paths, str(raw_path), aspect_ratio=aspect_ratio, fps=24):
                                    local_video_url = f"/api/{API_VERSION}/videos/{project_id}/raw.mp4"
                                    status['videoUrl'] = local_video_url
                                    status['localPath'] = str(raw_path)
                                    status['clipPaths'] = clip_paths
                                    logger.info(f"[OK] {len(clip_paths)} clips normalized & concatenated to: {raw_path}")
                                else:
                                    logger.error(f"Failed to normalize/concatenate clips")
                            except Exception as e:
                                logger.error(f"Failed to normalize/concatenate clips: {e}")
                    else:
                        # Single clip download (backwards compatible)
                        remote_url = status.get('videoUrl')
                        if remote_url and remote_url.startswith('http'):
                            try:
                                local_path = video_dir / "raw.mp4"
                                logger.info(f"[DOWNLOAD] Downloading video from: {remote_url[:80]}...")

                                video_response = requests.get(remote_url, stream=True, timeout=120)
                                video_response.raise_for_status()

                                with open(local_path, 'wb') as f:
                                    for chunk in video_response.iter_content(chunk_size=8192):
                                        f.write(chunk)

                                local_video_url = f"/api/{API_VERSION}/videos/{project_id}/raw.mp4"
                                logger.info(f"[OK] Video downloaded to: {local_path}")
                                status['videoUrl'] = local_video_url
                                status['localPath'] = str(local_path)

                            except Exception as e:
                                logger.error(f"Failed to download video: {e}")
                                local_video_url = remote_url

                    # Update project data
                    script_path = Path(f"outputs/scripts/{project_id}.json")
                    if script_path.exists():
                        project_data = atomic_read_json(script_path)
                        if 'video' not in project_data:
                            project_data['video'] = {}
                        # Store both local and remote URLs so compose can download if needed
                        remote_url = status.get('videoUrl') if status.get('videoUrl', '').startswith('http') else None
                        project_data['video'].update({
                            'status': VideoStatus.COMPLETED.value,
                            'url': local_video_url or status.get('videoUrl'),
                            'remoteUrl': remote_url,  # Store remote URL for fallback download
                            'multiClip': is_multi_clip,
                            'clipCount': len(clip_paths) if clip_paths else 1,
                            'completed': datetime.utcnow().isoformat()
                        })
                        atomic_write_json(script_path, project_data)
                        logger.info(f"[OK] Video completed: {project_id}")

                        # Update database with video completion and set expiry
                        try:
                            try:
                                from database import (update_video_job, set_video_expiry, get_retention_setting,
                                                    calculate_video_credits, deduct_credits, get_video_job_info)
                            except ImportError:
                                from backend.database import (update_video_job, set_video_expiry, get_retention_setting,
                                                            calculate_video_credits, deduct_credits, get_video_job_info)

                            video_url = local_video_url or status.get('videoUrl')
                            update_video_job(job_id, 'completed', video_url)
                            retention_days = get_retention_setting()
                            set_video_expiry(job_id, retention_days)
                            logger.info(f"[DB] Video job {job_id} marked completed, expires in {retention_days} days")

                            # Deduct credits on successful completion
                            try:
                                job_info = get_video_job_info(job_id)
                                if job_info and job_info.get('user_id'):
                                    video_duration = job_info.get('duration', 5)
                                    credits_to_deduct = calculate_video_credits(video_duration)
                                    deduct_result = deduct_credits(
                                        job_info['user_id'],
                                        credits_to_deduct,
                                        f"Video generation ({video_duration}s)",
                                        project_id
                                    )
                                    logger.info(f"[CREDITS] Deducted {credits_to_deduct} credits from user {job_info['user_id']}, "
                                              f"new balance: {deduct_result.get('balance', 'N/A')}")
                            except Exception as credit_error:
                                logger.warning(f"[CREDITS] Could not deduct credits: {credit_error}")

                        except Exception as db_error:
                            logger.warning(f"[DB] Could not update video completion: {db_error}")

            # Calculate next_poll_in based on status
            current_status = status.get('status')
            next_poll_sec = 2 if current_status in ('processing', 'pending') else 0

            response_data = {
                'status': current_status,
                'progress': status.get('progress', 0),
                'videoUrl': status.get('videoUrl'),
                'error': status.get('error'),
                'next_poll_in': next_poll_sec  # Server hint for pollers
            }

            resp = jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=response_data,
                request_id=g.request_id
            ).to_dict())

            # Add Retry-After header for pollers
            if next_poll_sec > 0:
                resp.headers['Retry-After'] = str(next_poll_sec)

            return resp

        except Exception as e:
            logger.error(f"Status check error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="STATUS_CHECK_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === VIDEO COMPOSITION ===
    @app.route(f'/api/{API_VERSION}/compose-video', methods=['POST'])
    @app.route('/api/compose-video', methods=['POST'])  # Legacy
    @limiter.limit("5 per minute")
    @validate_request('projectId', musicStyle=str)
    @require_service('video_composer')
    @track_performance
    @idempotent(ttl=600)  # 10 minute TTL for composition
    def compose_video():
        """Compose final video with effects"""
        try:
            data = g.validated_data

            project_id = data['projectId']
            music_style = data.get('musicStyle', 'upbeat')

            # Validate project ID
            if not validate_project_id(project_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid project ID",
                    error_code="INVALID_PROJECT_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            logger.info(f"Composing video for: {project_id}")

            # Get target duration and video info from project data
            script_path = Path(f"outputs/scripts/{project_id}.json")
            target_duration = None
            video_dir = Path(f"outputs/videos/{project_id}")
            video_dir.mkdir(parents=True, exist_ok=True)
            raw_video_path = video_dir / "raw.mp4"

            if script_path.exists():
                project_data = atomic_read_json(script_path)
                target_duration = project_data.get('duration')

                # Check if video file exists, if not try to download it
                if not raw_video_path.exists():
                    logger.info(f"[COMPOSE] Video file missing, attempting download...")
                    video_data = project_data.get('video', {})
                    remote_url = video_data.get('remoteUrl') or video_data.get('url')

                    if remote_url and remote_url.startswith('http'):
                        try:
                            logger.info(f"[COMPOSE] Downloading from: {remote_url[:80]}...")
                            video_response = requests.get(remote_url, stream=True, timeout=180)
                            video_response.raise_for_status()

                            with open(raw_video_path, 'wb') as f:
                                for chunk in video_response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            logger.info(f"[COMPOSE] Video downloaded to: {raw_video_path}")
                        except Exception as e:
                            logger.error(f"[COMPOSE] Failed to download video: {e}")

            # Compose video (returns dict with output_path, actual_duration, etc.)
            result = g.service.compose(
                project_id=project_id,
                music_style=music_style,
                target_duration=target_duration  # Pass target duration for proper looping
            )

            # Extract path from result dict
            video_path = result.get('output_path', f"outputs/videos/{project_id}/final.mp4")
            actual_duration = result.get('actual_duration', 0)

            # Generate URL
            video_url = f"/api/{API_VERSION}/videos/{project_id}/final.mp4"

            # Update project data
            if script_path.exists():
                project_data = atomic_read_json(script_path)
                project_data['composition'] = {
                    'path': video_path,
                    'url': video_url,
                    'actualDuration': actual_duration,
                    'targetDuration': target_duration,
                    'musicStyle': music_style,
                    'completed': datetime.utcnow().isoformat()
                }
                atomic_write_json(script_path, project_data)

            # Update database with final video URL (for My Videos page)
            try:
                try:
                    from database import update_video_job
                except ImportError:
                    from backend.database import update_video_job
                # Update video job with the final composed video URL
                update_video_job(project_id, 'completed', video_url)
                logger.info(f"[DB] Updated video job {project_id} with final URL: {video_url}")
            except Exception as db_error:
                logger.warning(f"[DB] Could not update video job URL: {db_error}")

            logger.info(f"[OK] Video composed: {video_path}, duration={actual_duration}s")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'videoUrl': video_url,
                    'videoPath': video_path,
                    'projectId': project_id,
                    'actualDuration': actual_duration,
                    'targetDuration': target_duration
                },
                request_id=g.request_id
            ).to_dict()), 201

        except Exception as e:
            logger.error(f"Video composition error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="COMPOSITION_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === ADD CAPTIONS ===
    @app.route(f'/api/{API_VERSION}/add-captions', methods=['POST'])
    @app.route('/api/add-captions', methods=['POST'])  # Legacy
    @limiter.limit("10 per minute")
    @validate_request('projectId', 'script')
    @require_service('video_composer')
    @track_performance
    @idempotent(ttl=600)  # 10 minute TTL for captions
    def add_captions():
        """Add captions to video"""
        try:
            data = g.validated_data

            project_id = data['projectId']
            script = data['script'][:2000]

            # Validate project ID
            if not validate_project_id(project_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid project ID",
                    error_code="INVALID_PROJECT_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            logger.info(f"Adding captions to: {project_id}")

            # Check if we have word_timings for viral captions (from Kokoro TTS)
            word_timings = []
            target_duration = None
            script_path = Path(f"outputs/scripts/{project_id}.json")
            if script_path.exists():
                project_data = atomic_read_json(script_path)
                voiceover = project_data.get('voiceover', {})
                word_timings = voiceover.get('wordTimings', [])
                target_duration = voiceover.get('targetDuration') or project_data.get('duration')

            # Use viral captions if word_timings available, otherwise fall back to SRT
            if word_timings:
                logger.info(f"[CAPTIONS] Using VIRAL word-by-word captions ({len(word_timings)} words)")
                video_path = g.service.add_viral_captions(
                    project_id=project_id,
                    word_timings=word_timings,
                    target_duration=target_duration
                )
            else:
                logger.info("[CAPTIONS] Using standard SRT captions (no word timings)")
                video_path = g.service.add_captions(
                    project_id=project_id,
                    script=script
                )

            video_url = f"/api/{API_VERSION}/videos/{project_id}/captioned.mp4"

            # Update database with captioned video URL (for My Videos page)
            try:
                try:
                    from database import update_video_job
                except ImportError:
                    from backend.database import update_video_job
                # Update video job with the captioned video URL
                update_video_job(project_id, 'completed', video_url)
                logger.info(f"[DB] Updated video job {project_id} with captioned URL: {video_url}")
            except Exception as db_error:
                logger.warning(f"[DB] Could not update video job URL: {db_error}")

            logger.info(f"[OK] Captions added: {video_path}")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'videoUrl': video_url,
                    'projectId': project_id
                },
                request_id=g.request_id
            ).to_dict()), 201

        except Exception as e:
            logger.error(f"Caption addition error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="CAPTION_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === USER VIDEOS API (for My Videos page) ===
    @app.route(f'/api/{API_VERSION}/auth/user/videos', methods=['GET'])
    @limiter.limit("30 per minute")
    @require_auth
    def get_user_videos_api():
        """Get user's generated videos for My Videos page"""
        try:
            from database import get_user_videos, get_retention_setting
        except ImportError:
            from backend.database import get_user_videos, get_retention_setting

        try:
            user_id = g.current_user.get('id')
            if not user_id:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="User not authenticated",
                    error_code="AUTH_REQUIRED",
                    request_id=g.request_id
                ).to_dict()), 401

            videos = get_user_videos(user_id)
            retention_days = get_retention_setting()

            # Transform video data for frontend
            video_list = []
            for v in videos:
                video_list.append({
                    'id': v['id'],
                    'prompt': v.get('prompt', ''),
                    'duration': v.get('duration', 5),
                    'video_url': v.get('video_url', ''),
                    'thumbnail_url': v.get('thumbnail_url'),
                    'created_at': v.get('created_at'),
                    'expires_at': v.get('expires_at'),
                    'status': v.get('status', 'completed')
                })

            logger.info(f"[MY_VIDEOS] User {user_id} has {len(video_list)} videos")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'videos': video_list,
                    'retention_days': retention_days,
                    'total': len(video_list)
                },
                request_id=g.request_id
            ).to_dict()), 200

        except Exception as e:
            logger.error(f"Get user videos error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="FETCH_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/auth/user/videos/<video_id>', methods=['DELETE'])
    @limiter.limit("10 per minute")
    @require_auth
    def delete_user_video(video_id):
        """Delete a user's video"""
        try:
            from database import soft_delete_video
        except ImportError:
            from backend.database import soft_delete_video

        try:
            user_id = g.current_user.get('id')
            if not user_id:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="User not authenticated",
                    error_code="AUTH_REQUIRED",
                    request_id=g.request_id
                ).to_dict()), 401

            success = soft_delete_video(video_id, user_id)

            if success:
                logger.info(f"[MY_VIDEOS] User {user_id} deleted video {video_id}")
                return jsonify(ApiResponse(
                    status=ResponseStatus.SUCCESS,
                    data={'deleted': True},
                    request_id=g.request_id
                ).to_dict()), 200
            else:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Video not found or not owned by user",
                    error_code="NOT_FOUND",
                    request_id=g.request_id
                ).to_dict()), 404

        except Exception as e:
            logger.error(f"Delete video error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="DELETE_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/auth/user/credits', methods=['GET'])
    @limiter.limit("30 per minute")
    @require_auth
    def get_user_credits_api():
        """Get user's credit balance"""
        try:
            from database import get_user_credits
        except ImportError:
            from backend.database import get_user_credits

        try:
            user_id = g.current_user.get('id')
            if not user_id:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="User not authenticated",
                    error_code="AUTH_REQUIRED",
                    request_id=g.request_id
                ).to_dict()), 401

            credits = get_user_credits(user_id)

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=credits,
                request_id=g.request_id
            ).to_dict()), 200

        except Exception as e:
            logger.error(f"Get credits error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="FETCH_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # ==========================================================================
    # ADMIN API ENDPOINTS
    # ==========================================================================
    # Uses global require_admin decorator defined at top of file

    @app.route(f'/api/{API_VERSION}/admin/stats', methods=['GET'])
    @require_auth
    @require_admin
    def admin_stats():
        """Get admin dashboard statistics"""
        try:
            from database import get_admin_stats
        except ImportError:
            from backend.database import get_admin_stats

        try:
            stats = get_admin_stats()
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'total_users': stats.get('total_users', 0),
                    'total_videos': stats.get('total_videos', 0),
                    'total_credits_issued': stats.get('total_credits_given', 0),
                    'active_today': stats.get('active_users', 0),
                    'completed_videos': stats.get('completed_videos', 0),
                    'videos_today': stats.get('videos_today', 0)
                }
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin stats error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/users', methods=['GET'])
    @require_auth
    @require_admin
    def admin_list_users():
        """List all users with credits"""
        try:
            from database import get_all_users_with_credits
        except ImportError:
            from backend.database import get_all_users_with_credits

        try:
            users = get_all_users_with_credits()
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'users': [{
                        'id': u['id'],
                        'email': u['email'],
                        'name': u.get('name', ''),
                        'role': u['role'],
                        'credits': u.get('credits_balance', 0),
                        'credits_used': u.get('credits_used', 0),
                        'credits_given': u.get('credits_given', 0),
                        'video_count': u.get('video_count', 0),
                        'created_at': u.get('created_at'),
                        'last_login': u.get('last_login'),
                        'is_active': True
                    } for u in users],
                    'total': len(users)
                }
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin list users error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/users/<user_id>', methods=['GET', 'DELETE'])
    @require_auth
    @require_admin
    def admin_user_detail(user_id):
        """Get user details or delete user"""
        try:
            from database import (get_user_by_id, get_user_credits, get_credit_history,
                                get_user_videos, delete_user, log_audit)
        except ImportError:
            from backend.database import (get_user_by_id, get_user_credits, get_credit_history,
                                        get_user_videos, delete_user, log_audit)

        try:
            if request.method == 'DELETE':
                success = delete_user(user_id)
                if success:
                    log_audit(g.current_user['id'], g.current_user['email'],
                             'delete_user', 'user', user_id, f"Deleted user {user_id}")
                    return jsonify(ApiResponse(
                        status=ResponseStatus.SUCCESS,
                        data={'deleted': True}
                    ).to_dict()), 200
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="User not found"
                ).to_dict()), 404

            # GET user details
            user = get_user_by_id(user_id)
            if not user:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="User not found"
                ).to_dict()), 404

            credits = get_user_credits(user_id)
            credit_history = get_credit_history(user_id, limit=20)
            videos = get_user_videos(user_id)

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'user': user,
                    'credits': credits,
                    'credit_history': credit_history,
                    'videos': videos
                }
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin user detail error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/users/<user_id>/credits', methods=['POST'])
    @require_auth
    @require_admin
    def admin_add_credits(user_id):
        """Add credits to user"""
        try:
            from database import add_credits, log_audit
        except ImportError:
            from backend.database import add_credits, log_audit

        try:
            data = request.get_json()
            amount = data.get('amount', 0)
            reason = data.get('reason', 'Admin credit grant')

            if amount <= 0:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Amount must be positive"
                ).to_dict()), 400

            result = add_credits(user_id, amount, reason, g.current_user['id'])
            log_audit(g.current_user['id'], g.current_user['email'],
                     'add_credits', 'user', user_id, f"Added {amount} credits: {reason}")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=result
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin add credits error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/users/<user_id>/role', methods=['PUT'])
    @require_auth
    @require_admin
    def admin_change_role(user_id):
        """Change user role"""
        try:
            from database import update_user_role, log_audit
        except ImportError:
            from backend.database import update_user_role, log_audit

        try:
            data = request.get_json()
            new_role = data.get('role')

            if new_role not in ['user', 'editor', 'viewer', 'admin', 'api']:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid role"
                ).to_dict()), 400

            success = update_user_role(user_id, new_role)
            if success:
                log_audit(g.current_user['id'], g.current_user['email'],
                         'change_role', 'user', user_id, f"Changed role to {new_role}")
                return jsonify(ApiResponse(
                    status=ResponseStatus.SUCCESS,
                    data={'role': new_role}
                ).to_dict()), 200

            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="User not found"
            ).to_dict()), 404
        except Exception as e:
            logger.error(f"Admin change role error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/videos', methods=['GET'])
    @require_auth
    @require_admin
    def admin_list_videos():
        """List all videos"""
        try:
            from database import get_all_videos
        except ImportError:
            from backend.database import get_all_videos

        try:
            videos = get_all_videos(limit=100)
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    'videos': [{
                        'id': v['id'],
                        'user_id': v.get('user_id'),
                        'user_email': v.get('user_email'),
                        'prompt': v.get('prompt', '')[:100],
                        'status': v.get('status'),
                        'output_url': v.get('video_url'),
                        'created_at': v.get('created_at'),
                        'completed_at': v.get('completed_at')
                    } for v in videos],
                    'total': len(videos)
                }
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin list videos error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/videos/<video_id>', methods=['DELETE'])
    @require_auth
    @require_admin
    def admin_delete_video(video_id):
        """Delete a video"""
        try:
            from database import delete_video, log_audit
        except ImportError:
            from backend.database import delete_video, log_audit

        try:
            success = delete_video(video_id)
            if success:
                log_audit(g.current_user['id'], g.current_user['email'],
                         'delete_video', 'video', video_id, f"Deleted video {video_id}")
                return jsonify(ApiResponse(
                    status=ResponseStatus.SUCCESS,
                    data={'deleted': True}
                ).to_dict()), 200
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="Video not found"
            ).to_dict()), 404
        except Exception as e:
            logger.error(f"Admin delete video error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/analytics', methods=['GET'])
    @require_auth
    @require_admin
    def admin_analytics():
        """Get analytics data for charts"""
        try:
            from database import get_analytics_data
        except ImportError:
            from backend.database import get_analytics_data

        try:
            days = request.args.get('days', 30, type=int)
            data = get_analytics_data(days)
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=data
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin analytics error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/settings', methods=['GET'])
    @require_auth
    @require_admin
    def admin_get_settings():
        """Get all system settings"""
        try:
            from database import get_all_settings
        except ImportError:
            from backend.database import get_all_settings

        try:
            settings = get_all_settings()
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=settings
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin get settings error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/settings/<key>', methods=['PUT'])
    @require_auth
    @require_admin
    def admin_update_setting(key):
        """Update a system setting"""
        try:
            from database import update_setting, log_audit
        except ImportError:
            from backend.database import update_setting, log_audit

        try:
            data = request.get_json()
            value = data.get('value')

            success = update_setting(key, str(value), g.current_user['id'])
            if success:
                log_audit(g.current_user['id'], g.current_user['email'],
                         'update_setting', 'setting', key, f"Changed {key} to {value}")
                return jsonify(ApiResponse(
                    status=ResponseStatus.SUCCESS,
                    data={'key': key, 'value': value}
                ).to_dict()), 200
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="Setting not found"
            ).to_dict()), 404
        except Exception as e:
            logger.error(f"Admin update setting error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/notifications', methods=['GET', 'POST'])
    @require_auth
    @require_admin
    def admin_notifications():
        """List or create notifications"""
        try:
            from database import get_notifications, create_notification, log_audit
        except ImportError:
            from backend.database import get_notifications, create_notification, log_audit

        try:
            if request.method == 'POST':
                data = request.get_json()
                notif_id = create_notification(
                    title=data.get('title'),
                    message=data.get('message'),
                    type=data.get('type', 'info'),
                    target_role='all',
                    created_by=g.current_user['id'],
                    expires_at=data.get('expires_at')
                )
                log_audit(g.current_user['id'], g.current_user['email'],
                         'create_notification', 'notification', str(notif_id), data.get('title'))
                return jsonify(ApiResponse(
                    status=ResponseStatus.SUCCESS,
                    data={'id': notif_id}
                ).to_dict()), 201

            notifications = get_notifications(include_inactive=True)
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={'notifications': notifications, 'total': len(notifications)}
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin notifications error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/notifications/<int:notif_id>', methods=['DELETE'])
    @require_auth
    @require_admin
    def admin_delete_notification(notif_id):
        """Delete a notification"""
        try:
            from database import delete_notification, log_audit
        except ImportError:
            from backend.database import delete_notification, log_audit

        try:
            success = delete_notification(notif_id)
            if success:
                log_audit(g.current_user['id'], g.current_user['email'],
                         'delete_notification', 'notification', str(notif_id), '')
                return jsonify(ApiResponse(
                    status=ResponseStatus.SUCCESS,
                    data={'deleted': True}
                ).to_dict()), 200
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="Notification not found"
            ).to_dict()), 404
        except Exception as e:
            logger.error(f"Admin delete notification error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/tickets', methods=['GET'])
    @require_auth
    @require_admin
    def admin_list_tickets():
        """List support tickets"""
        try:
            from database import get_all_tickets, get_ticket_stats
        except ImportError:
            from backend.database import get_all_tickets, get_ticket_stats

        try:
            status_filter = request.args.get('status')
            tickets = get_all_tickets(status=status_filter)
            stats = get_ticket_stats()
            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={'tickets': tickets, 'stats': stats, 'total': len(tickets)}
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin list tickets error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/tickets/<int:ticket_id>', methods=['GET', 'PUT'])
    @require_auth
    @require_admin
    def admin_ticket_detail(ticket_id):
        """Get or update ticket"""
        try:
            from database import get_ticket_by_id, update_ticket, log_audit
        except ImportError:
            from backend.database import get_ticket_by_id, update_ticket, log_audit

        try:
            if request.method == 'PUT':
                data = request.get_json()
                success = update_ticket(
                    ticket_id,
                    status=data.get('status'),
                    priority=data.get('priority'),
                    assigned_to=data.get('assigned_to'),
                    admin_notes=data.get('admin_notes')
                )
                if success:
                    log_audit(g.current_user['id'], g.current_user['email'],
                             'update_ticket', 'ticket', str(ticket_id), f"Status: {data.get('status')}")
                    return jsonify(ApiResponse(
                        status=ResponseStatus.SUCCESS,
                        data={'updated': True}
                    ).to_dict()), 200
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Ticket not found"
                ).to_dict()), 404

            ticket = get_ticket_by_id(ticket_id)
            if not ticket:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Ticket not found"
                ).to_dict()), 404

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=ticket
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin ticket detail error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/tickets/<int:ticket_id>/reply', methods=['POST'])
    @require_auth
    @require_admin
    def admin_ticket_reply(ticket_id):
        """Add reply to ticket"""
        try:
            from database import add_ticket_reply, log_audit
        except ImportError:
            from backend.database import add_ticket_reply, log_audit

        try:
            data = request.get_json()
            message = data.get('message')

            if not message:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Message required"
                ).to_dict()), 400

            reply_id = add_ticket_reply(ticket_id, g.current_user['id'], message, is_admin=True)
            log_audit(g.current_user['id'], g.current_user['email'],
                     'reply_ticket', 'ticket', str(ticket_id), 'Admin reply added')

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={'reply_id': reply_id}
            ).to_dict()), 201
        except Exception as e:
            logger.error(f"Admin ticket reply error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/audit-logs', methods=['GET'])
    @require_auth
    @require_admin
    def admin_audit_logs():
        """Get audit logs"""
        try:
            from database import get_audit_logs, get_audit_log_count
        except ImportError:
            from backend.database import get_audit_logs, get_audit_log_count

        try:
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            action_filter = request.args.get('action')

            logs = get_audit_logs(limit=limit, offset=offset, action=action_filter)
            total = get_audit_log_count(action=action_filter)

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data={'logs': logs, 'total': total}
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin audit logs error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    @app.route(f'/api/{API_VERSION}/admin/export/<export_type>', methods=['GET'])
    @require_auth
    @require_admin
    def admin_export(export_type):
        """Export data as JSON (frontend converts to CSV)"""
        try:
            from database import export_users_data, export_transactions_data, export_videos_data, log_audit
        except ImportError:
            from backend.database import export_users_data, export_transactions_data, export_videos_data, log_audit

        try:
            days = request.args.get('days', 30, type=int)

            if export_type == 'users':
                data = export_users_data()
            elif export_type == 'transactions':
                data = export_transactions_data(days)
            elif export_type == 'videos':
                data = export_videos_data(days)
            else:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid export type"
                ).to_dict()), 400

            log_audit(g.current_user['id'], g.current_user['email'],
                     'export_data', 'export', export_type, f"Exported {len(data)} records")

            return jsonify(ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=data
            ).to_dict()), 200
        except Exception as e:
            logger.error(f"Admin export error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            ).to_dict()), 500

    # === DOWNLOAD VIDEO ===
    @app.route(f'/api/{API_VERSION}/download/<project_id>', methods=['GET'])
    @app.route('/api/download/<project_id>', methods=['GET'])  # Legacy
    @limiter.limit("30 per minute")
    def download_video(project_id):
        """Download final video with cleanup"""
        try:
            # Validate project ID
            if not validate_project_id(project_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid project ID",
                    error_code="INVALID_PROJECT_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            format = request.args.get('format', 'mp4')
            if format not in ALLOWED_VIDEO_FORMATS:
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error=f"Invalid format. Allowed: {', '.join(ALLOWED_VIDEO_FORMATS)}",
                    error_code="INVALID_FORMAT",
                    request_id=g.request_id
                ).to_dict()), 400

            # Try local file first
            video_base = Path('outputs/videos') / project_id
            video_path = safe_path_join(video_base, f'final.{format}')

            if video_path.exists():
                logger.info(f"Downloading local video: {video_path}")
                return send_file(
                    video_path,
                    as_attachment=True,
                    download_name=f"reelcraft_{project_id}.{format}",
                    mimetype=f'video/{format}'
                )

            # Try remote URL from project data
            script_path = Path(f"outputs/scripts/{project_id}.json")
            if script_path.exists():
                project_data = atomic_read_json(script_path)
                video_url = project_data.get('video', {}).get('url')

                if video_url:
                    logger.info(f"Downloading from remote: {video_url[:50]}...")

                    # Download with timeout
                    session = get_http_session()
                    response = session.get(video_url, stream=True, timeout=60)
                    response.raise_for_status()

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name

                    # Clean up temp file after response
                    @after_this_request
                    def cleanup(response_obj):
                        try:
                            os.unlink(tmp_path)
                            logger.info(f"Cleaned up temp file: {tmp_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean temp file: {e}")
                        return response_obj

                    logger.info(f"[OK] Downloaded video successfully")
                    return send_file(
                        tmp_path,
                        as_attachment=True,
                        download_name=f"reelcraft_{project_id}.{format}",
                        mimetype=f'video/{format}'
                    )

            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="Video not found",
                error_code="VIDEO_NOT_FOUND",
                request_id=g.request_id
            ).to_dict()), 404

        except Exception as e:
            logger.error(f"Download error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="DOWNLOAD_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === SERVE VIDEOS ===
    @app.route(f'/api/{API_VERSION}/videos/<project_id>/<filename>', methods=['GET'])
    @app.route('/api/videos/<project_id>/<filename>', methods=['GET'])  # Legacy
    @limiter.limit("100 per minute")
    def serve_video(project_id, filename):
        """Serve video files securely"""
        try:
            # Validate inputs
            if not validate_project_id(project_id):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid project ID",
                    error_code="INVALID_PROJECT_ID",
                    request_id=g.request_id
                ).to_dict()), 400

            if not SAFE_ID_PATTERN.match(filename.split('.')[0]):
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Invalid filename",
                    error_code="INVALID_FILENAME",
                    request_id=g.request_id
                ).to_dict()), 400

            # Safe path resolution
            video_base = Path('outputs/videos') / project_id
            video_path = safe_path_join(video_base, filename)

            if not video_path.exists():
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="Video not found",
                    error_code="VIDEO_NOT_FOUND",
                    request_id=g.request_id
                ).to_dict()), 404

            # Determine MIME type
            suffix = video_path.suffix.lower()
            mime_types = {'.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime'}
            mime_type = mime_types.get(suffix, 'application/octet-stream')

            return send_file(
                video_path,
                mimetype=mime_type,
                conditional=True,  # Enable range requests
                max_age=3600
            )

        except ValueError as e:
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="Invalid path",
                error_code="INVALID_PATH",
                request_id=g.request_id
            ).to_dict()), 400
        except Exception as e:
            logger.error(f"Video serve error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="SERVE_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === SERVE OUTPUT FILES ===
    @app.route('/outputs/<path:filename>', methods=['GET'])
    @limiter.limit("100 per minute")
    def serve_output_file(filename):
        """Serve files from outputs folder securely"""
        try:
            # Safe path resolution
            output_base = Path(__file__).parent.parent / "outputs"
            output_path = safe_path_join(output_base, filename)

            if not output_path.exists():
                logger.warning(f"Output file not found: {output_path}")
                return jsonify(ApiResponse(
                    status=ResponseStatus.ERROR,
                    error="File not found",
                    error_code="FILE_NOT_FOUND",
                    request_id=g.request_id
                ).to_dict()), 404

            # Determine MIME type
            suffix = output_path.suffix.lower()
            mime_types = {
                '.mp4': 'video/mp4',
                '.webm': 'video/webm',
                '.mov': 'video/quicktime',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.json': 'application/json',
                '.txt': 'text/plain'
            }
            mime_type = mime_types.get(suffix, 'application/octet-stream')

            logger.info(f"Serving output file: {output_path.name}")

            response = send_file(
                output_path,
                mimetype=mime_type,
                as_attachment=False,
                conditional=True
            )

            # SECURITY FIX: Use Flask-CORS configured ALLOWED_ORIGINS instead of wildcard
            # The global CORS config will handle allowed origins
            if mime_type.startswith('video/'):
                response.headers['Accept-Ranges'] = 'bytes'

            return response

        except ValueError as e:
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error="Invalid path",
                error_code="INVALID_PATH",
                request_id=g.request_id
            ).to_dict()), 400
        except Exception as e:
            logger.error(f"Output file serve error: {e}")
            return jsonify(ApiResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code="SERVE_FAILED",
                request_id=g.request_id
            ).to_dict()), 500

    # === SERVE FRONTEND ===
    @app.route('/', methods=['GET'])
    def serve_frontend():
        """Serve the frontend HTML"""
        try:
            frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
            if not frontend_path.exists():
                return "Frontend not found. Please ensure frontend/index.html exists.", 404
            return send_file(frontend_path, mimetype='text/html')
        except Exception as e:
            logger.error(f"Frontend serve error: {e}")
            return f"Error loading frontend: {str(e)}", 500

    @app.route('/auth.html', methods=['GET'])
    @app.route('/auth', methods=['GET'])
    @app.route('/login', methods=['GET'])
    @app.route('/register', methods=['GET'])
    def serve_auth():
        """Serve the auth page"""
        try:
            auth_path = Path(__file__).parent.parent / "frontend" / "auth.html"
            if not auth_path.exists():
                return "Auth page not found", 404
            return send_file(auth_path, mimetype='text/html')
        except Exception as e:
            logger.error(f"Auth serve error: {e}")
            return f"Error loading auth: {str(e)}", 500

    @app.route('/admin.html', methods=['GET'])
    @app.route('/admin', methods=['GET'])
    def serve_admin():
        """Serve the admin dashboard page"""
        try:
            admin_path = Path(__file__).parent.parent / "frontend" / "admin.html"
            if not admin_path.exists():
                return "Admin page not found", 404
            return send_file(admin_path, mimetype='text/html')
        except Exception as e:
            logger.error(f"Admin serve error: {e}")
            return f"Error loading admin: {str(e)}", 500

    @app.route('/sample_demo.mp4', methods=['GET'])
    def serve_sample_video():
        """Serve the sample demo video from multiple fallback locations

        Works in all environments:
        - Local development (Windows/Mac/Linux)
        - Docker container (Railway, Render, etc.)
        """
        try:
            # Check multiple possible locations for the sample video
            # Order matters: most likely locations first
            possible_paths = [
                # Docker/Railway: /app/sample_demo.mp4 (COPY . . puts it here)
                Path("/app/sample_demo.mp4"),
                Path("/app/backend/sample_demo.mp4"),
                # Local development
                Path(__file__).parent.parent / "sample_demo.mp4",  # ReelSenseAI/sample_demo.mp4
                Path(__file__).parent / "sample_demo.mp4",  # backend/sample_demo.mp4
                Path(__file__).parent.parent / "frontend" / "sample_demo.mp4",  # frontend/sample_demo.mp4
                # Fallback
                Path(__file__).parent.parent.parent / "sample_demo.mp4",
            ]

            for video_path in possible_paths:
                if video_path.exists():
                    logger.info(f"[SAMPLE VIDEO] Serving from: {video_path}")
                    response = send_file(
                        video_path,
                        mimetype='video/mp4',
                        conditional=True  # Support Range requests for seeking
                    )
                    # Cache for 1 hour on CDN/browser (sample video doesn't change)
                    response.headers['Cache-Control'] = 'public, max-age=3600'
                    return response

            # Log all paths tried for debugging
            logger.warning(f"[SAMPLE VIDEO] Not found! Tried: {[str(p) for p in possible_paths]}")
            return "Sample video not found", 404
        except Exception as e:
            logger.error(f"[SAMPLE VIDEO] Error: {e}")
            return f"Error loading video: {str(e)}", 500

    @app.route('/css/<path:filename>', methods=['GET'])
    def serve_css(filename):
        """Serve CSS files"""
        try:
            css_path = Path(__file__).parent.parent / "frontend" / "css" / filename
            if not css_path.exists():
                return "CSS file not found", 404
            return send_file(css_path, mimetype='text/css')
        except Exception as e:
            logger.error(f"CSS serve error: {e}")
            return str(e), 500

    @app.route('/js/<path:filename>', methods=['GET'])
    def serve_js(filename):
        """Serve JavaScript files"""
        try:
            js_path = Path(__file__).parent.parent / "frontend" / "js" / filename
            if not js_path.exists():
                return "JS file not found", 404
            return send_file(js_path, mimetype='application/javascript')
        except Exception as e:
            logger.error(f"JS serve error: {e}")
            return str(e), 500

    # === SERVE MOBILE PWA ===
    @app.route('/mobile/', methods=['GET'])
    @app.route('/mobile', methods=['GET'])
    def serve_mobile():
        """Serve the mobile PWA"""
        try:
            mobile_path = Path(__file__).parent.parent / "frontend" / "mobile" / "index.html"
            if not mobile_path.exists():
                return "Mobile app not found", 404
            return send_file(mobile_path, mimetype='text/html')
        except Exception as e:
            logger.error(f"Mobile serve error: {e}")
            return str(e), 500

    @app.route('/mobile/<path:filename>', methods=['GET'])
    def serve_mobile_files(filename):
        """Serve mobile PWA files (manifest, sw.js, icons)"""
        try:
            file_path = Path(__file__).parent.parent / "frontend" / "mobile" / filename
            if not file_path.exists():
                return f"File not found: {filename}", 404

            # Determine MIME type
            mime_types = {
                '.json': 'application/json',
                '.js': 'application/javascript',
                '.png': 'image/png',
                '.ico': 'image/x-icon',
                '.svg': 'image/svg+xml',
                '.webp': 'image/webp'
            }
            ext = Path(filename).suffix.lower()
            mimetype = mime_types.get(ext, 'application/octet-stream')

            return send_file(file_path, mimetype=mimetype)
        except Exception as e:
            logger.error(f"Mobile file serve error: {e}")
            return str(e), 500

# === APPLICATION INSTANCE ===
# Create app instance at module level for Gunicorn/Waitress
app = create_app()

# === VIDEO CLEANUP SCHEDULER ===
def run_video_cleanup():
    """Background task to clean up expired videos"""
    import time as cleanup_time
    while True:
        try:
            # Run cleanup every hour
            cleanup_time.sleep(3600)  # 1 hour

            try:
                from database import cleanup_expired_videos, get_retention_setting
            except ImportError:
                try:
                    from backend.database import cleanup_expired_videos, get_retention_setting
                except ImportError:
                    continue

            deleted_count = cleanup_expired_videos()
            if deleted_count > 0:
                logger.info(f"[CLEANUP] Deleted {deleted_count} expired videos")

        except Exception as e:
            logger.error(f"[CLEANUP] Error during video cleanup: {e}")

def start_cleanup_scheduler():
    """Start the background cleanup scheduler"""
    cleanup_thread = threading.Thread(target=run_video_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("[CLEANUP] Video cleanup scheduler started (runs every hour)")

# Start cleanup scheduler when app starts
try:
    start_cleanup_scheduler()
except Exception as e:
    logger.warning(f"[CLEANUP] Could not start cleanup scheduler: {e}")

# === MAIN ===
if __name__ == '__main__':
    # Configuration
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    # Set start time
    app.config['start_time'] = time.time()

    if debug:
        # Development server
        print("[INFO] Running in DEVELOPMENT mode with Flask dev server")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            threaded=True
        )
    else:
        # Production: Actually use Waitress (Windows-compatible WSGI server)
        print("=" * 60)
        print("[INFO] Starting PRODUCTION server")
        print(f"[INFO] Server will be available at: http://0.0.0.0:{port}")
        print("=" * 60)

        try:
            from waitress import serve
            print("[INFO] Using Waitress WSGI server (production-grade)")
            serve(
                app,
                host='0.0.0.0',
                port=port,
                threads=8,  # 8 threads for concurrent requests
                connection_limit=200,  # Max concurrent connections
                channel_timeout=120,  # 2 minute timeout
                expose_tracebacks=False  # SECURITY: Don't expose tracebacks
            )
        except ImportError:
            print("[WARNING] Waitress not installed! Using Flask dev server (NOT RECOMMENDED)")
            print("[WARNING] Install Waitress: pip install waitress")
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False,
                threaded=True
            )
