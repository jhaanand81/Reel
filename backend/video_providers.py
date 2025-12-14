# Video Providers - Multi-provider support with multi-clip generation
# See VIDEO_PROVIDERS_GUIDE.md for setup instructions

import os, requests, base64, logging, time, json
from pathlib import Path
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_PROMPT_LENGTH = 2000
DEFAULT_FPS = 16
logger = logging.getLogger(__name__)

# Store for tracking multi-clip jobs
_multi_clip_jobs = {}  # job_id -> {clip_jobs: [...], clips_needed, target_duration, ...}

class VideoProviderBase(ABC):
    @abstractmethod
    def is_available(self) -> bool: pass
    @abstractmethod
    def generate_video(self, prompt: str, duration: int, **kwargs) -> Dict[str, Any]: pass
    @abstractmethod
    def check_status(self, job_id: str) -> Dict[str, Any]: pass
    @property
    @abstractmethod
    def provider_name(self) -> str: pass
    @property
    @abstractmethod
    def max_single_clip_duration(self) -> int: pass

class ReplicateService(VideoProviderBase):
    """Wan 2.2 via Replicate - Apache 2.0, FREE commercial use, up to 1080p

    Uses WAN 2.2 models (NO built-in voice) so app can add custom voiceover:
    - 480p: wan-video/wan-2.2-t2v-480p
    - 720p: wan-video/wan-2.2-t2v-720p
    - 1080p: wan-video/wan-2.2-t2v-1080p

    For Image-to-Video: wan-video/wan-2.1-i2v-720p
    """
    def __init__(self):
        self.api_key = os.getenv('REPLICATE_API_TOKEN')
        self.base_url = "https://api.replicate.com/v1"

        # WAN models - correct Replicate model names (2025)
        self.models_by_resolution = {
            "480p": "wan-video/wan-2.2-t2v-fast",
            "720p": "wan-video/wan-2.2-t2v-fast",
            "1080p": "wan-video/wan-2.2-t2v-fast"
        }
        self.model_i2v = "wan-video/wan-2.2-i2v-fast"  # Image-to-video
        self.resolution = "480p"  # Default to fast 480p model

    def _get_model_for_resolution(self, resolution: str) -> str:
        """Get the appropriate WAN 2.2 model for the requested resolution"""
        return self.models_by_resolution.get(resolution, self.models_by_resolution["1080p"])

    @property
    def provider_name(self): return "replicate"
    @property
    def max_single_clip_duration(self): return 5
    def is_available(self): return bool(self.api_key)

    def _create_scene_prompts(self, base_prompt: str, num_clips: int) -> List[str]:
        """Create varied scene prompts for each clip to ensure unique content"""
        if num_clips == 1:
            return [base_prompt]

        # Scene variation keywords for visual diversity
        scene_variations = [
            "opening shot, establishing view,",
            "close-up detail, focused perspective,",
            "dynamic angle, movement shot,",
            "wide angle, panoramic view,",
            "dramatic lighting, cinematic moment,"
        ]

        prompts = []
        for i in range(num_clips):
            variation = scene_variations[i % len(scene_variations)]
            # Add scene number and variation to prompt
            scene_prompt = f"Scene {i+1}: {variation} {base_prompt}"
            prompts.append(scene_prompt[:MAX_PROMPT_LENGTH])

        return prompts

    def _start_single_clip(self, prompt: str, clip_index: int, resolution: str = None) -> Dict[str, Any]:
        """Start generation for a single clip"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Use provided resolution or default
        res = resolution or self.resolution
        # Get the correct WAN 2.2 model for this resolution
        model = self._get_model_for_resolution(res)
        logger.info(f"[CLIP {clip_index+1}] Using model: {model}, resolution: {res}")

        # Resolution dimensions (9:16 vertical for reels)
        resolution_map = {
            "480p": {"width": 480, "height": 854},
            "720p": {"width": 720, "height": 1280},
            "1080p": {"width": 1080, "height": 1920}
        }
        dims = resolution_map.get(res, resolution_map["480p"])

        payload = {
            "input": {
                "prompt": prompt,
                "width": dims["width"],
                "height": dims["height"],
                "aspect_ratio": "9:16"  # Vertical for reels/shorts
            }
        }

        try:
            # Use models endpoint for community models (e.g., wan-video/wan-2.2-t2v-1080p)
            url = f"{self.base_url}/models/{model}/predictions"
            logger.info(f"[CLIP {clip_index+1}] POST {url}")
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return {
                "clip_index": clip_index,
                "job_id": data.get('id'),
                "status": "starting",
                "prompt": prompt[:100]
            }
        except Exception as e:
            logger.error(f"Failed to start clip {clip_index}: {e}")
            return {
                "clip_index": clip_index,
                "job_id": None,
                "status": "failed",
                "error": str(e)
            }

    def generate_video(self, prompt, duration=5, **kwargs):
        """Generate video - starts multiple clips in parallel for longer durations"""
        clip_duration = self.max_single_clip_duration
        clips_needed = max(1, (duration + clip_duration - 1) // clip_duration)

        # Get resolution from kwargs (passed from frontend)
        resolution = kwargs.get('resolution', self.resolution)
        # Validate resolution
        valid_resolutions = ["480p", "720p", "1080p"]
        if resolution not in valid_resolutions:
            resolution = "1080p"

        model = self._get_model_for_resolution(resolution)
        logger.info(f"[MULTI-CLIP] Generating {clips_needed} unique clips for {duration}s video")
        logger.info(f"[MULTI-CLIP] Model: {model}, Resolution: {resolution}")

        # Create unique prompts for each clip
        scene_prompts = self._create_scene_prompts(prompt, clips_needed)

        # Start all clips in parallel
        clip_jobs = []
        with ThreadPoolExecutor(max_workers=min(clips_needed, 3)) as executor:
            futures = {
                executor.submit(self._start_single_clip, scene_prompts[i], i, resolution): i
                for i in range(clips_needed)
            }
            for future in as_completed(futures):
                result = future.result()
                clip_jobs.append(result)
                logger.info(f"[CLIP {result['clip_index']+1}/{clips_needed}] Job started: {result.get('job_id', 'FAILED')}")

        # Sort by clip index
        clip_jobs.sort(key=lambda x: x['clip_index'])

        # Create master job ID (use first clip's job ID as primary)
        primary_job_id = clip_jobs[0]['job_id'] if clip_jobs[0]['job_id'] else f"multi_{int(time.time())}"

        # Store multi-clip job info
        _multi_clip_jobs[primary_job_id] = {
            "clip_jobs": clip_jobs,
            "clips_needed": clips_needed,
            "clip_duration": clip_duration,
            "target_duration": duration,
            "base_prompt": prompt,
            "created_at": time.time()
        }

        # Collect all job IDs
        all_job_ids = [c['job_id'] for c in clip_jobs if c['job_id']]
        logger.info(f"[MULTI-CLIP] Started {len(all_job_ids)} clips: {all_job_ids}")

        return {
            "job_id": primary_job_id,
            "provider": "replicate",
            "status": "starting",
            "clips_needed": clips_needed,
            "clip_duration": clip_duration,
            "target_duration": duration,
            "clip_jobs": all_job_ids,
            "all_job_ids": all_job_ids,  # Required by main.py
            "multi_clip": clips_needed > 1
        }

    def generate_video_from_image(self, image_url: str, prompt: str = "", duration: int = 5, **kwargs) -> Dict[str, Any]:
        """Generate video from an image using WAN 2.5 i2v model

        Args:
            image_url: URL or base64 data URI of the input image
            prompt: Optional text prompt to guide the animation
            duration: Target duration (will generate multiple clips if > 5s)
            **kwargs: Additional options (resolution, etc.)

        Returns:
            Job info dict with job_id and status
        """
        if not self.is_available():
            raise Exception("Replicate API key not configured")

        resolution = kwargs.get('resolution', self.resolution)
        valid_resolutions = ["480p", "720p", "1080p"]
        if resolution not in valid_resolutions:
            resolution = "1080p"

        clip_duration = self.max_single_clip_duration
        clips_needed = max(1, (duration + clip_duration - 1) // clip_duration)

        logger.info(f"[I2V] Generating {clips_needed} clips from image, resolution: {resolution}")
        logger.info(f"[I2V] Model: {self.model_i2v}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # For i2v, we use the same image for all clips but vary the prompt
        clip_jobs = []
        for i in range(clips_needed):
            # Vary the prompt slightly for each clip
            if clips_needed > 1:
                motion_hints = ["gentle movement", "smooth motion", "dynamic action", "subtle animation", "flowing movement"]
                clip_prompt = f"{prompt}, {motion_hints[i % len(motion_hints)]}, scene {i+1}"
            else:
                clip_prompt = prompt if prompt else "smooth cinematic motion, high quality animation"

            payload = {
                "version": self.model_i2v,
                "input": {
                    "image": image_url,
                    "prompt": clip_prompt[:MAX_PROMPT_LENGTH],
                    "num_frames": 81,  # ~5 seconds at 16fps
                    "resolution": resolution,
                    "sample_steps": 30
                }
            }

            try:
                r = requests.post(f"{self.base_url}/predictions", headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                clip_jobs.append({
                    "clip_index": i,
                    "job_id": data.get('id'),
                    "status": "starting",
                    "prompt": clip_prompt[:100]
                })
                logger.info(f"[I2V CLIP {i+1}/{clips_needed}] Job started: {data.get('id')}")
            except Exception as e:
                logger.error(f"[I2V] Failed to start clip {i}: {e}")
                clip_jobs.append({
                    "clip_index": i,
                    "job_id": None,
                    "status": "failed",
                    "error": str(e)
                })

        # Create master job ID
        primary_job_id = clip_jobs[0]['job_id'] if clip_jobs[0].get('job_id') else f"i2v_{int(time.time())}"

        # Store multi-clip job info
        _multi_clip_jobs[primary_job_id] = {
            "clip_jobs": clip_jobs,
            "clips_needed": clips_needed,
            "clip_duration": clip_duration,
            "target_duration": duration,
            "mode": "i2v",
            "created_at": time.time()
        }

        all_job_ids = [c['job_id'] for c in clip_jobs if c.get('job_id')]

        return {
            "job_id": primary_job_id,
            "provider": "replicate",
            "status": "starting",
            "mode": "i2v",
            "clips_needed": clips_needed,
            "clip_duration": clip_duration,
            "target_duration": duration,
            "clip_jobs": all_job_ids,
            "all_job_ids": all_job_ids,
            "multi_clip": clips_needed > 1
        }

    def check_status(self, job_id):
        """Check status - handles both single and multi-clip jobs"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Check if this is a multi-clip job
        if job_id in _multi_clip_jobs:
            return self._check_multi_clip_status(job_id, headers)

        # Single clip status check
        r = requests.get(f"{self.base_url}/predictions/{job_id}", headers=headers, timeout=30)
        data = r.json()

        status = data.get('status', 'processing')
        result = {
            "job_id": job_id,
            "status": "completed" if status == "succeeded" else "processing"
        }

        if status == "succeeded":
            output = data.get('output')
            if isinstance(output, str):
                result['videoUrl'] = output
            elif isinstance(output, list) and len(output) > 0:
                result['videoUrl'] = output[0]
        elif status == "failed":
            result['status'] = "failed"
            result['error'] = data.get('error', 'Video generation failed')

        return result

    def _check_multi_clip_status(self, job_id: str, headers: Dict) -> Dict[str, Any]:
        """Check status of all clips in a multi-clip job"""
        job_info = _multi_clip_jobs[job_id]
        clip_jobs = job_info['clip_jobs']
        clips_needed = job_info['clips_needed']

        completed_clips = []
        failed_clips = []
        processing_clips = []

        for clip in clip_jobs:
            if not clip.get('job_id'):
                failed_clips.append(clip)
                continue

            try:
                r = requests.get(f"{self.base_url}/predictions/{clip['job_id']}", headers=headers, timeout=30)
                data = r.json()
                clip_status = data.get('status', 'processing')

                if clip_status == "succeeded":
                    output = data.get('output')
                    video_url = output if isinstance(output, str) else (output[0] if output else None)
                    completed_clips.append({
                        "clip_index": clip['clip_index'],
                        "job_id": clip['job_id'],
                        "videoUrl": video_url
                    })
                elif clip_status == "failed":
                    failed_clips.append({
                        "clip_index": clip['clip_index'],
                        "error": data.get('error', 'Unknown error')
                    })
                else:
                    processing_clips.append(clip)
            except Exception as e:
                logger.warning(f"Error checking clip {clip['clip_index']}: {e}")
                processing_clips.append(clip)

        # Calculate progress
        total = clips_needed
        done = len(completed_clips)
        progress = int((done / total) * 100) if total > 0 else 0

        # Determine overall status
        if len(completed_clips) == clips_needed:
            # All clips done - sort by index and return URLs
            completed_clips.sort(key=lambda x: x['clip_index'])
            return {
                "job_id": job_id,
                "status": "completed",
                "clips_needed": clips_needed,
                "clips_completed": len(completed_clips),
                "progress": 100,
                "videoUrls": [c['videoUrl'] for c in completed_clips],
                "videoUrl": completed_clips[0]['videoUrl'],  # Primary URL for backwards compat
                "multi_clip": True
            }
        elif failed_clips and len(failed_clips) + len(completed_clips) == clips_needed:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": f"{len(failed_clips)} clips failed to generate",
                "clips_completed": len(completed_clips),
                "clips_failed": len(failed_clips)
            }
        else:
            return {
                "job_id": job_id,
                "status": "processing",
                "clips_needed": clips_needed,
                "clips_completed": len(completed_clips),
                "clips_processing": len(processing_clips),
                "progress": progress
            }

class RunPodService(VideoProviderBase):
    """Video generation using RunPod Serverless with WAN 2.2

    Requires:
    - RUNPOD_API_KEY: Your RunPod API key
    - RUNPOD_ENDPOINT_ID: Your serverless endpoint ID
    """

    def __init__(self):
        self.api_key = os.getenv('RUNPOD_API_KEY')
        self.endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}" if self.endpoint_id else None
        self.active_jobs = {}

    @property
    def provider_name(self): return "runpod"

    @property
    def max_single_clip_duration(self): return 5

    def is_available(self):
        return bool(self.api_key and self.endpoint_id)

    def generate_video(self, prompt, duration=5, **kwargs):
        """Start video generation on RunPod"""
        if not self.is_available():
            raise Exception("RunPod API key or endpoint ID not configured")

        clip_duration = self.max_single_clip_duration
        clips_needed = max(1, (duration + clip_duration - 1) // clip_duration)

        # Get resolution from kwargs
        resolution = kwargs.get('resolution', '480p')
        valid_resolutions = ["480p", "720p", "1080p"]
        if resolution not in valid_resolutions:
            resolution = "480p"

        # Resolution dimensions (9:16 vertical for reels)
        resolution_map = {
            "480p": {"width": 480, "height": 854},
            "720p": {"width": 720, "height": 1280},
            "1080p": {"width": 1080, "height": 1920}
        }
        dims = resolution_map.get(resolution, resolution_map["480p"])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # RunPod serverless payload
        payload = {
            "input": {
                "prompt": prompt[:MAX_PROMPT_LENGTH],
                "num_frames": 81,  # ~5 seconds at 16fps
                "width": dims["width"],
                "height": dims["height"],
                "resolution": resolution,
                "fps": DEFAULT_FPS,
                "duration": duration,
                "clips_needed": clips_needed
            }
        }

        try:
            logger.info(f"[RUNPOD] Starting video generation: {duration}s, {resolution}")
            response = requests.post(
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
                "prompt": prompt[:100],
                "duration": duration,
                "clips_needed": clips_needed,
                "created_at": time.time()
            }

            logger.info(f"[RUNPOD] Job started: {job_id}")

            return {
                "job_id": job_id,
                "provider": "runpod",
                "status": "starting",
                "clips_needed": clips_needed,
                "clip_duration": clip_duration,
                "target_duration": duration,
                "multi_clip": clips_needed > 1
            }

        except requests.exceptions.RequestException as e:
            raise Exception(f"RunPod API error: {str(e)}")

    def check_status(self, job_id):
        """Check status of a RunPod video generation job"""
        if not self.is_available():
            raise Exception("RunPod API not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(
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

                # Handle video URL or base64
                if 'video_url' in output:
                    result['videoUrl'] = output['video_url']
                elif 'video_base64' in output:
                    result['videoUrl'] = self._save_video_from_base64(output['video_base64'], job_id)
                elif 'videoUrl' in output:
                    result['videoUrl'] = output['videoUrl']

                # Handle multiple clips
                if 'video_urls' in output:
                    result['videoUrls'] = output['video_urls']
                    result['multi_clip'] = True

                result['progress'] = 100
                self.active_jobs.pop(job_id, None)

            elif runpod_status == 'FAILED':
                result['error'] = data.get('error', 'Video generation failed')
                self.active_jobs.pop(job_id, None)

            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"RunPod status check error: {str(e)}")

    def _save_video_from_base64(self, video_base64: str, job_id: str) -> str:
        """Save base64 video to file and return URL"""
        try:
            video_bytes = base64.b64decode(video_base64)

            output_dir = Path("output/videos")
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"runpod_{job_id}.mp4"
            filepath = output_dir / filename

            with open(filepath, 'wb') as f:
                f.write(video_bytes)

            logger.info(f"[RUNPOD] Saved video: {filepath}")
            return f"/output/videos/{filename}"

        except Exception as e:
            logger.error(f"[RUNPOD] Failed to save video: {e}")
            raise

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
            'IN_QUEUE': 10,
            'IN_PROGRESS': 50,
            'COMPLETED': 100,
            'FAILED': 0,
            'CANCELLED': 0,
            'TIMED_OUT': 0
        }
        return progress_map.get(runpod_status, 0)


class DemoService(VideoProviderBase):
    @property
    def provider_name(self): return "demo"
    @property
    def max_single_clip_duration(self): return 15
    def is_available(self): return True
    def generate_video(self, prompt, duration=10, **kwargs):
        import uuid
        clip_duration = min(duration, self.max_single_clip_duration)
        clips_needed = max(1, (duration + clip_duration - 1) // clip_duration)
        return {
            "job_id": f"demo_{uuid.uuid4().hex[:8]}",
            "provider": "demo",
            "status": "processing",
            "demo_mode": True,
            "clips_needed": clips_needed,
            "clip_duration": clip_duration,
            "target_duration": duration
        }
    def check_status(self, job_id):
        return {
            "job_id": job_id,
            "status": "completed",
            "videoUrl": "/sample_demo.mp4",
            "demo_mode": True,
            "progress": 100
        }

class VideoProviderFactory:
    def __init__(self):
        self._providers = {
            'demo': DemoService(),
            'replicate': ReplicateService(),
            'runpod': RunPodService()
        }

    def get_provider(self, name=None):
        """Get video provider by name or from VIDEO_PROVIDER env var

        Provider switching:
        - Set VIDEO_PROVIDER=replicate and REPLICATE_API_TOKEN for Replicate
        - Set VIDEO_PROVIDER=runpod and RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID for RunPod
        - Defaults to 'demo' if provider not available
        """
        name = (name or os.getenv('VIDEO_PROVIDER', 'demo')).lower().replace('-', '_')
        logger.info(f"[PROVIDER] Requested: {name}")

        if name in self._providers:
            provider = self._providers[name]
            if provider.is_available():
                logger.info(f"[PROVIDER] Using: {name}")
                return provider
            else:
                logger.warning(f"[PROVIDER] {name} not available, falling back to demo")

        return self._providers['demo']

    def list_available_providers(self):
        return [{"name": n, "available": p.is_available()} for n, p in self._providers.items()]

video_provider_factory = VideoProviderFactory()
def get_video_provider(name=None): return video_provider_factory.get_provider(name)
def list_video_providers(): return video_provider_factory.list_available_providers()
