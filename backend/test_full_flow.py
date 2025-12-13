#!/usr/bin/env python3
"""
Full Flow Test - Tests complete video generation pipeline
Script → Voiceover → Video → Compose

Run: python test_full_flow.py
"""

import os
import sys
import time
import json
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
TARGET_DURATION = 10  # Target: 10 seconds exactly

# Test project
PROJECT_ID = f"test_{uuid.uuid4().hex[:8]}"

# Directories
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SCRIPTS_DIR = OUTPUTS_DIR / "scripts"
AUDIO_DIR = OUTPUTS_DIR / "audio" / PROJECT_ID
VIDEO_DIR = OUTPUTS_DIR / "videos" / PROJECT_ID

# Create directories
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}")

def print_result(success, message):
    icon = "[OK]" if success else "[FAIL]"
    print(f"{icon} {message}")

def get_ffmpeg_path():
    """Get ffmpeg path from imageio_ffmpeg"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'

def get_media_duration(file_path):
    """Get duration of media file using ffmpeg"""
    import subprocess
    import re
    ffmpeg = get_ffmpeg_path()
    try:
        result = subprocess.run([
            ffmpeg, '-i', str(file_path)
        ], capture_output=True, text=True, timeout=30)
        # FFmpeg outputs duration to stderr
        output = result.stderr
        # Look for Duration: HH:MM:SS.MS
        match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', output)
        if match:
            hours, minutes, seconds, ms = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 100
        return 0
    except:
        return 0

def step1_generate_script():
    """Generate a script for the target duration"""
    print_step(1, "GENERATE SCRIPT")

    # Script optimized for 10 seconds (~25 words at 150 WPM)
    # Nature/Wellness skincare commercial
    script = """Beautiful morning light streaming through water droplets on green leaves,
glass skincare bottle emerging from misty botanical garden, soft golden
hour glow, luxurious natural spa atmosphere, premium beauty commercial."""

    # Clean up script
    script = ' '.join(script.split())
    word_count = len(script.split())
    estimated_duration = word_count / 2.5  # ~2.5 words per second for voiceover

    print(f"Script: {script[:80]}...")
    print(f"Word count: {word_count}")
    print(f"Estimated voiceover duration: {estimated_duration:.1f}s")
    print(f"Target duration: {TARGET_DURATION}s")

    # Save project data
    project_data = {
        "projectId": PROJECT_ID,
        "script": script,
        "duration": TARGET_DURATION,
        "wordCount": word_count,
        "estimatedDuration": estimated_duration
    }

    project_file = SCRIPTS_DIR / f"{PROJECT_ID}.json"
    with open(project_file, 'w') as f:
        json.dump(project_data, f, indent=2)

    print_result(True, f"Script saved to {project_file}")
    return script

def trim_script_to_duration(script, target_seconds, words_per_second=2.3):
    """Trim script to fit target duration at natural speaking pace"""
    words = script.split()
    max_words = int(target_seconds * words_per_second)

    if len(words) <= max_words:
        return script, False

    trimmed = ' '.join(words[:max_words])
    for punct in ['.', ',', '—', '-']:
        last_punct = trimmed.rfind(punct)
        if last_punct > len(trimmed) * 0.7:
            trimmed = trimmed[:last_punct + 1]
            break

    return trimmed, True

def step2_generate_voiceover(script):
    """Generate voiceover using Kokoro TTS at natural pace (smart script trimming)"""
    print_step(2, "GENERATE VOICEOVER (Kokoro TTS - Natural Pace)")

    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline

        voice = "af_heart"  # Best natural voice
        audio_path = AUDIO_DIR / "voiceover.wav"

        # Trim script to fit naturally at 1.0x speed
        original_words = len(script.split())
        script, was_trimmed = trim_script_to_duration(script, TARGET_DURATION, words_per_second=2.3)
        word_count = len(script.split())

        print(f"Original: {original_words} words")
        if was_trimmed:
            print(f"Trimmed:  {word_count} words (to fit {TARGET_DURATION}s naturally)")
        print(f"Estimated duration: {word_count / 2.3:.1f}s at natural pace")

        print(f"Loading Kokoro TTS (voice: {voice})...")
        tts = KPipeline(lang_code="a")  # American English

        print(f"Generating voiceover at natural speed (1.0x)...")
        t0 = time.time()

        # Generate audio at natural 1.0x speed
        audio_parts = []
        for _, _, audio in tts(script, voice=voice, speed=1.0):
            audio_parts.append(audio)

        if not audio_parts:
            print_result(False, "No audio generated")
            return None

        # Concatenate and save
        audio = np.concatenate(audio_parts)
        sf.write(str(audio_path), audio, 24000)  # Kokoro outputs at 24kHz

        gen_time = time.time() - t0
        print(f"   Generation time: {gen_time:.1f}s")

        if not audio_path.exists():
            print_result(False, "Voiceover file not created")
            return None

        duration = get_media_duration(audio_path)
        print_result(True, f"Voiceover generated: {audio_path}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Voice: Kokoro {voice} (ultra-natural)")

        # Check if we need to adjust speed
        if abs(duration - TARGET_DURATION) > 1.0:
            print(f"   [WARN] Duration mismatch! Will be handled in compose step.")

        return str(audio_path), duration

    except ImportError as e:
        print_result(False, f"Kokoro not installed: {e}")
        print("   Run: pip install kokoro soundfile numpy")
        return None
    except Exception as e:
        print_result(False, f"Voiceover error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_scene_prompts(base_prompt, num_clips):
    """Create varied scene prompts for visual diversity"""
    if num_clips == 1:
        return [base_prompt]

    scene_variations = [
        "opening shot, establishing view,",
        "close-up detail, focused perspective,",
        "dynamic angle, movement shot,",
    ]

    prompts = []
    for i in range(num_clips):
        variation = scene_variations[i % len(scene_variations)]
        scene_prompt = f"Scene {i+1}: {variation} {base_prompt}"
        prompts.append(scene_prompt[:2000])

    return prompts


def start_clip_generation(prompt, clip_index, headers):
    """Start a single clip generation"""
    payload = {
        "version": "wan-video/wan-2.2-t2v-fast",
        "input": {
            "prompt": prompt,
            "num_frames": 81,
            "resolution": "480p",
            "sample_steps": 30
        }
    }

    try:
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return {
            "clip_index": clip_index,
            "job_id": data.get('id'),
            "status": "starting"
        }
    except Exception as e:
        return {
            "clip_index": clip_index,
            "job_id": None,
            "status": "failed",
            "error": str(e)
        }


def poll_clip_completion(job_id, headers, max_wait=300):
    """Poll for a single clip to complete"""
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            return None, "timeout"

        try:
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{job_id}",
                headers=headers,
                timeout=30
            )
            data = response.json()
            status = data.get('status')

            if status == "succeeded":
                output = data.get('output')
                video_url = output if isinstance(output, str) else output[0] if output else None
                return video_url, "completed"
            elif status == "failed":
                return None, data.get('error', 'failed')

            time.sleep(3)
        except Exception as e:
            time.sleep(5)

    return None, "unknown"


def concatenate_clips(clip_paths, output_path):
    """Concatenate video clips using FFmpeg"""
    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return True

    ffmpeg = get_ffmpeg_path()

    # Create concat file
    concat_file = Path(output_path).parent / "concat_list.txt"
    with open(concat_file, 'w') as f:
        for clip_path in clip_paths:
            safe_path = str(clip_path).replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        ffmpeg, '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',
        str(output_path)
    ]

    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        concat_file.unlink(missing_ok=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Concat error: {e}")
        return False


def step3_generate_video(script):
    """Generate video using Replicate API - Multi-clip for longer durations"""
    print_step(3, "GENERATE VIDEO (Replicate/Wan 2.2 - MULTI-CLIP)")

    if not REPLICATE_API_TOKEN:
        print_result(False, "REPLICATE_API_TOKEN not set in .env")
        return None

    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Calculate clips needed (5 seconds per clip)
    clip_duration = 5
    clips_needed = max(1, (TARGET_DURATION + clip_duration - 1) // clip_duration)

    print(f"Target: {TARGET_DURATION}s = {clips_needed} unique clips x {clip_duration}s each")

    # Create scene-varied prompts
    scene_prompts = create_scene_prompts(script, clips_needed)

    # Start all clips in parallel
    print(f"\nStarting {clips_needed} clip generations in parallel...")
    clip_jobs = []
    for i, prompt in enumerate(scene_prompts):
        print(f"   Clip {i+1}: {prompt[:60]}...")
        job_info = start_clip_generation(prompt, i, headers)
        clip_jobs.append(job_info)
        if job_info['job_id']:
            print(f"   [OK] Job ID: {job_info['job_id']}")
        else:
            print(f"   [FAIL] {job_info.get('error', 'Unknown error')}")

    # Check if all jobs started
    valid_jobs = [j for j in clip_jobs if j['job_id']]
    if len(valid_jobs) < clips_needed:
        print_result(False, f"Only {len(valid_jobs)}/{clips_needed} clips started")
        return None

    # Poll all clips for completion
    print(f"\nPolling {clips_needed} clips for completion...")
    completed_clips = []
    start_time = time.time()

    for job in clip_jobs:
        job_id = job['job_id']
        clip_index = job['clip_index']
        print(f"   Waiting for Clip {clip_index + 1}...")

        video_url, status = poll_clip_completion(job_id, headers)
        elapsed = time.time() - start_time

        if video_url:
            print(f"   [OK] Clip {clip_index + 1} completed ({elapsed:.0f}s total)")
            completed_clips.append({
                "clip_index": clip_index,
                "url": video_url
            })
        else:
            print(f"   [FAIL] Clip {clip_index + 1}: {status}")

    if len(completed_clips) < clips_needed:
        print_result(False, f"Only {len(completed_clips)}/{clips_needed} clips completed")
        return None

    # Download all clips
    print(f"\nDownloading {clips_needed} clips...")
    completed_clips.sort(key=lambda x: x['clip_index'])
    clip_paths = []

    for clip in completed_clips:
        clip_path = VIDEO_DIR / f"clip_{clip['clip_index']}.mp4"
        try:
            response = requests.get(clip['url'], stream=True, timeout=120)
            response.raise_for_status()
            with open(clip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            clip_paths.append(str(clip_path))
            clip_duration = get_media_duration(clip_path)
            print(f"   [OK] Clip {clip['clip_index'] + 1}: {clip_path.name} ({clip_duration:.1f}s)")
        except Exception as e:
            print(f"   [FAIL] Clip {clip['clip_index'] + 1}: {e}")

    if len(clip_paths) < clips_needed:
        print_result(False, f"Only {len(clip_paths)}/{clips_needed} clips downloaded")
        return None

    # Concatenate clips
    print(f"\nConcatenating {clips_needed} clips...")
    raw_path = VIDEO_DIR / "raw.mp4"

    if concatenate_clips(clip_paths, str(raw_path)):
        duration = get_media_duration(raw_path)
        print_result(True, f"Video concatenated: {raw_path}")
        print(f"   Total duration: {duration:.1f}s ({clips_needed} clips)")
        print(f"   Size: {raw_path.stat().st_size / 1024:.1f} KB")
        return str(raw_path), duration
    else:
        print_result(False, "Failed to concatenate clips")
        return None

def step4_compose(audio_path, audio_duration, video_path, video_duration):
    """Compose video + audio with exact duration control"""
    print_step(4, "COMPOSE VIDEO + AUDIO")

    import subprocess

    output_path = VIDEO_DIR / "final.mp4"
    target = TARGET_DURATION

    print(f"Video duration: {video_duration:.1f}s")
    print(f"Audio duration: {audio_duration:.1f}s")
    print(f"Target duration: {target}s")

    # Build filter complex
    filters = []

    # Video filter
    if video_duration < target - 0.5:
        # Loop video
        loop_count = int(target / video_duration) + 1
        filters.append(f"[0:v]loop=loop={loop_count}:size={int(video_duration * 16)}:start=0,trim=duration={target},setpts=PTS-STARTPTS[v]")
        print(f"   Video: Looping {loop_count}x to reach {target}s")
    elif video_duration > target + 0.5:
        # Trim video
        filters.append(f"[0:v]trim=duration={target},setpts=PTS-STARTPTS[v]")
        print(f"   Video: Trimming to {target}s")
    else:
        filters.append("[0:v]setpts=PTS-STARTPTS[v]")
        print("   Video: Duration OK")

    # Audio filter - Max 1.1x speed-up to preserve natural sound
    MAX_SPEEDUP = 1.1  # Barely noticeable speed increase

    if audio_duration > target + 0.5:
        speed_factor = audio_duration / target
        if speed_factor <= MAX_SPEEDUP:
            print(f"   Audio: Speeding up {speed_factor:.2f}x (natural)")
            filters.append(f"[1:a]atempo={speed_factor:.4f}[a]")
        else:
            print(f"   Audio: Speeding up {MAX_SPEEDUP:.2f}x (max) + trimming")
            filters.append(f"[1:a]atempo={MAX_SPEEDUP:.4f},atrim=duration={target},asetpts=PTS-STARTPTS[a]")
    elif audio_duration < target - 0.5:
        speed_factor = audio_duration / target
        if speed_factor >= 0.9:
            print(f"   Audio: Slowing down {speed_factor:.2f}x to match {target}s")
            filters.append(f"[1:a]atempo={speed_factor:.4f}[a]")
        else:
            print(f"   Audio: Padding with silence to reach {target}s")
            filters.append(f"[1:a]apad=whole_dur={target}[a]")
    else:
        filters.append("[1:a]anull[a]")
        print("   Audio: Duration OK")

    filter_complex = ';'.join(filters)

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, '-y',
        '-i', video_path,
        '-i', audio_path,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-t', str(target),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        str(output_path)
    ]

    print("\nRunning FFmpeg...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print_result(False, f"FFmpeg error: {result.stderr[:200]}")
            return None

        if not output_path.exists():
            print_result(False, "Output file not created")
            return None

        final_duration = get_media_duration(output_path)

        print_result(True, f"Final video: {output_path}")
        print(f"   Duration: {final_duration:.1f}s")
        print(f"   Target: {target}s")
        print(f"   Difference: {abs(final_duration - target):.2f}s")

        if abs(final_duration - target) <= 1.0:
            print_result(True, "Duration is within tolerance!")
        else:
            print_result(False, f"Duration mismatch! Expected {target}s, got {final_duration:.1f}s")

        return str(output_path), final_duration

    except Exception as e:
        print_result(False, f"Compose error: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("  REELSENSE AI - FULL FLOW TEST")
    print("="*60)
    print(f"Project ID: {PROJECT_ID}")
    print(f"Target Duration: {TARGET_DURATION} seconds")

    # Step 1: Script
    script = step1_generate_script()
    if not script:
        return False

    # Step 2: Voiceover
    voiceover_result = step2_generate_voiceover(script)
    if not voiceover_result:
        return False
    audio_path, audio_duration = voiceover_result

    # Step 3: Video
    video_result = step3_generate_video(script)
    if not video_result:
        return False
    video_path, video_duration = video_result

    # Step 4: Compose
    compose_result = step4_compose(audio_path, audio_duration, video_path, video_duration)
    if not compose_result:
        return False
    final_path, final_duration = compose_result

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"Project ID: {PROJECT_ID}")
    print(f"Script: {script[:50]}...")
    print(f"Voiceover: {audio_path} ({audio_duration:.1f}s)")
    print(f"Raw Video: {video_path} ({video_duration:.1f}s)")
    print(f"Final Video: {final_path} ({final_duration:.1f}s)")
    print(f"Target: {TARGET_DURATION}s")

    success = abs(final_duration - TARGET_DURATION) <= 1.0

    if success:
        print("\n[SUCCESS] TEST PASSED - Full flow works correctly!")
        print(f"\n[VIDEO] Play the video: {final_path}")
    else:
        print(f"\n[FAILED] TEST FAILED - Duration mismatch")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
