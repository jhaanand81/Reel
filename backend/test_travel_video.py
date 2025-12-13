#!/usr/bin/env python3
"""
Test: 15-Second Travel Video with 3 Clips + Kokoro Voiceover
- 3 unique AI-generated clips (5 sec each)
- Stitched together seamlessly
- Exact-fit voiceover with Kokoro TTS
"""

import os
import sys
import time
import subprocess
import json
import requests
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
TARGET_DURATION = 15  # seconds
CLIP_DURATION = 5     # seconds per clip
NUM_CLIPS = 3

# Travel Prompt
SCRIPT = "Escape the ordinary. Discover hidden beaches, ancient cities, breathtaking mountains. Your adventure awaits. Book your dream trip today."

# Scene prompts for each clip
SCENE_PROMPTS = [
    "Hidden tropical beach with crystal clear turquoise water, palm trees swaying, golden sunset light, drone aerial shot, cinematic travel advertisement, 4K quality",
    "Ancient historic city ruins with beautiful architecture, old stone buildings, warm afternoon light, tourists exploring, cinematic travel commercial, 4K quality",
    "Majestic snow-capped mountain peaks at golden hour, dramatic clouds, epic landscape, adventure travel mood, cinematic drone footage, 4K quality"
]

# Output directory
OUTPUT_DIR = Path(r"C:\Users\Anand Jha\Documents\ReelSenseAI_v1.0_Windows_20251208\ReelSenseAI_Windows\backend\outputs\videos\test_travel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'

FFMPEG = get_ffmpeg_path()

def get_replicate_api_key():
    """Get Replicate API key from environment or .env file"""
    key = os.environ.get('REPLICATE_API_TOKEN')
    if key:
        return key

    # Try .env file - check multiple locations
    env_paths = [
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / '.env',
        Path(r"C:\Users\Anand Jha\Documents\ReelSenseAI_v1.0_Windows_20251208\ReelSenseAI_Windows\.env"),
    ]
    for env_path in env_paths:
        if env_path.exists():
            print(f"   Found .env: {env_path}")
            with open(env_path) as f:
                for line in f:
                    if line.startswith('REPLICATE_API_TOKEN='):
                        key = line.split('=', 1)[1].strip().strip('"\'')
                        print(f"   API Key: {key[:10]}...{key[-5:]}")
                        return key
    return None

def get_duration_precise(file_path):
    """Get duration using ffmpeg -i (works reliably on Windows)"""
    import re
    try:
        result = subprocess.run([FFMPEG, '-i', str(file_path)], capture_output=True, text=True, timeout=30)
        match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', result.stderr)
        if match:
            h, m, s, ms = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 100
    except Exception as e:
        print(f"   Duration error: {e}")
    return 0

def start_clip_generation(prompt, clip_index, api_key):
    """Start a single clip generation on Replicate"""
    print(f"   [CLIP {clip_index + 1}] Starting generation...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "version": "wan-video/wan-2.2-t2v-fast",
        "input": {
            "prompt": prompt,
            "num_frames": 81,  # ~5 seconds at 16fps
            "resolution": "480p",
            "sample_steps": 30
        }
    }

    try:
        r = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=30
        )
        data = r.json()
        job_id = data.get('id')
        if not job_id:
            print(f"   [CLIP {clip_index + 1}] API Error: {data}")
        else:
            print(f"   [CLIP {clip_index + 1}] Job ID: {job_id}")
        return {"clip_index": clip_index, "job_id": job_id, "status": "starting"}
    except Exception as e:
        print(f"   [CLIP {clip_index + 1}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"clip_index": clip_index, "job_id": None, "status": "failed", "error": str(e)}

def wait_for_clip(job_id, clip_index, api_key, max_wait=600):
    """Wait for a clip to complete"""
    headers = {"Authorization": f"Bearer {api_key}"}
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            r = requests.get(
                f"https://api.replicate.com/v1/predictions/{job_id}",
                headers=headers,
                timeout=30
            )
            data = r.json()
            status = data.get('status')

            if status == 'succeeded':
                output = data.get('output')
                video_url = output if isinstance(output, str) else (output[0] if output else None)
                print(f"   [CLIP {clip_index + 1}] Completed!")
                return {"clip_index": clip_index, "status": "completed", "video_url": video_url}
            elif status == 'failed':
                error = data.get('error', 'Unknown error')
                print(f"   [CLIP {clip_index + 1}] Failed: {error}")
                return {"clip_index": clip_index, "status": "failed", "error": error}
            else:
                elapsed = int(time.time() - start_time)
                print(f"   [CLIP {clip_index + 1}] Processing... ({elapsed}s)", end='\r')
                time.sleep(5)
        except Exception as e:
            print(f"   [CLIP {clip_index + 1}] Error checking status: {e}")
            time.sleep(5)

    return {"clip_index": clip_index, "status": "timeout"}

def download_clip(url, output_path):
    """Download a video clip"""
    print(f"   Downloading: {output_path.name}")
    r = requests.get(url, timeout=120)
    with open(output_path, 'wb') as f:
        f.write(r.content)
    return output_path

def stitch_clips(clip_paths, output_path, target_duration):
    """Stitch multiple clips into one video"""
    print(f"\n   Stitching {len(clip_paths)} clips...")

    # Create concat file
    concat_file = OUTPUT_DIR / "concat_list.txt"
    with open(concat_file, 'w') as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    # FFmpeg concat
    cmd = [
        FFMPEG, '-y', '-f', 'concat', '-safe', '0',
        '-i', str(concat_file),
        '-t', str(target_duration),
        '-c:v', 'libx264', '-preset', 'fast',
        '-an',  # No audio yet
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"   Stitch failed: {result.stderr[-300:]}")
        return None

    duration = get_duration_precise(output_path)
    print(f"   Stitched video: {duration:.2f}s")
    return output_path

def fit_audio_exact(input_path, output_path, target_seconds, sr=48000):
    """Two-pass exact-fit audio processing"""
    print(f"\n   [EXACT-FIT] Fitting audio to {target_seconds}s...")

    actual = get_duration_precise(input_path)
    if actual <= 0:
        raise RuntimeError(f"Cannot read audio duration")

    ratio = actual / float(target_seconds)
    print(f"   [EXACT-FIT] Source: {actual:.2f}s -> Target: {target_seconds}s (ratio: {ratio:.3f})")

    def atempo_chain(r):
        chain = []
        while r > 2.0:
            chain.append('atempo=2.0')
            r /= 2.0
        while r < 0.5:
            chain.append('atempo=0.5')
            r /= 0.5
        if abs(r - 1.0) > 1e-6:
            chain.append(f'atempo={r:.6f}')
        return ','.join(chain) if chain else 'anull'

    stretch = atempo_chain(ratio)
    filter_str = f"{stretch},apad=whole_dur={target_seconds},atrim=duration={target_seconds}"

    cmd = [
        FFMPEG, '-y', '-i', str(input_path),
        '-af', filter_str,
        '-c:a', 'pcm_s16le', '-ar', str(sr),
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode == 0:
        final_dur = get_duration_precise(output_path)
        print(f"   [EXACT-FIT] Output: {final_dur:.3f}s")
        return str(output_path), final_dur
    else:
        raise RuntimeError(f"Exact-fit failed: {result.stderr[-200:]}")

def generate_voiceover(script, target_duration):
    """Generate Kokoro TTS voiceover with exact-fit"""
    print("\n" + "="*50)
    print("  GENERATING KOKORO VOICEOVER")
    print("="*50)

    try:
        from kokoro import KPipeline

        # Use af_bella for warm, inspiring travel voice
        voice = "af_bella"
        print(f"   Voice: {voice} (warm, inspiring)")
        print(f"   Script: {script[:50]}...")

        raw_audio_path = OUTPUT_DIR / "voiceover_raw.wav"
        exact_audio_path = OUTPUT_DIR / "voiceover_exact.wav"

        # Generate TTS
        print(f"\n   [PASS 1] Generating TTS...")
        t0 = time.time()
        tts = KPipeline(lang_code="a")

        audio_parts = []
        for _, _, audio in tts(script, voice=voice, speed=1.0):
            audio_parts.append(audio)

        if not audio_parts:
            print("   [FAIL] No audio generated")
            return None

        audio = np.concatenate(audio_parts)
        sf.write(str(raw_audio_path), audio, 24000)

        raw_duration = get_duration_precise(raw_audio_path)
        print(f"   [PASS 1] Raw: {raw_duration:.2f}s (generated in {time.time()-t0:.1f}s)")

        # Exact-fit
        print(f"\n   [PASS 2] Exact-fit to {target_duration}s...")
        exact_path, exact_duration = fit_audio_exact(raw_audio_path, exact_audio_path, target_duration)

        print(f"\n   [OK] Voiceover ready: {exact_duration:.3f}s")
        return exact_path, exact_duration

    except ImportError as e:
        print(f"   [FAIL] Kokoro not installed: {e}")
        return None
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compose_final_video(video_path, audio_path, output_path, target_duration):
    """Compose final video with audio"""
    print("\n" + "="*50)
    print("  COMPOSING FINAL VIDEO")
    print("="*50)

    video_dur = get_duration_precise(video_path)
    audio_dur = get_duration_precise(audio_path)

    print(f"   Video: {video_dur:.2f}s")
    print(f"   Audio: {audio_dur:.2f}s")
    print(f"   Target: {target_duration}s")

    cmd = [
        FFMPEG, '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-filter_complex', f"[0:v]trim=duration={target_duration},setpts=PTS-STARTPTS[v];[1:a]atrim=duration={target_duration}[a]",
        '-map', '[v]', '-map', '[a]',
        '-t', str(target_duration),
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"   [FAIL] Composition failed: {result.stderr[-300:]}")
        return None

    final_dur = get_duration_precise(output_path)
    file_size = output_path.stat().st_size / (1024 * 1024)

    print(f"\n   [OK] Final video created!")
    print(f"   File: {output_path}")
    print(f"   Duration: {final_dur:.2f}s")
    print(f"   Size: {file_size:.1f} MB")

    return str(output_path)

def main():
    print("\n" + "#"*60)
    print("  TRAVEL VIDEO TEST - 15 Seconds, 3 Clips")
    print("  Script: " + SCRIPT[:40] + "...")
    print("#"*60)

    t_start = time.time()

    # Get API key
    api_key = get_replicate_api_key()
    if not api_key:
        print("\n[ERROR] No Replicate API key found!")
        print("Set REPLICATE_API_TOKEN environment variable")
        return

    print(f"\n[1/4] GENERATING {NUM_CLIPS} VIDEO CLIPS")
    print("="*50)

    # Start all clips in parallel
    clip_jobs = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(start_clip_generation, SCENE_PROMPTS[i], i, api_key): i
            for i in range(NUM_CLIPS)
        }
        for future in as_completed(futures):
            result = future.result()
            clip_jobs.append(result)

    clip_jobs.sort(key=lambda x: x['clip_index'])

    # Wait for all clips
    print("\n   Waiting for clips to complete...")
    completed_clips = []
    for job in clip_jobs:
        if job.get('job_id'):
            result = wait_for_clip(job['job_id'], job['clip_index'], api_key)
            if result['status'] == 'completed':
                completed_clips.append(result)

    if len(completed_clips) < NUM_CLIPS:
        print(f"\n[ERROR] Only {len(completed_clips)}/{NUM_CLIPS} clips completed")
        return

    print(f"\n   All {NUM_CLIPS} clips generated!")

    # Download clips
    print("\n[2/4] DOWNLOADING CLIPS")
    print("="*50)

    clip_paths = []
    for clip in sorted(completed_clips, key=lambda x: x['clip_index']):
        clip_path = OUTPUT_DIR / f"clip_{clip['clip_index']}.mp4"
        download_clip(clip['video_url'], clip_path)
        clip_paths.append(clip_path)

    # Stitch clips
    print("\n[3/4] STITCHING CLIPS")
    print("="*50)

    raw_video = OUTPUT_DIR / "raw_stitched.mp4"
    if not stitch_clips(clip_paths, raw_video, TARGET_DURATION):
        print("[ERROR] Failed to stitch clips")
        return

    # Generate voiceover
    print("\n[4/4] GENERATING VOICEOVER")

    vo_result = generate_voiceover(SCRIPT, TARGET_DURATION)
    if not vo_result:
        print("[ERROR] Failed to generate voiceover")
        return

    audio_path, audio_dur = vo_result

    # Compose final video
    final_video = OUTPUT_DIR / "final_travel.mp4"
    result = compose_final_video(raw_video, audio_path, final_video, TARGET_DURATION)

    if result:
        total_time = time.time() - t_start
        print("\n" + "#"*60)
        print("  SUCCESS!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Output: {final_video}")
        print("#"*60)
    else:
        print("\n[ERROR] Final composition failed")

if __name__ == "__main__":
    main()
