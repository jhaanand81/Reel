#!/usr/bin/env python3
"""
Quick Kokoro TTS Test - Uses existing raw.mp4 video
Tests Kokoro TTS voiceover generation and composition

TWO-PASS EXACT-FIT SYSTEM:
1. Generate TTS naturally at 1.0x speed
2. Exact-fit pass: trim silence, time-stretch, pad/trim to exact duration
"""

import os
import sys
import time
import subprocess
import re
import json
import numpy as np
import soundfile as sf
from pathlib import Path

# Existing video from previous test
RAW_VIDEO = Path(r"C:\Users\Anand Jha\Documents\ReelSenseAI_v1.0_Windows_20251208\ReelSenseAI_Windows\backend\outputs\videos\test_e123f55f\raw.mp4")
OUTPUT_DIR = RAW_VIDEO.parent
TARGET_DURATION = 10

def get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'

FFMPEG = get_ffmpeg_path()

def get_duration(file_path):
    try:
        result = subprocess.run([FFMPEG, '-i', str(file_path)],
                               capture_output=True, text=True, timeout=30)
        match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', result.stderr)
        if match:
            h, m, s, ms = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 100
        return 0
    except:
        return 0

def get_duration_precise(file_path):
    """Get precise duration using ffprobe JSON output"""
    try:
        # Try ffprobe first (more accurate)
        ffprobe = FFMPEG.replace('ffmpeg', 'ffprobe') if 'ffmpeg' in FFMPEG.lower() else 'ffprobe'
        result = subprocess.run([
            ffprobe, '-v', 'quiet', '-print_format', 'json', '-show_format', str(file_path)
        ], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info['format']['duration'])
    except:
        pass
    # Fallback to ffmpeg parsing
    return get_duration(file_path)

def fit_audio_exact(input_path, output_path, target_seconds, sr=48000, max_rubberband_change=0.18):
    """
    Two-pass exact-fit: trim silence, time-stretch precisely, pad/trim to exact length.

    This gives natural pacing plus sample-accurate duration.

    Operations:
    1. silenceremove - kills invisible drift from TTS (tiny silence at head/tail)
    2. atempo - precise time-stretch by ratio = actual/target
    3. apad + atrim - guarantees exact length even with sub-millisecond rounding
    """
    print(f"\n   [EXACT-FIT] Fitting audio to {target_seconds}s...")

    actual = get_duration_precise(input_path)
    if actual <= 0:
        raise RuntimeError(f"Cannot read audio duration from {input_path}")

    ratio = actual / float(target_seconds)  # >1 means speed up, <1 means slow down
    change_percent = abs(1.0 - ratio) * 100

    print(f"   [EXACT-FIT] Source: {actual:.2f}s, Target: {target_seconds}s")
    print(f"   [EXACT-FIT] Ratio: {ratio:.4f} ({change_percent:.1f}% {'speed up' if ratio > 1 else 'slow down'})")

    def atempo_chain(r):
        """Build atempo filter chain (FFmpeg atempo supports 0.5-2.0)"""
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
    print(f"   [EXACT-FIT] Method: atempo ({change_percent:.1f}% change)")

    # Filter configurations - prioritize reliable ones
    # NOTE: silenceremove with -50dB is too aggressive for TTS audio
    # It treats speech as silence. Use only for very quiet padding removal.
    filter_configs = [
        # Config 1: Time-stretch + pad + trim (most reliable)
        [
            stretch,
            f"apad=whole_dur={target_seconds}",
            f"atrim=duration={target_seconds}"
        ],
        # Config 2: With gentle silence removal (only removes actual silence, -30dB)
        [
            "silenceremove=start_periods=1:start_threshold=-30dB:start_silence=0.01:stop_periods=1:stop_threshold=-30dB:stop_silence=0.01",
            stretch,
            f"apad=whole_dur={target_seconds}",
            f"atrim=duration={target_seconds}"
        ],
        # Config 3: Minimal - just stretch and trim
        [
            stretch,
            f"atrim=duration={target_seconds}"
        ]
    ]

    for i, afilters in enumerate(filter_configs):
        # Skip anull if ratio is ~1.0
        if 'anull' in afilters:
            afilters = [f for f in afilters if f != 'anull']

        filter_str = ','.join(afilters) if afilters else 'anull'

        cmd = [
            FFMPEG, '-y', '-i', str(input_path),
            '-af', filter_str,
            '-c:a', 'pcm_s16le',
            '-ar', str(sr),
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            # Verify output duration
            final_dur = get_duration_precise(output_path)
            print(f"   [EXACT-FIT] Output: {final_dur:.3f}s (target: {target_seconds}s)")

            if abs(final_dur - target_seconds) > 0.1:
                print(f"   [EXACT-FIT] Warning: Duration off by {abs(final_dur - target_seconds):.3f}s")
            else:
                print(f"   [EXACT-FIT] Perfect fit!")

            return str(output_path), final_dur
        else:
            if i < len(filter_configs) - 1:
                print(f"   [EXACT-FIT] Config {i+1} failed, trying simpler config...")

    raise RuntimeError(f"FFmpeg exact-fit failed after all configs: {result.stderr[-300:]}")

# Kokoro American English Voice Selection
KOKORO_VOICES = {
    # Female Voices
    "af_bella": {"name": "Bella", "style": "warm, conversational", "best_for": ["lifestyle", "wellness", "beauty"]},
    "af_sarah": {"name": "Sarah", "style": "clear, professional", "best_for": ["corporate", "tech", "education"]},
    "af_nicole": {"name": "Nicole", "style": "friendly, casual", "best_for": ["social", "food", "travel"]},
    "af_sky": {"name": "Sky", "style": "youthful, energetic", "best_for": ["fitness", "gaming", "youth"]},
    "af_nova": {"name": "Nova", "style": "modern, engaging", "best_for": ["fashion", "luxury", "modern"]},
    "af_jessica": {"name": "Jessica", "style": "professional narrator", "best_for": ["documentary", "explainer", "product"]},
    "af_river": {"name": "River", "style": "calm, soothing", "best_for": ["meditation", "spa", "nature"]},
    "af_heart": {"name": "Heart", "style": "emotional, expressive", "best_for": ["emotional", "story", "inspirational"]},
    "af_aoede": {"name": "Aoede", "style": "storytelling", "best_for": ["narrative", "brand story", "heritage"]},
    # Male Voices
    "am_adam": {"name": "Adam", "style": "deep, authoritative", "best_for": ["finance", "automotive", "serious"]},
    "am_michael": {"name": "Michael", "style": "warm, conversational", "best_for": ["casual", "friendly", "approachable"]},
}

def select_voice_for_content(script, category=None):
    """Select best voice based on script content or category"""
    script_lower = script.lower()

    # Category-based selection
    category_voice_map = {
        "beauty": "af_river",      # Calm, soothing for skincare/beauty
        "skincare": "af_river",
        "wellness": "af_bella",    # Warm for wellness
        "fitness": "af_sky",       # Energetic for fitness
        "tech": "af_sarah",        # Professional for tech
        "luxury": "af_nova",       # Modern for luxury
        "food": "af_nicole",       # Friendly for food
        "automotive": "am_adam",   # Authoritative for cars
        "finance": "am_adam",      # Serious for finance
        "lifestyle": "af_bella",   # Warm for lifestyle
        "nature": "af_river",      # Soothing for nature
        "inspirational": "af_heart",  # Emotional for inspiration
    }

    if category and category.lower() in category_voice_map:
        return category_voice_map[category.lower()]

    # Content-based detection
    if any(word in script_lower for word in ["skincare", "beauty", "glow", "spa", "botanical", "serum"]):
        return "af_river"  # Calm, soothing
    elif any(word in script_lower for word in ["energy", "workout", "fitness", "power", "dynamic"]):
        return "af_sky"  # Youthful, energetic
    elif any(word in script_lower for word in ["luxury", "premium", "elegant", "sophisticated"]):
        return "af_nova"  # Modern, engaging
    elif any(word in script_lower for word in ["tech", "innovation", "smart", "digital"]):
        return "af_sarah"  # Clear, professional
    elif any(word in script_lower for word in ["story", "journey", "heritage", "tradition"]):
        return "af_aoede"  # Storytelling
    elif any(word in script_lower for word in ["inspire", "dream", "believe", "heart"]):
        return "af_heart"  # Emotional, expressive

    # Default: af_heart (most natural and expressive)
    return "af_heart"

def trim_script_to_duration(script, target_seconds, words_per_second=2.3):
    """Trim script to fit target duration at natural speaking pace"""
    words = script.split()
    max_words = int(target_seconds * words_per_second)

    if len(words) <= max_words:
        return script, False  # No trimming needed

    # Trim at sentence/phrase boundary if possible
    trimmed = ' '.join(words[:max_words])

    # Try to end at a natural break (comma, period)
    for punct in ['.', ',', '—', '-']:
        last_punct = trimmed.rfind(punct)
        if last_punct > len(trimmed) * 0.7:  # Keep at least 70%
            trimmed = trimmed[:last_punct + 1]
            break

    return trimmed, True

def generate_kokoro_voiceover(use_exact_fit=True):
    """
    Generate voiceover with Kokoro TTS - TWO-PASS EXACT-FIT SYSTEM

    Pass 1: Generate TTS naturally at 1.0x speed (no forcing)
    Pass 2: Exact-fit to target duration (silence trim, time-stretch, pad/trim)

    This gives natural pacing PLUS sample-accurate duration!
    """
    print("\n" + "="*60)
    print("  KOKORO TTS VOICEOVER (Two-Pass Exact-Fit)")
    print("="*60)

    original_script = """Beautiful morning light streaming through water droplets on green leaves,
    glass skincare bottle emerging from misty botanical garden, soft golden
    hour glow, luxurious natural spa atmosphere, premium beauty commercial."""
    original_script = ' '.join(original_script.split())

    original_words = len(original_script.split())

    # PASS 1: Generate naturally - allow slightly longer scripts since we'll exact-fit
    # Target ~2.5 words/sec to allow for natural speech variation
    script, was_trimmed = trim_script_to_duration(original_script, TARGET_DURATION, words_per_second=2.5)
    word_count = len(script.split())

    print(f"Original: {original_words} words")
    if was_trimmed:
        print(f"Trimmed:  {word_count} words (allows natural pace)")
        print(f"Script: {script}")
    else:
        print(f"Script: {script[:60]}...")

    estimated_duration = word_count / 2.3
    print(f"Estimated TTS duration: {estimated_duration:.1f}s")
    print(f"Target duration: {TARGET_DURATION}s")

    try:
        from kokoro import KPipeline

        # Smart voice selection based on content
        voice = select_voice_for_content(script)
        voice_info = KOKORO_VOICES.get(voice, {})
        print(f"Voice: {voice} ({voice_info.get('name', '')}) - {voice_info.get('style', '')}")

        raw_audio_path = OUTPUT_DIR / "voiceover_raw.wav"
        exact_audio_path = OUTPUT_DIR / "voiceover_exact.wav"

        # PASS 1: Generate TTS naturally
        print(f"\n[PASS 1] Generating TTS at natural speed (1.0x)...")
        t0 = time.time()
        tts = KPipeline(lang_code="a")
        print(f"   Model loaded in {time.time()-t0:.1f}s")

        t0 = time.time()
        audio_parts = []
        for _, _, audio in tts(script, voice=voice, speed=1.0):
            audio_parts.append(audio)

        if not audio_parts:
            print("[FAIL] No audio generated")
            return None

        audio = np.concatenate(audio_parts)
        sf.write(str(raw_audio_path), audio, 24000)

        gen_time = time.time() - t0
        raw_duration = get_duration_precise(raw_audio_path)

        print(f"[PASS 1] Raw voiceover: {raw_duration:.2f}s (generated in {gen_time:.1f}s)")

        # PASS 2: Exact-fit to target duration
        if use_exact_fit:
            print(f"\n[PASS 2] Exact-fit to {TARGET_DURATION}s...")
            try:
                exact_path, exact_duration = fit_audio_exact(
                    raw_audio_path,
                    exact_audio_path,
                    TARGET_DURATION,
                    sr=48000
                )
                final_path = exact_path
                final_duration = exact_duration
                print(f"[PASS 2] Fitted voiceover: {exact_duration:.3f}s")
            except Exception as e:
                print(f"[PASS 2] Exact-fit failed: {e}")
                print("[PASS 2] Using raw audio with composition-time adjustment")
                final_path = str(raw_audio_path)
                final_duration = raw_duration
        else:
            final_path = str(raw_audio_path)
            final_duration = raw_duration

        print(f"\n[OK] Voiceover ready!")
        print(f"   File: {final_path}")
        print(f"   Duration: {final_duration:.3f}s (target: {TARGET_DURATION}s)")
        print(f"   Voice: Kokoro {voice} ({voice_info.get('name', '')})")

        return final_path, final_duration

    except ImportError as e:
        print(f"[FAIL] Kokoro not installed: {e}")
        return None
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compose_video(audio_path, audio_duration):
    """
    Compose video with exact-fit voiceover.

    Since audio is already exact-fit, composition is simple:
    - Trim/loop video to target duration
    - Use audio as-is (already perfect duration)
    """
    print("\n" + "="*60)
    print("  COMPOSE VIDEO + EXACT-FIT VOICEOVER")
    print("="*60)

    if not RAW_VIDEO.exists():
        print(f"[FAIL] Raw video not found: {RAW_VIDEO}")
        return None

    video_duration = get_duration(RAW_VIDEO)
    print(f"Video: {RAW_VIDEO} ({video_duration:.1f}s)")
    print(f"Audio: {audio_path} ({audio_duration:.3f}s)")
    print(f"Target: {TARGET_DURATION}s")

    output_path = OUTPUT_DIR / "final_kokoro.mp4"
    target = TARGET_DURATION

    # Build filter complex
    filters = []

    # Video handling - trim/loop to exact target
    if video_duration < target:
        loop_count = int(target / video_duration) + 1
        filters.append(f"[0:v]loop=loop={loop_count}:size=32767,trim=duration={target},setpts=PTS-STARTPTS[v]")
        print(f"   Video: Looping to {target}s")
    else:
        filters.append(f"[0:v]trim=duration={target},setpts=PTS-STARTPTS[v]")
        print(f"   Video: Trimmed to {target}s")

    # Audio handling - already exact-fit, just ensure duration
    # Small tolerance for any sub-millisecond differences
    if abs(audio_duration - target) < 0.5:
        print(f"   Audio: Already exact-fit ({audio_duration:.3f}s)")
        filters.append(f"[1:a]atrim=duration={target}[a]")
    else:
        # Fallback if exact-fit wasn't applied
        print(f"   Audio: Applying final adjustment...")
        if audio_duration > target:
            ratio = audio_duration / target
            if ratio <= 1.15:
                filters.append(f"[1:a]atempo={ratio:.6f},atrim=duration={target}[a]")
            else:
                filters.append(f"[1:a]atempo=1.15,atrim=duration={target}[a]")
        else:
            filters.append(f"[1:a]apad=whole_dur={target},atrim=duration={target}[a]")

    filter_complex = ';'.join(filters)

    cmd = [
        FFMPEG, '-y',
        '-i', str(RAW_VIDEO),
        '-i', audio_path,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-t', str(target),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-c:a', 'aac',
        '-b:a', '192k',
        str(output_path)
    ]

    print("\nComposing...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"[FAIL] FFmpeg error: {result.stderr[-500:]}")
        return None

    final_duration = get_duration_precise(output_path)
    print(f"\n[OK] Final video: {output_path}")
    print(f"   Duration: {final_duration:.3f}s (target: {target}s)")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Verify duration accuracy
    if abs(final_duration - target) <= 0.1:
        print(f"   Accuracy: PERFECT (within 100ms)")
    elif abs(final_duration - target) <= 0.5:
        print(f"   Accuracy: Good (within 500ms)")
    else:
        print(f"   Accuracy: Off by {abs(final_duration - target):.2f}s")

    return str(output_path), final_duration

def main():
    print("\n" + "#"*60)
    print("  KOKORO TTS - TWO-PASS EXACT-FIT TEST")
    print("  Natural pacing + sample-accurate duration")
    print("#"*60)

    t_start = time.time()

    # Step 1: Generate voiceover with two-pass exact-fit
    result = generate_kokoro_voiceover(use_exact_fit=True)
    if not result:
        print("\n[FAIL] Voiceover generation failed")
        return False

    audio_path, audio_duration = result

    # Step 2: Compose video with exact-fit audio
    result = compose_video(audio_path, audio_duration)
    if not result:
        print("\n[FAIL] Composition failed")
        return False

    final_path, final_duration = result
    total_time = time.time() - t_start

    # Summary
    print("\n" + "="*60)
    print("  RESULTS - TWO-PASS EXACT-FIT SYSTEM")
    print("="*60)
    print(f"Total processing time: {total_time:.1f}s")
    print(f"")
    print(f"Voiceover:")
    print(f"   Raw TTS: Generated at natural 1.0x speed")
    print(f"   Exact-fit: Silence trimmed, time-stretched, padded")
    print(f"   Final audio duration: {audio_duration:.3f}s")
    print(f"")
    print(f"Final video: {final_path}")
    print(f"   Duration: {final_duration:.3f}s (target: {TARGET_DURATION}s)")
    print(f"   Error: {abs(final_duration - TARGET_DURATION)*1000:.1f}ms")

    if abs(final_duration - TARGET_DURATION) <= 0.1:
        print("\n[SUCCESS] Perfect fit! Duration within 100ms")
        print(f"\nPlay the video: {final_path}")
        return True
    elif abs(final_duration - TARGET_DURATION) <= 0.5:
        print("\n[SUCCESS] Good fit! Duration within 500ms")
        print(f"\nPlay the video: {final_path}")
        return True
    else:
        print("\n[FAIL] Duration mismatch")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
