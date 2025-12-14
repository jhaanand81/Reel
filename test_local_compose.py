"""
REELSENSE AI - Perfect Sync Composition
- FULL voiceover plays (video loops if needed)
- ACTUAL TTS timing for captions
"""

import os
import sys
import subprocess
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

VIDEO_1 = r"C:\Users\Anand Jha\Downloads\Himalayas\output_30fps (7).mp4"
VIDEO_2 = r"C:\Users\Anand Jha\Downloads\Himalayas\output_30fps (8).mp4"
OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set to None for voice-driven duration, or a number (e.g., 10.0) for exact duration
MAX_DURATION = 10.0  # Force exactly 10 seconds (300 frames @ 30fps)

TEST_SCRIPT = """The majestic Himalayas rise above the clouds. Ancient peaks touch the heavens. Snow-capped mountains stretch across the horizon. A place of wonder and timeless beauty."""

def get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except:
        return "ffmpeg"

def get_video_info(video_path):
    ffmpeg = get_ffmpeg_path()
    result = subprocess.run([ffmpeg, "-i", str(video_path)], capture_output=True, text=True)
    output = result.stderr

    dur_match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', output)
    duration = 0
    if dur_match:
        h, m, s, ms = dur_match.groups()
        # Fix: proper millisecond parsing (ms can be 2 or 3 digits)
        ms_value = float(f"0.{ms}")
        duration = int(h) * 3600 + int(m) * 60 + int(s) + ms_value

    dim_match = re.search(r'(\d{3,4})x(\d{3,4})', output)
    width, height = 0, 0
    if dim_match:
        width, height = int(dim_match.group(1)), int(dim_match.group(2))

    return {"duration": duration, "width": width, "height": height}

def combine_videos_to_vertical():
    """Combine videos to 9:16 vertical"""
    print("\n" + "="*60)
    print("STEP 1: Combine Videos to 9:16 Vertical")
    print("="*60)

    ffmpeg = get_ffmpeg_path()

    info1 = get_video_info(VIDEO_1)
    info2 = get_video_info(VIDEO_2)
    print(f"Video 1: {info1['duration']:.2f}s")
    print(f"Video 2: {info2['duration']:.2f}s")
    print(f"TOTAL: {info1['duration'] + info2['duration']:.2f}s")

    combined_video = OUTPUT_DIR / "combined_vertical.mp4"

    # Use filter_complex for reliable video concatenation (avoids path issues)
    # Trim each video to exactly 5.000s (150 frames @ 30fps) for clean 10.000s total
    cmd = [
        ffmpeg, "-y",
        "-i", VIDEO_1,
        "-i", VIDEO_2,
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,"
        "trim=0:5,setpts=PTS-STARTPTS,setsar=1[v0];"
        "[1:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,"
        "trim=0:5,setpts=PTS-STARTPTS,setsar=1[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[outv]",
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(combined_video)
    ]

    print("Concatenating videos...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        out_info = get_video_info(combined_video)
        print(f"Combined: {out_info['width']}x{out_info['height']}, {out_info['duration']:.2f}s")
        return str(combined_video), out_info['duration']
    else:
        print(f"ERROR: {result.stderr[-300:]}")
        return None, 0

def generate_voiceover_with_timing():
    """Generate voiceover with PROPORTIONAL word-level timing"""
    print("\n" + "="*60)
    print("STEP 2: Generate Voiceover with Word-Level Timing")
    print("="*60)

    try:
        from kokoro import KPipeline
        import soundfile as sf
        import numpy as np

        print("Initializing Kokoro TTS...")
        pipeline = KPipeline(lang_code="a")

        print(f"Generating speech...")

        audio_chunks = []
        chunk_timings = []  # Renamed: these are CHUNK timings
        current_sample = 0

        for gs, ps, audio in pipeline(TEST_SCRIPT, voice="af_heart"):
            chunk_samples = len(audio)
            start_time = current_sample / 24000
            end_time = (current_sample + chunk_samples) / 24000

            chunk_timings.append({
                "text": gs.strip(),
                "start": start_time,
                "end": end_time
            })

            audio_chunks.append(audio)
            current_sample += chunk_samples
            print(f"  CHUNK [{start_time:.2f}s - {end_time:.2f}s] '{gs.strip()}'")

        full_audio = np.concatenate(audio_chunks)
        voice_duration = len(full_audio) / 24000
        print(f"\nVOICE DURATION: {voice_duration:.2f}s")

        # PROPORTIONAL WORD TIMING: distribute chunk time across words
        word_timings = []
        print("\nCalculating word-level timing (proportional):")

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
                print(f"    [{current_time:.2f}s - {current_time + word_duration:.2f}s] '{word}'")
                current_time += word_duration

        print(f"\nGenerated {len(word_timings)} word timings")

        audio_path = OUTPUT_DIR / "voiceover.wav"
        sf.write(str(audio_path), full_audio, 24000)

        return str(audio_path), word_timings, voice_duration

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, [], 0

def generate_synced_captions(word_timings):
    """Generate VIRAL captions with WORD-BY-WORD highlighting"""
    print("\n" + "="*60)
    print("STEP 3: Generate VIRAL Captions (Word-by-Word Highlight)")
    print("="*60)

    if not word_timings:
        print("No timing data!")
        return None

    def format_ass_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # Title case helper (don't capitalize small words)
    def smart_title(word):
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'of', 'in'}
        w = word.lower().rstrip('.,!?')
        if w in small_words and len(word) < 4:
            return word.lower()
        return word.capitalize()

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

    print(f"Created {len(groups)} semantic phrase groups:")

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
                word = smart_title(wt["text"].strip())
                if j == word_idx:
                    # HIGHLIGHT current word: Yellow color
                    line_parts.append(f"{{\\c&H00FFFF&}}{word}{{\\c&HFFFFFF&}}")
                else:
                    line_parts.append(word)

            caption_text = " ".join(line_parts)
            # Clean punctuation spacing
            caption_text = caption_text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")

            event = f"Dialogue: 0,{format_ass_time(word_start)},{format_ass_time(word_end)},Main,,0,0,0,,{caption_text}"
            events.append(event)

        # Print group summary
        group_text = " ".join([smart_title(w["text"].strip()) for w in group])
        group_text = group_text.replace(" .", ".").replace(" ,", ",")
        print(f"  [{format_ass_time(group[0]['start'])} -> {format_ass_time(group[-1]['end'])}] {group_text}")

    ass_content += "\n".join(events)

    ass_path = OUTPUT_DIR / "captions_viral.ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    print(f"\nViral captions with word highlight saved: {ass_path}")
    print(f"Total dialogue events: {len(events)} (one per word)")
    return str(ass_path)

def extend_video_to_duration(video_path, target_duration):
    """Loop/extend video to match voice duration"""
    print("\n" + "="*60)
    print(f"STEP 3.5: Extend Video to {target_duration:.2f}s")
    print("="*60)

    ffmpeg = get_ffmpeg_path()
    video_info = get_video_info(video_path)

    if video_info['duration'] >= target_duration:
        print(f"Video ({video_info['duration']:.2f}s) is long enough")
        return video_path

    print(f"Video ({video_info['duration']:.2f}s) is shorter than voice ({target_duration:.2f}s)")
    print("Looping video to match voice duration...")

    extended_video = OUTPUT_DIR / "extended_video.mp4"

    # Loop video to match target duration
    cmd = [
        ffmpeg, "-y",
        "-stream_loop", "-1",  # Loop infinitely
        "-i", video_path,
        "-t", str(target_duration),  # Cut at target duration
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(extended_video)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        out_info = get_video_info(extended_video)
        print(f"Extended video: {out_info['duration']:.2f}s")
        return str(extended_video)
    else:
        print(f"Loop failed, using original: {result.stderr[-200:]}")
        return video_path

def final_composition(video_path, audio_path, ass_path, target_duration, fps=30):
    """Final composition with EXACT duration control"""
    import math

    print("\n" + "="*60)
    print(f"STEP 4: Final Composition - EXACT {target_duration:.3f}s @ {fps}fps")
    print("="*60)

    ffmpeg = get_ffmpeg_path()
    output_path = OUTPUT_DIR / "FINAL_MASTERPIECE.mp4"

    # Escape path for FFmpeg ASS filter
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

    # Calculate exact frame count for target duration
    frames = int(round(target_duration * fps))
    print(f"Video: {frames} frames = {frames/fps:.3f}s @ {fps}fps")

    # AAC-safe audio duration (avoid packet padding pushing over target)
    # AAC uses 1024 samples per frame. Trim to nearest lower boundary.
    sr = 24000  # Kokoro sample rate
    aac_frame = 1024
    audio_cap = math.floor(target_duration * sr / aac_frame) * aac_frame / sr
    print(f"Audio: {audio_cap:.4f}s (AAC-safe, avoids packet padding)")

    # Use filter_complex to control video and audio separately
    vf = f"fps={fps},ass='{ass_escaped}'"
    af = f"atrim=0:{audio_cap},asetpts=PTS-STARTPTS"

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", f"[0:v]{vf}[v];[1:a]{af}[a]",
        "-map", "[v]",
        "-map", "[a]",
        "-t", str(target_duration),  # Video duration
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    print(f"Creating VIRAL masterpiece...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        out_info = get_video_info(output_path)
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"File: {output_path}")
        print(f"Size: {out_info['width']}x{out_info['height']}")
        print(f"Duration: {out_info['duration']:.2f}s")
        return str(output_path)
    else:
        print(f"ERROR: {result.stderr[-500:]}")
        return None

def main():
    print("\n" + "="*60)
    print("REELSENSE AI - PERFECT SYNC")
    print("Voice duration = Final video duration")
    print("="*60)

    if not Path(VIDEO_1).exists() or not Path(VIDEO_2).exists():
        print("ERROR: Input videos not found")
        return

    # Step 1: Combine videos
    combined_video, video_duration = combine_videos_to_vertical()
    if not combined_video:
        return

    # Step 2: Generate voiceover FIRST (this determines final duration)
    audio_path, word_timings, voice_duration = generate_voiceover_with_timing()
    if not audio_path:
        return

    # Calculate final_duration: use MAX_DURATION exactly when set
    if MAX_DURATION is not None:
        final_duration = MAX_DURATION  # Force exact duration (pads with silence if voice shorter)
    else:
        final_duration = voice_duration

    print(f"\n>>> VOICE DURATION: {voice_duration:.2f}s")
    print(f">>> FINAL DURATION: {final_duration:.3f}s (MAX_DURATION={MAX_DURATION}) <<<\n")

    # Step 3: Generate captions
    srt_path = generate_synced_captions(word_timings)
    if not srt_path:
        return

    # Step 3.5: Extend video if shorter than final_duration
    final_video_source = extend_video_to_duration(combined_video, final_duration)

    # Step 4: Final composition with EXACT final_duration
    final_video = final_composition(final_video_source, audio_path, srt_path, final_duration)

    if final_video:
        print(f"\n{'='*60}")
        print("OPEN YOUR VIDEO:")
        print(f"  {final_video}")
        print(f"\nNOTE: If VLC shows filename at bottom,")
        print("go to Subtitle > Sub Track > Disable")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
