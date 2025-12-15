"""
Regenerate missing thumbnails for existing videos.
Run: python regenerate_thumbnails.py
"""

import os
import subprocess
from pathlib import Path

VIDEOS_DIR = Path("outputs/videos")
THUMBNAIL_WIDTH = 480

def generate_thumbnail_ffmpeg(video_path: Path, thumbnail_path: Path) -> bool:
    """Generate thumbnail from video using FFmpeg"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", "00:00:01",
            "-vframes", "1",
            "-vf", f"scale={THUMBNAIL_WIDTH}:-1",
            "-q:v", "2",
            str(thumbnail_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except FileNotFoundError:
        return None  # FFmpeg not found
    except Exception as e:
        print(f"  FFmpeg error: {e}")
        return False

def generate_thumbnail_cv2(video_path: Path, thumbnail_path: Path) -> bool:
    """Generate thumbnail using OpenCV (fallback)"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))

        # Seek to 1 second
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        # Resize to thumbnail width
        h, w = frame.shape[:2]
        new_w = THUMBNAIL_WIDTH
        new_h = int(h * new_w / w)
        frame = cv2.resize(frame, (new_w, new_h))

        cv2.imwrite(str(thumbnail_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True
    except ImportError:
        return None  # OpenCV not installed
    except Exception as e:
        print(f"  OpenCV error: {e}")
        return False

def generate_thumbnail(video_path: Path, thumbnail_path: Path) -> bool:
    """Generate thumbnail - tries FFmpeg first, then OpenCV"""
    # Try FFmpeg
    result = generate_thumbnail_ffmpeg(video_path, thumbnail_path)
    if result is True:
        return True
    elif result is False:
        return False  # FFmpeg failed with error

    # FFmpeg not found, try OpenCV
    result = generate_thumbnail_cv2(video_path, thumbnail_path)
    if result is True:
        return True
    elif result is False:
        return False

    # Neither available
    print("  Neither FFmpeg nor OpenCV available!")
    return False

def main():
    print("=" * 60)
    print("Thumbnail Regeneration Tool")
    print("=" * 60)

    if not VIDEOS_DIR.exists():
        print(f"Videos directory not found: {VIDEOS_DIR}")
        return

    # Find all video directories
    video_dirs = [d for d in VIDEOS_DIR.iterdir() if d.is_dir()]
    print(f"\nFound {len(video_dirs)} video projects\n")

    generated = 0
    skipped = 0
    failed = 0

    for video_dir in video_dirs:
        project_id = video_dir.name
        thumbnail_path = video_dir / "thumbnail.jpg"

        # Check if thumbnail already exists
        if thumbnail_path.exists():
            print(f"[SKIP] {project_id} - thumbnail exists")
            skipped += 1
            continue

        # Find video file (prefer final.mp4, then captioned.mp4, then raw.mp4)
        video_path = None
        for video_name in ["final.mp4", "captioned.mp4", "raw.mp4"]:
            candidate = video_dir / video_name
            if candidate.exists():
                video_path = candidate
                break

        if not video_path:
            print(f"[SKIP] {project_id} - no video file found")
            skipped += 1
            continue

        # Generate thumbnail
        print(f"[GEN] {project_id} from {video_path.name}...", end=" ")
        if generate_thumbnail(video_path, thumbnail_path):
            size_kb = thumbnail_path.stat().st_size / 1024
            print(f"OK ({size_kb:.1f}KB)")
            generated += 1
        else:
            print("FAILED")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
