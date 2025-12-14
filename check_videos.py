"""Quick check of video dimensions"""
import subprocess
import re

def get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except:
        return "ffmpeg"

def check_video(path):
    ffmpeg = get_ffmpeg_path()
    result = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True)
    output = result.stderr

    # Duration
    dur_match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', output)
    if dur_match:
        h, m, s, ms = dur_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 100
    else:
        duration = 0

    # Dimensions
    dim_match = re.search(r'(\d{3,4})x(\d{3,4})', output)
    if dim_match:
        w, h = int(dim_match.group(1)), int(dim_match.group(2))
        aspect = w / h if h > 0 else 0
        orientation = "9:16 VERTICAL" if aspect < 1 else "16:9 horizontal"
    else:
        w, h, aspect, orientation = 0, 0, 0, "unknown"

    return {"duration": duration, "width": w, "height": h, "aspect": aspect, "orientation": orientation}

videos = [
    r"C:\Users\Anand Jha\Downloads\Himalayas\output_30fps (7).mp4",
    r"C:\Users\Anand Jha\Downloads\Himalayas\output_30fps (8).mp4"
]

print("\n" + "="*70)
print("VIDEO COMPARISON")
print("="*70)

for v in videos:
    info = check_video(v)
    print(f"\n{v.split(chr(92))[-1]}:")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Size: {info['width']}x{info['height']}")
    print(f"  Aspect: {info['aspect']:.3f} ({info['orientation']})")
