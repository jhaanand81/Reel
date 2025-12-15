"""Check thumbnail status for all videos in database vs file system"""
import sqlite3
from pathlib import Path

VIDEOS_DIR = Path("outputs/videos")
db_path = Path('data/reelsense.db')

print("=" * 80)
print("THUMBNAIL STATUS CHECK")
print("=" * 80)

# Get all video folders that have thumbnails
existing_thumbs = set()
if VIDEOS_DIR.exists():
    for folder in VIDEOS_DIR.iterdir():
        if folder.is_dir():
            thumb = folder / "thumbnail.jpg"
            if thumb.exists():
                existing_thumbs.add(folder.name)

print(f"\nThumbnails found in file system ({len(existing_thumbs)}):")
for proj in sorted(existing_thumbs):
    print(f"  - {proj}")

# Check database
print("\n" + "=" * 80)
print("VIDEOS IN DATABASE:")
print("=" * 80)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
    SELECT id, project_id, title, video_url, thumbnail_url, status
    FROM videos
    ORDER BY created_at DESC
''')
videos = cursor.fetchall()

missing_thumbs = []
for v in videos:
    vid_id, proj_id, title, video_url, thumb_url, status = v

    # Check if folder exists
    folder_exists = (VIDEOS_DIR / proj_id).exists() if proj_id else False
    has_thumb = proj_id in existing_thumbs if proj_id else False

    # Check if video_url is remote (Replicate URL)
    is_remote = video_url and ('replicate.delivery' in video_url or 'http' in video_url)

    print(f"\nTitle: {title}")
    print(f"  ID: {vid_id}")
    print(f"  Project ID: {proj_id}")
    print(f"  Folder exists: {folder_exists}")
    print(f"  Has thumbnail: {has_thumb}")
    print(f"  Video URL: {video_url[:60] if video_url else 'None'}...")
    print(f"  Is remote: {is_remote}")
    print(f"  Status: {status}")

    if not has_thumb:
        missing_thumbs.append({
            'id': vid_id,
            'project_id': proj_id,
            'title': title,
            'video_url': video_url,
            'is_remote': is_remote
        })

conn.close()

print("\n" + "=" * 80)
print(f"MISSING THUMBNAILS ({len(missing_thumbs)}):")
print("=" * 80)
for m in missing_thumbs:
    print(f"\n  {m['title']}")
    print(f"    Project ID: {m['project_id']}")
    print(f"    Video URL: {m['video_url'][:80] if m['video_url'] else 'None'}...")
    print(f"    Is remote: {m['is_remote']}")
