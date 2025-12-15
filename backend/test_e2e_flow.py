"""
End-to-End Flow Test for ReelSenseAI
=====================================
Tests the complete video generation flow for user prem22@gmail.com:
1. User login
2. Credit check (needs 10 credits for 10-second video)
3. Script generation
4. Video generation (10 seconds, 1080p)
5. Video composition with voiceover
6. Thumbnail generation
7. Video appears in "My Videos"
8. Credit deduction verification
9. Admin dashboard visibility

Run: python test_e2e_flow.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# Configuration
BASE_URL = os.getenv('TEST_BASE_URL', 'http://localhost:5000')
API_VERSION = 'v1'
TEST_USER_EMAIL = 'prem22@gmail.com'
TEST_USER_PASSWORD = 'test123'  # Update if different
VIDEO_DURATION = 10  # seconds
VIDEO_QUALITY = '1080p'

# Test results
results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_test(name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {name}" + (f" - {message}" if message else ""))
    if passed:
        results['passed'].append(name)
    else:
        results['failed'].append((name, message))

def log_warning(message):
    print(f"\033[93m[WARN]\033[0m {message}")
    results['warnings'].append(message)

def log_info(message):
    print(f"\033[94m[INFO]\033[0m {message}")

def api_call(method, endpoint, data=None, token=None, timeout=300):
    """Make API call with authentication"""
    url = f"{BASE_URL}/api/{API_VERSION}/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        if method == 'GET':
            resp = requests.get(url, headers=headers, timeout=timeout)
        elif method == 'POST':
            resp = requests.post(url, json=data, headers=headers, timeout=timeout)
        elif method == 'DELETE':
            resp = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unknown method: {method}")

        return resp.status_code, resp.json() if resp.text else {}
    except requests.exceptions.Timeout:
        return 408, {'error': 'Request timeout'}
    except requests.exceptions.ConnectionError:
        return 0, {'error': 'Connection failed - is server running?'}
    except json.JSONDecodeError:
        return resp.status_code, {'raw': resp.text}
    except Exception as e:
        return 0, {'error': str(e)}

def test_server_health():
    """Test 1: Check server is running"""
    log_info("Testing server health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/{API_VERSION}/health", timeout=10)
        passed = resp.status_code == 200
        log_test("Server Health Check", passed, f"Status: {resp.status_code}")
        return passed
    except Exception as e:
        log_test("Server Health Check", False, str(e))
        return False

def test_user_login():
    """Test 2: Login as test user"""
    log_info(f"Logging in as {TEST_USER_EMAIL}...")

    status, data = api_call('POST', 'auth/login', {
        'email': TEST_USER_EMAIL,
        'password': TEST_USER_PASSWORD
    })

    if status == 200 and data.get('status') == 'success':
        token = data.get('data', {}).get('access_token')
        if token:
            log_test("User Login", True, f"Token received")
            return token

    # Try to register if login fails
    log_warning(f"Login failed, trying to check if user exists...")
    log_test("User Login", False, f"Status: {status}, Error: {data.get('error', 'Unknown')}")
    return None

def test_get_credits(token):
    """Test 3: Check user credits"""
    log_info("Checking user credits...")

    status, data = api_call('GET', 'auth/credits', token=token)

    if status == 200:
        credits = data.get('data', {}).get('balance', 0)
        required = VIDEO_DURATION  # 1 credit per second
        passed = credits >= required
        log_test("Credit Check", passed,
                f"Balance: {credits} credits (need {required} for {VIDEO_DURATION}s video)")
        return credits

    log_test("Credit Check", False, f"Status: {status}")
    return 0

def test_generate_script(token):
    """Test 4: Generate AI script"""
    log_info("Generating AI script...")

    status, data = api_call('POST', 'generate-script', {
        'topic': 'Amazing facts about the universe and space exploration',
        'duration': VIDEO_DURATION,
        'style': 'engaging',
        'tone': 'educational'
    }, token=token, timeout=60)

    if status == 200 and data.get('status') == 'success':
        script = data.get('data', {}).get('script', '')
        word_count = len(script.split())
        log_test("Script Generation", True, f"Generated {word_count} words")
        return script

    log_test("Script Generation", False, f"Status: {status}, Error: {data.get('error', 'Unknown')}")
    return None

def test_generate_video(token, script):
    """Test 5: Generate video (10 seconds, 1080p)"""
    log_info(f"Starting video generation ({VIDEO_DURATION}s, {VIDEO_QUALITY})...")

    status, data = api_call('POST', 'generate', {
        'script': script,
        'duration': VIDEO_DURATION,
        'quality': VIDEO_QUALITY,
        'reel_type': 'educational',
        'aspect_ratio': '9:16'  # Portrait for reels
    }, token=token, timeout=30)

    if status in [200, 202] and data.get('status') in ['success', 'processing']:
        project_id = data.get('data', {}).get('project_id') or data.get('data', {}).get('jobId')
        if project_id:
            log_test("Video Generation Started", True, f"Project ID: {project_id}")
            return project_id

    log_test("Video Generation Started", False, f"Status: {status}, Error: {data.get('error', 'Unknown')}")
    return None

def test_poll_video_status(token, project_id, max_wait=600):
    """Test 6: Poll for video completion"""
    log_info(f"Polling video status (max wait: {max_wait}s)...")

    start_time = time.time()
    last_status = None

    while time.time() - start_time < max_wait:
        status, data = api_call('GET', f'status/{project_id}', token=token)

        if status == 200:
            video_status = data.get('data', {}).get('status', data.get('status'))

            if video_status != last_status:
                log_info(f"  Status: {video_status}")
                last_status = video_status

            if video_status == 'completed':
                video_url = data.get('data', {}).get('videoUrl') or data.get('data', {}).get('video_url')
                elapsed = int(time.time() - start_time)
                log_test("Video Generation Complete", True, f"Completed in {elapsed}s")
                return video_url

            elif video_status == 'failed':
                error = data.get('data', {}).get('error', 'Unknown error')
                log_test("Video Generation Complete", False, f"Failed: {error}")
                return None

        time.sleep(5)  # Poll every 5 seconds

    log_test("Video Generation Complete", False, f"Timeout after {max_wait}s")
    return None

def test_thumbnail_exists(project_id):
    """Test 7: Check thumbnail was generated"""
    log_info("Checking thumbnail generation...")

    thumbnail_url = f"{BASE_URL}/api/{API_VERSION}/videos/{project_id}/thumbnail.jpg"

    try:
        resp = requests.get(thumbnail_url, timeout=10)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('image/'):
            size_kb = len(resp.content) / 1024
            log_test("Thumbnail Generation", True, f"Size: {size_kb:.1f}KB")
            return True
        else:
            log_test("Thumbnail Generation", False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        log_test("Thumbnail Generation", False, str(e))
        return False

def test_my_videos(token, project_id):
    """Test 8: Check video appears in My Videos"""
    log_info("Checking My Videos list...")

    status, data = api_call('GET', 'my-videos', token=token)

    if status == 200:
        videos = data.get('data', {}).get('videos', [])
        found = any(v.get('id') == project_id for v in videos)
        log_test("Video in My Videos", found,
                f"Found {len(videos)} videos" + (" (including new one)" if found else " (new video NOT found)"))
        return found

    log_test("Video in My Videos", False, f"Status: {status}")
    return False

def test_credits_deducted(token, original_credits):
    """Test 9: Verify credits were deducted"""
    log_info("Verifying credit deduction...")

    status, data = api_call('GET', 'auth/credits', token=token)

    if status == 200:
        new_credits = data.get('data', {}).get('balance', 0)
        expected_deduction = VIDEO_DURATION  # 1 credit per second
        actual_deduction = original_credits - new_credits

        passed = actual_deduction == expected_deduction
        log_test("Credit Deduction", passed,
                f"Deducted: {actual_deduction} (expected: {expected_deduction}), New balance: {new_credits}")
        return passed

    log_test("Credit Deduction", False, f"Status: {status}")
    return False

def test_admin_visibility(admin_token, project_id):
    """Test 10: Check video visible in admin dashboard"""
    log_info("Checking admin dashboard visibility...")

    status, data = api_call('GET', 'admin/videos', token=admin_token)

    if status == 200:
        videos = data.get('data', {}).get('videos', [])
        found = any(v.get('id') == project_id for v in videos)
        log_test("Admin Dashboard Visibility", found,
                f"Found in admin dashboard with {len(videos)} total videos")
        return found

    log_test("Admin Dashboard Visibility", False, f"Status: {status}")
    return False

def test_video_download(token, project_id):
    """Test 11: Check video can be downloaded"""
    log_info("Testing video download...")

    download_url = f"{BASE_URL}/api/{API_VERSION}/videos/{project_id}/download?format=mp4"

    try:
        resp = requests.get(download_url, headers={'Authorization': f'Bearer {token}'},
                          stream=True, timeout=30)

        if resp.status_code == 200:
            content_type = resp.headers.get('Content-Type', '')
            content_length = int(resp.headers.get('Content-Length', 0))
            size_mb = content_length / (1024 * 1024)

            passed = 'video' in content_type or 'octet-stream' in content_type
            log_test("Video Download", passed, f"Size: {size_mb:.2f}MB, Type: {content_type}")
            return passed
        else:
            log_test("Video Download", False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        log_test("Video Download", False, str(e))
        return False

def print_summary():
    """Print test summary"""
    total = len(results['passed']) + len(results['failed'])

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\033[92mPassed: {len(results['passed'])}/{total}\033[0m")
    print(f"\033[91mFailed: {len(results['failed'])}/{total}\033[0m")

    if results['warnings']:
        print(f"\033[93mWarnings: {len(results['warnings'])}\033[0m")

    if results['failed']:
        print("\nFailed Tests:")
        for name, msg in results['failed']:
            print(f"  - {name}: {msg}")

    print("="*60)

    return len(results['failed']) == 0

def main():
    print("\n" + "="*60)
    print("ReelSenseAI End-to-End Flow Test")
    print(f"User: {TEST_USER_EMAIL}")
    print(f"Video: {VIDEO_DURATION}s @ {VIDEO_QUALITY}")
    print(f"Server: {BASE_URL}")
    print("="*60 + "\n")

    # Test 1: Server health
    if not test_server_health():
        print("\n\033[91mServer not running! Start with: python main.py\033[0m")
        return False

    # Test 2: User login
    token = test_user_login()
    if not token:
        print("\n\033[91mLogin failed! Check user credentials.\033[0m")
        print(f"Expected user: {TEST_USER_EMAIL}")
        return False

    # Test 3: Credit check
    original_credits = test_get_credits(token)
    if original_credits < VIDEO_DURATION:
        log_warning(f"Not enough credits ({original_credits} < {VIDEO_DURATION}). Add credits first.")
        return False

    # Test 4: Generate script
    script = test_generate_script(token)
    if not script:
        log_warning("Script generation failed. Check Groq API key.")
        # Use fallback script
        script = "The universe is vast and mysterious. Every second, stars are born and die across billions of galaxies. Our own sun, a medium-sized star, will continue to shine for another five billion years. Space exploration has revealed wonders beyond imagination."
        log_info(f"Using fallback script: {script[:50]}...")

    # Test 5: Generate video
    project_id = test_generate_video(token, script)
    if not project_id:
        print("\n\033[91mVideo generation failed to start!\033[0m")
        return False

    # Test 6: Poll for completion
    video_url = test_poll_video_status(token, project_id)
    if not video_url:
        print("\n\033[91mVideo generation failed or timed out!\033[0m")
        return False

    # Test 7: Thumbnail
    test_thumbnail_exists(project_id)

    # Test 8: My Videos
    test_my_videos(token, project_id)

    # Test 9: Credit deduction
    test_credits_deducted(token, original_credits)

    # Test 10: Video download
    test_video_download(token, project_id)

    # Test 11: Admin visibility (optional - requires admin login)
    # Skipped unless admin credentials provided
    log_info("Skipping admin test (run separately with admin credentials)")

    # Print summary
    all_passed = print_summary()

    if all_passed:
        print("\n\033[92m" + "="*60)
        print("ALL TESTS PASSED!")
        print("Ready for Railway deployment.")
        print("="*60 + "\033[0m\n")
    else:
        print("\n\033[91m" + "="*60)
        print("SOME TESTS FAILED!")
        print("Fix issues before deploying to Railway.")
        print("="*60 + "\033[0m\n")

    return all_passed

if __name__ == '__main__':
    # Allow overriding test user password via environment
    if len(sys.argv) > 1:
        TEST_USER_PASSWORD = sys.argv[1]

    success = main()
    sys.exit(0 if success else 1)
