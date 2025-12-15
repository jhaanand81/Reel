"""
Quick test for 10-second video generation (2 clips)
Tests that both Scene 1 and Scene 2 are generated correctly.
"""

import requests
import time
import json

BASE_URL = 'http://localhost:5000'
API = f'{BASE_URL}/api/v1'

# Test user
EMAIL = 'prem22@gmail.com'
PASSWORD = 'anand@123'

import sys
if len(sys.argv) > 1:
    PASSWORD = sys.argv[1]

def main():
    print("=" * 60)
    print("10-Second Video Test (2 Clips)")
    print("=" * 60)

    # 1. Login
    print("\n[1/4] Logging in as", EMAIL)
    resp = requests.post(f'{API}/auth/login', json={
        'email': EMAIL,
        'password': PASSWORD
    })

    if resp.status_code != 200:
        print(f"Login failed: {resp.status_code}")
        print(resp.text)
        return

    data = resp.json()
    if data.get('status') != 'success':
        print(f"Login failed: {data.get('error')}")
        return

    token = data['data']['access_token']
    print(f"✓ Login successful")

    headers = {'Authorization': f'Bearer {token}'}

    # 2. Check credits
    print("\n[2/4] Checking credits...")
    resp = requests.get(f'{API}/auth/user/credits', headers=headers)
    credit_data = resp.json()
    print(f"  Raw response: {credit_data}")

    # Try different response formats
    credits = 0
    if credit_data.get('data'):
        credits = credit_data['data'].get('balance', credit_data['data'].get('credits', 0))
    elif credit_data.get('balance'):
        credits = credit_data['balance']
    elif credit_data.get('credits'):
        credits = credit_data['credits']

    print(f"✓ Credits: {credits}")

    if credits < 10:
        print("Not enough credits for 10-second video!")
        print("Continuing anyway to test the flow...")
        # Don't return - continue to test

    # 3. Generate script
    print("\n[3/4] Generating script...")
    resp = requests.post(f'{API}/generate-script', json={
        'topic': 'Amazing facts about Mount Fuji in Japan',
        'duration': 10,
        'style': 'educational'
    }, headers=headers, timeout=60)

    if resp.status_code != 200:
        print(f"Script generation failed: {resp.status_code}")
        # Use fallback script
        script = "Mount Fuji stands at 3,776 meters, making it Japan's tallest peak. Every year, over 300,000 people climb this sacred mountain. Its perfect symmetrical cone shape has inspired artists for centuries."
    else:
        script = resp.json().get('data', {}).get('script', '')

    print(f"✓ Script: {script[:100]}...")

    # 4. Generate 10-second video
    print("\n[4/4] Starting 10-second video generation...")
    print("      (This should create 2 clips: Scene 1 + Scene 2)")

    # Generate a project ID
    import uuid
    project_id = f"test_{uuid.uuid4().hex[:12]}"
    print(f"  Project ID: {project_id}")

    resp = requests.post(f'{API}/generate-video', json={
        'script': script,
        'projectId': project_id,
        'duration': 10,
        'quality': 'standard',
        'reelType': 'educational',
        'aspectRatio': '9:16',
        'resolution': '1080p'
    }, headers=headers, timeout=60)

    print(f"\nResponse status: {resp.status_code}")
    result = resp.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if result.get('status') == 'success':
        data = result.get('data', {})
        print("\n" + "=" * 60)
        print("VIDEO GENERATION STARTED")
        print("=" * 60)
        print(f"Project ID: {data.get('projectId')}")
        print(f"Job ID: {data.get('jobId')}")
        print(f"Clips Needed: {data.get('clipsNeeded')}")
        print(f"Clips Requested: {data.get('clipsRequested')}")
        print(f"Target Duration: {data.get('targetDuration')}s")
        print(f"Requested Duration: {data.get('requestedDuration')}s")

        if data.get('partial'):
            print(f"\n⚠️  WARNING: {data.get('warning')}")
        else:
            print(f"\n✓ All clips started successfully!")

        # Poll for status
        print("\nPolling for completion...")
        project_id = data.get('projectId')
        job_id = data.get('jobId')

        for i in range(60):  # Max 5 minutes
            time.sleep(5)
            status_resp = requests.get(f'{API}/status/{job_id}?projectId={project_id}', headers=headers)
            status = status_resp.json().get('data', {})

            progress = status.get('progress', 0)
            current_status = status.get('status', 'unknown')
            clips_completed = status.get('completedClips', 0)
            total_clips = status.get('totalClips', 2)

            print(f"  [{i*5}s] Status: {current_status}, Progress: {progress}%, Clips: {clips_completed}/{total_clips}")

            if current_status == 'completed':
                print(f"\n✓ VIDEO COMPLETED!")
                print(f"  URL: {status.get('videoUrl')}")
                break
            elif current_status == 'failed':
                print(f"\n✗ VIDEO FAILED: {status.get('error')}")
                break
    else:
        print(f"\n✗ Failed to start video: {result.get('error')}")

if __name__ == '__main__':
    main()
