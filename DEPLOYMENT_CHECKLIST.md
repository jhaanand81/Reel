# ReelSenseAI Deployment Checklist

## Pre-Deployment Verification

### Manual Test Flow (prem22@gmail.com)

Run locally before deploying to Railway:

1. **Start Server**
   ```cmd
   cd backend
   python main.py
   ```

2. **Login Test**
   - Open http://localhost:5000/auth.html
   - Login as prem22@gmail.com
   - Verify: Login successful, redirects to home

3. **Credit Check**
   - Check credits displayed in UI
   - Should have >= 10 credits for 10-second video

4. **Generate 10-Second Video**
   - Topic: "Amazing facts about the universe"
   - Duration: 10 seconds
   - Quality: 1080p
   - Click "Generate"

5. **Verify Video Flow**
   - [ ] Stage 1: Script generated
   - [ ] Stage 2: Video clip generated
   - [ ] Stage 3: Voiceover generated
   - [ ] Stage 4: Final video composed
   - [ ] Video plays correctly

6. **Check My Videos**
   - Click "My Videos" tab
   - Verify: New video appears in list
   - Verify: Can play/download video

7. **Credit Deduction**
   - Check credits after video
   - Should be: Original - 10 = New balance

8. **Admin Dashboard**
   - Login as admin@contentsense.ai
   - Go to http://localhost:5000/admin.html
   - Videos tab:
     - [ ] Video shows in list
     - [ ] Thumbnail displays correctly
     - [ ] User email shown
     - [ ] Status shows "completed"

9. **Thumbnail Test**
   - Open: http://localhost:5000/api/v1/videos/{PROJECT_ID}/thumbnail.jpg
   - Should display video thumbnail image

---

## Railway Deployment

### Pre-Deployment Checklist
- [ ] All manual tests passed
- [ ] No errors in console
- [ ] Database persistence verified (data survives server restart)

### Deploy Commands
```bash
# Navigate to project
cd "C:\Users\Anand Jha\Documents\ReelSenseAI_v1.0_Windows_20251208\ReelSenseAI_Distribution\ReelSenseAI_Windows\ReelSenseAI"

# Push to GitHub (if using)
git add .
git commit -m "E2E tested: thumbnails, 3-day retention, admin dashboard"
git push origin main

# Railway auto-deploys on push
# OR manual deploy:
railway up
```

### Post-Deployment Verification
1. Open Railway URL
2. Login as test user
3. Create a short video (5 seconds to save credits)
4. Verify all features work

### Required Railway Environment Variables
```
VIDEO_PROVIDER=replicate
REPLICATE_API_TOKEN=your_token
GROQ_API_KEY=your_key
JWT_SECRET_KEY=your_secret
PASSWORD_SALT=your_salt
DATABASE_PATH=/app/backend/data/reelsense.db
ADMIN_EMAIL=admin@contentsense.ai
ADMIN_PASSWORD=your_admin_password
```

### Railway Volume Setup
1. Dashboard → Your Service → Ctrl+K → "volume"
2. Mount Path: `/app/backend/data`
3. Size: 1GB minimum

---

## Feature Summary (This Release)

### New Features
1. **Thumbnail Generation**
   - Auto-generated on video completion
   - 480px JPEG, ~20-50KB each
   - Displayed in Admin Dashboard

2. **3-Day Video Retention**
   - Videos auto-expire after 3 days
   - User "My Videos" - videos removed
   - Admin Dashboard - records kept with thumbnails

3. **Credit System**
   - 1 credit = 1 second of video
   - Default: 100 credits for new users
   - Deducted on video completion

4. **Admin Dashboard Enhancements**
   - Shows all videos (including expired)
   - Displays thumbnails instead of video players
   - "EXPIRED" overlay for deleted videos
   - User email and status visible

---

## Troubleshooting

### Database Reset on Deploy
- Ensure Railway volume is mounted at `/app/backend/data`
- Add env var: `DATABASE_PATH=/app/backend/data/reelsense.db`

### Thumbnail Not Showing
- Check FFmpeg is installed in container
- Video must exist to generate thumbnail
- Placeholder shown if generation fails

### Credits Not Deducting
- Check user is logged in
- Verify video completed successfully
- Check database for credit_transactions table
