# ReelSense AI - Backup & Security Setup Guide

## Quick Start

### 1. Install Required Dependencies

```bash
pip install google-api-python-client google-auth cryptography
```

### 2. Run Manual Backup (Test)

```bash
cd backend
python backup_manager.py --backup
```

---

## Google Drive Automatic Backup Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., "ReelSense Backup")
3. Enable the **Google Drive API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"

### Step 2: Create Service Account

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Name it (e.g., "reelsense-backup")
4. Click "Create and Continue"
5. Skip the optional steps, click "Done"

### Step 3: Download Service Account Key

1. Click on your new service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Select "JSON" and click "Create"
5. Save the downloaded file as `service_account.json` in the `backend` folder

### Step 4: Create Google Drive Folder

1. Go to [Google Drive](https://drive.google.com/)
2. Create a new folder (e.g., "ReelSense Backups")
3. Right-click the folder > "Share"
4. Add the service account email (found in your JSON file, looks like: `xxx@xxx.iam.gserviceaccount.com`)
5. Give it "Editor" access
6. Copy the folder ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`

### Step 5: Update Configuration

Edit `backend/backup_config.json`:

```json
{
    "encryption_enabled": true,
    "backup_database": true,
    "backup_config": true,
    "backup_videos": false,
    "backup_logs": true,
    "max_local_backups": 7,
    "google_drive_enabled": true,
    "google_drive_folder_id": "YOUR_FOLDER_ID_HERE",
    "service_account_file": "service_account.json",
    "notification_email": "your@email.com"
}
```

### Step 6: Test Google Drive Upload

```bash
python backup_manager.py --backup
```

Check your Google Drive folder for the backup file!

---

## Windows Task Scheduler Setup (Daily Automatic Backups)

### Method 1: Using GUI

1. Open **Task Scheduler** (search in Start menu)
2. Click "Create Basic Task..."
3. Name: "ReelSense Daily Backup"
4. Trigger: Daily, set time (e.g., 2:00 AM)
5. Action: Start a program
6. Program: `C:\Users\Anand Jha\Documents\ReelSenseAI_v1.0_Windows_20251208\ReelSenseAI_Distribution\ReelSenseAI_Windows\ReelSenseAI\run_backup.bat`
7. Check "Open Properties dialog" > Finish
8. In Properties:
   - Check "Run whether user is logged on or not"
   - Check "Run with highest privileges"

### Method 2: Using Command Line

Run this in PowerShell (as Administrator):

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Users\Anand Jha\Documents\ReelSenseAI_v1.0_Windows_20251208\ReelSenseAI_Distribution\ReelSenseAI_Windows\ReelSenseAI\run_backup.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "ReelSense Daily Backup" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Daily backup of ReelSense AI database and config"
```

---

## Security Best Practices

### 1. Environment Variables

Set a custom encryption key (optional but recommended):

```bash
set BACKUP_ENCRYPTION_KEY=your-secure-encryption-key-here
```

Or add to your system environment variables permanently.

### 2. Firewall Rules

Ensure only necessary ports are open:
- Port 5000 (Flask backend) - only if needed externally
- Port 80/443 for web access

### 3. Database Security

The backup script automatically:
- Encrypts backups with AES-256 encryption
- Uses SQLite's backup API for safe database copies
- Maintains versioned backups

### 4. Access Control

- Keep `service_account.json` secure (don't commit to git)
- Use strong passwords for admin accounts
- Regularly rotate API keys

---

## Restore from Backup

### Restore Last Backup

```bash
cd backend
python backup_manager.py --restore "backups/reelsense_backup_20251211_020000.zip"
```

### Manual Restore

1. Extract the backup ZIP file
2. Decrypt files (if encrypted): Files ending in `.enc` need the same encryption key
3. Replace database: Copy `database_*.db` to `backend/data/reelsense.db`
4. Restore config: Extract config files to their original locations

---

## Troubleshooting

### Backup Fails

1. Check `backup.log` for errors
2. Verify Python path in `run_backup.bat`
3. Ensure write permissions to `backups` folder

### Google Drive Upload Fails

1. Verify service account has access to the folder
2. Check internet connection
3. Verify `google_drive_folder_id` is correct
4. Check API quota in Google Cloud Console

### Encryption Issues

If you lose your encryption key:
- Unencrypted backup metadata is still readable
- Database structure can be recreated
- Consider storing encryption key securely (e.g., password manager)

---

## Backup Contents

Each backup includes:

| File | Contents |
|------|----------|
| `database_*.db` | SQLite database (users, videos, credits, settings) |
| `config_*.zip` | .env, backup_config.json, config.js |
| `logs_*.zip` | Application logs |
| `videos_*.zip` | Generated videos (if enabled) |
| `backup_metadata.json` | Backup timestamp and file list |

---

## Support

For issues or questions:
- Check `backend/backup.log` for detailed error messages
- Review this guide for common solutions
- Contact support if issues persist
