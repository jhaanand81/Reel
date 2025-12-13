# ReelSense AI - Deployment Guide

## Quick Deploy (One-Click)

Double-click `push_to_live.bat` to deploy your changes!

---

## Deployment Methods

### Method 1: Git-Based Deployment (Recommended)

Best for: GitHub/GitLab with auto-deploy webhooks

#### Setup:

1. **Initialize Git repository (if not done):**
   ```bash
   cd ReelSenseAI
   git init
   git remote add origin https://github.com/yourusername/reelsense.git
   ```

2. **Configure deploy_config.json:**
   ```json
   {
       "deployment_method": "git",
       "git": {
           "remote": "origin",
           "branch": "main",
           "auto_push": true
       }
   }
   ```

3. **Deploy:**
   ```bash
   python deploy.py --deploy -m "Fixed login bug"
   ```
   Or double-click `push_to_live.bat`

#### Auto-Deploy on Server:

Set up a webhook on your server that pulls changes when you push:

```bash
# On your server, create /var/www/reelsense/pull_updates.sh
#!/bin/bash
cd /var/www/reelsense
git pull origin main
pip install -r requirements.txt
sudo systemctl restart reelsense
```

---

### Method 2: SSH/SFTP Deployment

Best for: Direct server access via SSH

#### Setup:

1. **Install paramiko:**
   ```bash
   pip install paramiko
   ```

2. **Configure SSH in deploy_config.json:**
   ```json
   {
       "deployment_method": "ssh",
       "ssh": {
           "host": "your-server.com",
           "port": 22,
           "username": "deploy",
           "key_file": "C:/Users/YourName/.ssh/id_rsa",
           "remote_path": "/var/www/reelsense"
       }
   }
   ```

3. **Generate SSH key (if needed):**
   ```bash
   ssh-keygen -t rsa -b 4096
   # Copy public key to server
   ssh-copy-id deploy@your-server.com
   ```

4. **Deploy:**
   ```bash
   python deploy.py --deploy --method ssh
   ```

---

### Method 3: FTP/FTPS Deployment

Best for: Shared hosting with FTP access

#### Setup:

1. **Configure FTP in deploy_config.json:**
   ```json
   {
       "deployment_method": "ftp",
       "ftp": {
           "host": "ftp.your-host.com",
           "port": 21,
           "username": "your-ftp-user",
           "password": "",
           "remote_path": "/public_html/reelsense",
           "use_tls": true
       }
   }
   ```

2. **Set FTP password (environment variable):**
   ```bash
   set FTP_PASSWORD=your-ftp-password
   ```

3. **Deploy:**
   ```bash
   python deploy.py --deploy --method ftp
   ```

---

### Method 4: Local/Network Sync

Best for: Same machine or network drive deployment

#### Setup:

1. **Configure local path in deploy_config.json:**
   ```json
   {
       "deployment_method": "local",
       "local": {
           "target_path": "D:/LiveServer/ReelSenseAI"
       }
   }
   ```

2. **Deploy:**
   ```bash
   python deploy.py --deploy --method local
   ```

---

## Cloudflare Tunnel (Expose Local to Internet)

For testing your local development with external access:

### Setup Cloudflare Tunnel:

1. **Install cloudflared:**
   Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/

2. **Login to Cloudflare:**
   ```bash
   cloudflared tunnel login
   ```

3. **Create tunnel:**
   ```bash
   cloudflared tunnel create reelsense
   ```

4. **Configure tunnel (config.yml):**
   ```yaml
   tunnel: YOUR_TUNNEL_ID
   credentials-file: C:\Users\YourName\.cloudflared\YOUR_TUNNEL_ID.json

   ingress:
     - hostname: reelsense.yourdomain.com
       service: http://localhost:5000
     - service: http_status:404
   ```

5. **Run tunnel:**
   ```bash
   cloudflared tunnel run reelsense
   ```

Now your local server is accessible at `https://reelsense.yourdomain.com`!

---

## Claude Code Integration

### Quick Commands for Claude Code:

When using Claude Code, you can say:

- **"Push to live"** → Runs `push_to_live.bat`
- **"Deploy changes"** → Runs deployment script
- **"Rollback"** → Reverts to previous version

### Workflow:

1. **Fix issue locally** (Claude Code helps you edit files)
2. **Test locally** (run server, check fix works)
3. **Deploy** (say "push to live" or run `push_to_live.bat`)

---

## Command Reference

```bash
# Initialize configuration
python deploy.py --init

# Deploy (uses default method from config)
python deploy.py --deploy

# Deploy with specific method
python deploy.py --deploy --method git
python deploy.py --deploy --method ssh
python deploy.py --deploy --method ftp
python deploy.py --deploy --method local

# Deploy with custom message
python deploy.py --deploy -m "Fixed bug in login"

# Create release package only (no deploy)
python deploy.py --package

# Rollback to previous release
python deploy.py --rollback

# Rollback to specific release
python deploy.py --rollback release_20251211_143000
```

---

## Troubleshooting

### Git Push Fails

```bash
# Check remote configuration
git remote -v

# Force push (use carefully!)
git push -f origin main
```

### SSH Connection Fails

```bash
# Test SSH connection
ssh -i ~/.ssh/id_rsa deploy@your-server.com

# Check key permissions
chmod 600 ~/.ssh/id_rsa
```

### FTP Upload Fails

- Check firewall allows FTP ports (21, passive ports)
- Try disabling TLS if having issues: `"use_tls": false`
- Verify credentials are correct

---

## Production Server Setup (Linux)

### 1. Install Requirements

```bash
sudo apt update
sudo apt install python3 python3-pip nginx
```

### 2. Setup Application

```bash
# Create directory
sudo mkdir -p /var/www/reelsense
sudo chown $USER:$USER /var/www/reelsense

# Clone or upload code
cd /var/www/reelsense
# Upload your release package here

# Install dependencies
pip3 install -r requirements.txt

# Create .env file
cp backend/.env.template backend/.env
nano backend/.env  # Edit with production values
```

### 3. Create Systemd Service

```bash
sudo nano /etc/systemd/system/reelsense.service
```

```ini
[Unit]
Description=ReelSense AI Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/reelsense/backend
ExecStart=/usr/bin/python3 main.py
Restart=always
Environment=FLASK_ENV=production

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable reelsense
sudo systemctl start reelsense
```

### 4. Configure Nginx

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /var/www/reelsense/frontend;
    }
}
```

---

## Security Notes

- Never commit `.env` files with real credentials
- Use environment variables for sensitive data
- Keep `service_account.json` secure
- Rotate SSH keys periodically
- Use HTTPS in production
