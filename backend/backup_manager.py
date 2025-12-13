"""
ReelSense AI - Automated Backup Manager
Handles daily backups to Google Drive with encryption and versioning
"""

import os
import sys
import json
import sqlite3
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import zipfile
import base64

# Optional imports for Google Drive
try:
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# Optional imports for encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupManager:
    """Manages automated backups with encryption and Google Drive upload"""

    def __init__(self, config_path: Optional[str] = None):
        self.base_path = Path(__file__).parent.parent
        self.backend_path = self.base_path / "backend"
        self.backup_path = self.base_path / "backups"
        self.config = self._load_config(config_path)

        # Create backup directory
        self.backup_path.mkdir(exist_ok=True)

        # Initialize encryption key if available
        self.cipher = None
        if ENCRYPTION_AVAILABLE and self.config.get('encryption_enabled', True):
            self._init_encryption()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load backup configuration"""
        default_config = {
            'encryption_enabled': True,
            'backup_database': True,
            'backup_config': True,
            'backup_videos': False,  # Videos can be large, optional
            'backup_logs': True,
            'max_local_backups': 7,  # Keep 7 days locally
            'google_drive_enabled': False,
            'google_drive_folder_id': None,
            'service_account_file': None,
            'notification_email': None,
        }

        config_file = Path(config_path) if config_path else self.backend_path / "backup_config.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        return default_config

    def _init_encryption(self):
        """Initialize encryption using a key derived from environment"""
        try:
            # Use a combination of machine-specific info and salt
            password = os.getenv('BACKUP_ENCRYPTION_KEY', 'reelsense-backup-key')
            salt = b'reelsense_backup_salt_v1'

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self.cipher = Fernet(key)
            logger.info("Encryption initialized successfully")
        except Exception as e:
            logger.warning(f"Encryption initialization failed: {e}")
            self.cipher = None

    def _encrypt_file(self, file_path: Path) -> Optional[Path]:
        """Encrypt a file and return path to encrypted version"""
        if not self.cipher:
            return file_path

        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            encrypted_data = self.cipher.encrypt(data)
            encrypted_path = file_path.with_suffix(file_path.suffix + '.enc')

            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)

            return encrypted_path
        except Exception as e:
            logger.error(f"Encryption failed for {file_path}: {e}")
            return file_path

    def _decrypt_file(self, encrypted_path: Path, output_path: Path) -> bool:
        """Decrypt an encrypted file"""
        if not self.cipher:
            shutil.copy(encrypted_path, output_path)
            return True

        try:
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)

            with open(output_path, 'wb') as f:
                f.write(decrypted_data)

            return True
        except Exception as e:
            logger.error(f"Decryption failed for {encrypted_path}: {e}")
            return False

    def backup_database(self) -> Optional[Path]:
        """Create a backup of the SQLite database"""
        db_path = self.backend_path / "data" / "reelsense.db"

        if not db_path.exists():
            logger.warning(f"Database not found at {db_path}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"database_{timestamp}.db"
        backup_file = self.backup_path / backup_name

        try:
            # Use SQLite's backup API for safe copy
            source = sqlite3.connect(db_path)
            dest = sqlite3.connect(backup_file)
            source.backup(dest)
            source.close()
            dest.close()

            logger.info(f"Database backed up to {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None

    def backup_config_files(self) -> Optional[Path]:
        """Backup configuration files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_backup = self.backup_path / f"config_{timestamp}.zip"

        config_files = [
            self.backend_path / ".env",
            self.backend_path / "backup_config.json",
            self.base_path / "frontend" / "js" / "config.js",
        ]

        try:
            with zipfile.ZipFile(config_backup, 'w', zipfile.ZIP_DEFLATED) as zf:
                for config_file in config_files:
                    if config_file.exists():
                        zf.write(config_file, config_file.name)
                        logger.info(f"Added {config_file.name} to config backup")

            logger.info(f"Config files backed up to {config_backup}")
            return config_backup
        except Exception as e:
            logger.error(f"Config backup failed: {e}")
            return None

    def backup_videos(self) -> Optional[Path]:
        """Backup generated videos (optional, can be large)"""
        videos_path = self.backend_path / "outputs" / "videos"

        if not videos_path.exists():
            logger.info("No videos directory found")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        videos_backup = self.backup_path / f"videos_{timestamp}.zip"

        try:
            with zipfile.ZipFile(videos_backup, 'w', zipfile.ZIP_DEFLATED) as zf:
                for video_file in videos_path.rglob("*.mp4"):
                    arcname = video_file.relative_to(videos_path)
                    zf.write(video_file, arcname)

            logger.info(f"Videos backed up to {videos_backup}")
            return videos_backup
        except Exception as e:
            logger.error(f"Videos backup failed: {e}")
            return None

    def backup_logs(self) -> Optional[Path]:
        """Backup log files"""
        logs_path = self.backend_path / "logs"

        if not logs_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_backup = self.backup_path / f"logs_{timestamp}.zip"

        try:
            with zipfile.ZipFile(logs_backup, 'w', zipfile.ZIP_DEFLATED) as zf:
                for log_file in logs_path.glob("*.log"):
                    zf.write(log_file, log_file.name)

            logger.info(f"Logs backed up to {logs_backup}")
            return logs_backup
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            return None

    def create_full_backup(self) -> Optional[Path]:
        """Create a complete backup package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_backup = self.backup_path / f"reelsense_backup_{timestamp}.zip"

        backup_files = []

        # Backup database
        if self.config.get('backup_database', True):
            db_backup = self.backup_database()
            if db_backup:
                backup_files.append(db_backup)

        # Backup config
        if self.config.get('backup_config', True):
            config_backup = self.backup_config_files()
            if config_backup:
                backup_files.append(config_backup)

        # Backup videos (optional)
        if self.config.get('backup_videos', False):
            videos_backup = self.backup_videos()
            if videos_backup:
                backup_files.append(videos_backup)

        # Backup logs
        if self.config.get('backup_logs', True):
            logs_backup = self.backup_logs()
            if logs_backup:
                backup_files.append(logs_backup)

        if not backup_files:
            logger.error("No files to backup")
            return None

        try:
            # Create final backup package
            with zipfile.ZipFile(full_backup, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add metadata
                metadata = {
                    'created_at': timestamp,
                    'version': '1.0',
                    'files': [f.name for f in backup_files],
                    'encrypted': self.config.get('encryption_enabled', False) and self.cipher is not None
                }
                zf.writestr('backup_metadata.json', json.dumps(metadata, indent=2))

                # Add all backup files
                for backup_file in backup_files:
                    if self.config.get('encryption_enabled', True) and self.cipher:
                        encrypted_file = self._encrypt_file(backup_file)
                        zf.write(encrypted_file, encrypted_file.name)
                        if encrypted_file != backup_file:
                            encrypted_file.unlink()  # Remove temp encrypted file
                    else:
                        zf.write(backup_file, backup_file.name)

                    # Clean up individual backup files
                    backup_file.unlink()

            logger.info(f"Full backup created: {full_backup}")
            return full_backup

        except Exception as e:
            logger.error(f"Full backup creation failed: {e}")
            return None

    def upload_to_google_drive(self, file_path: Path) -> bool:
        """Upload backup to Google Drive"""
        if not GOOGLE_DRIVE_AVAILABLE:
            logger.warning("Google Drive API not available. Install: pip install google-api-python-client google-auth")
            return False

        if not self.config.get('google_drive_enabled', False):
            logger.info("Google Drive upload disabled")
            return False

        service_account_file = self.config.get('service_account_file')
        folder_id = self.config.get('google_drive_folder_id')

        if not service_account_file or not folder_id:
            logger.warning("Google Drive not configured. Set service_account_file and google_drive_folder_id in backup_config.json")
            return False

        try:
            # Authenticate
            credentials = ServiceAccountCredentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )

            service = build('drive', 'v3', credentials=credentials)

            # Upload file
            file_metadata = {
                'name': file_path.name,
                'parents': [folder_id]
            }

            media = MediaFileUpload(
                str(file_path),
                mimetype='application/zip',
                resumable=True
            )

            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()

            logger.info(f"Uploaded to Google Drive: {file.get('name')} (ID: {file.get('id')})")
            return True

        except Exception as e:
            logger.error(f"Google Drive upload failed: {e}")
            return False

    def cleanup_old_backups(self):
        """Remove old local backups beyond retention period"""
        max_backups = self.config.get('max_local_backups', 7)

        # Get all backup files sorted by modification time
        backup_files = sorted(
            self.backup_path.glob("reelsense_backup_*.zip"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # Remove excess backups
        for old_backup in backup_files[max_backups:]:
            try:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")

    def run_backup(self) -> Dict:
        """Run the complete backup process"""
        logger.info("=" * 50)
        logger.info("Starting ReelSense AI Backup")
        logger.info("=" * 50)

        result = {
            'success': False,
            'backup_file': None,
            'google_drive_uploaded': False,
            'errors': []
        }

        try:
            # Create backup
            backup_file = self.create_full_backup()

            if backup_file:
                result['backup_file'] = str(backup_file)
                result['success'] = True

                # Upload to Google Drive
                if self.config.get('google_drive_enabled', False):
                    result['google_drive_uploaded'] = self.upload_to_google_drive(backup_file)

                # Cleanup old backups
                self.cleanup_old_backups()

                logger.info("Backup completed successfully!")
            else:
                result['errors'].append("Failed to create backup")

        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Backup failed: {e}")

        logger.info("=" * 50)
        return result

    def restore_backup(self, backup_file: Path, restore_path: Optional[Path] = None) -> bool:
        """Restore from a backup file"""
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        restore_path = restore_path or self.backup_path / "restore_temp"
        restore_path.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(backup_file, 'r') as zf:
                zf.extractall(restore_path)

            # Decrypt files if needed
            for enc_file in restore_path.glob("*.enc"):
                output_file = enc_file.with_suffix('')
                if self._decrypt_file(enc_file, output_file):
                    enc_file.unlink()

            logger.info(f"Backup restored to {restore_path}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False


def create_default_config():
    """Create a default backup configuration file"""
    config = {
        "encryption_enabled": True,
        "backup_database": True,
        "backup_config": True,
        "backup_videos": False,
        "backup_logs": True,
        "max_local_backups": 7,
        "google_drive_enabled": False,
        "google_drive_folder_id": "YOUR_GOOGLE_DRIVE_FOLDER_ID",
        "service_account_file": "service_account.json",
        "notification_email": None
    }

    config_path = Path(__file__).parent / "backup_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Default config created at: {config_path}")
    return config_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReelSense AI Backup Manager')
    parser.add_argument('--init', action='store_true', help='Create default configuration')
    parser.add_argument('--backup', action='store_true', help='Run backup')
    parser.add_argument('--restore', type=str, help='Restore from backup file')

    args = parser.parse_args()

    if args.init:
        create_default_config()
    elif args.backup:
        manager = BackupManager()
        result = manager.run_backup()
        print(json.dumps(result, indent=2))
    elif args.restore:
        manager = BackupManager()
        manager.restore_backup(Path(args.restore))
    else:
        # Default: run backup
        manager = BackupManager()
        result = manager.run_backup()
        print(json.dumps(result, indent=2))
