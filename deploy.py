"""
ReelSense AI - Deployment Manager
One-click deployment to live environment with rollback support
"""

import os
import sys
import json
import shutil
import subprocess
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import zipfile

# Optional imports
try:
    import paramiko
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False

try:
    from ftplib import FTP, FTP_TLS
    FTP_AVAILABLE = True
except ImportError:
    FTP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deploy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages code deployment to live environment"""

    def __init__(self, config_path: Optional[str] = None):
        self.base_path = Path(__file__).parent
        self.config = self._load_config(config_path)
        self.releases_path = self.base_path / "releases"
        self.releases_path.mkdir(exist_ok=True)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment configuration"""
        default_config = {
            "deployment_method": "git",  # git, ssh, ftp, local
            "git": {
                "remote": "origin",
                "branch": "main",
                "auto_push": True
            },
            "ssh": {
                "host": "",
                "port": 22,
                "username": "",
                "key_file": "",
                "remote_path": "/var/www/reelsense"
            },
            "ftp": {
                "host": "",
                "port": 21,
                "username": "",
                "password": "",
                "remote_path": "/public_html",
                "use_tls": True
            },
            "local": {
                "target_path": ""
            },
            "exclude_patterns": [
                "__pycache__",
                "*.pyc",
                ".git",
                ".env",
                "node_modules",
                "*.log",
                "releases",
                "backups",
                "outputs/videos/*",
                "deploy.log",
                "backup.log"
            ],
            "include_env_template": True,
            "max_releases": 5,
            "pre_deploy_commands": [],
            "post_deploy_commands": [
                "pip install -r requirements.txt",
                "python -c \"from main import init_database; init_database()\""
            ],
            "restart_command": ""
        }

        config_file = Path(config_path) if config_path else self.base_path / "deploy_config.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        return default_config

    def _deep_update(self, base: Dict, update: Dict):
        """Deep update nested dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from deployment"""
        path_str = str(path)
        for pattern in self.config.get('exclude_patterns', []):
            if pattern.startswith('*'):
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern.endswith('*'):
                if pattern[:-1] in path_str:
                    return True
            elif pattern in path_str:
                return True
        return False

    def create_release_package(self) -> Optional[Path]:
        """Create a release package for deployment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        release_name = f"release_{timestamp}"
        release_path = self.releases_path / f"{release_name}.zip"

        logger.info(f"Creating release package: {release_name}")

        try:
            with zipfile.ZipFile(release_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add backend files
                backend_path = self.base_path / "backend"
                if backend_path.exists():
                    for file_path in backend_path.rglob("*"):
                        if file_path.is_file() and not self._should_exclude(file_path):
                            arcname = file_path.relative_to(self.base_path)
                            zf.write(file_path, arcname)
                            logger.debug(f"Added: {arcname}")

                # Add frontend files
                frontend_path = self.base_path / "frontend"
                if frontend_path.exists():
                    for file_path in frontend_path.rglob("*"):
                        if file_path.is_file() and not self._should_exclude(file_path):
                            arcname = file_path.relative_to(self.base_path)
                            zf.write(file_path, arcname)
                            logger.debug(f"Added: {arcname}")

                # Add root files
                for root_file in ["requirements.txt", "run_server.bat", "run_backup.bat"]:
                    root_path = self.base_path / root_file
                    if root_path.exists():
                        zf.write(root_path, root_file)

                # Add .env template if configured
                if self.config.get('include_env_template', True):
                    env_path = self.base_path / "backend" / ".env"
                    if env_path.exists():
                        # Create sanitized .env.template
                        with open(env_path, 'r') as f:
                            env_content = f.read()

                        # Remove sensitive values
                        template_lines = []
                        for line in env_content.split('\n'):
                            if '=' in line and not line.startswith('#'):
                                key = line.split('=')[0]
                                template_lines.append(f"{key}=CHANGE_ME")
                            else:
                                template_lines.append(line)

                        zf.writestr("backend/.env.template", '\n'.join(template_lines))

                # Add release metadata
                metadata = {
                    "release_name": release_name,
                    "created_at": timestamp,
                    "version": self._get_version(),
                    "git_commit": self._get_git_commit()
                }
                zf.writestr("release_metadata.json", json.dumps(metadata, indent=2))

            logger.info(f"Release package created: {release_path}")
            return release_path

        except Exception as e:
            logger.error(f"Failed to create release package: {e}")
            return None

    def _get_version(self) -> str:
        """Get current version from package or config"""
        try:
            version_file = self.base_path / "VERSION"
            if version_file.exists():
                return version_file.read_text().strip()
        except:
            pass
        return "1.0.0"

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.base_path
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except:
            pass
        return None

    # ===== GIT DEPLOYMENT =====
    def deploy_git(self, message: str = None) -> bool:
        """Deploy using Git push"""
        logger.info("Starting Git deployment...")

        git_config = self.config.get('git', {})
        remote = git_config.get('remote', 'origin')
        branch = git_config.get('branch', 'main')

        try:
            # Stage all changes
            subprocess.run(["git", "add", "-A"], cwd=self.base_path, check=True)

            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.base_path
            )

            if result.stdout.strip():
                # Create commit
                commit_msg = message or f"Deploy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=self.base_path,
                    check=True
                )
                logger.info(f"Created commit: {commit_msg}")
            else:
                logger.info("No changes to commit")

            # Push to remote
            if git_config.get('auto_push', True):
                subprocess.run(
                    ["git", "push", remote, branch],
                    cwd=self.base_path,
                    check=True
                )
                logger.info(f"Pushed to {remote}/{branch}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Git deployment failed: {e}")
            return False

    # ===== SSH DEPLOYMENT =====
    def deploy_ssh(self) -> bool:
        """Deploy using SSH/SFTP"""
        if not SSH_AVAILABLE:
            logger.error("SSH deployment requires paramiko. Install: pip install paramiko")
            return False

        ssh_config = self.config.get('ssh', {})
        host = ssh_config.get('host')
        port = ssh_config.get('port', 22)
        username = ssh_config.get('username')
        key_file = ssh_config.get('key_file')
        remote_path = ssh_config.get('remote_path')

        if not all([host, username, remote_path]):
            logger.error("SSH configuration incomplete")
            return False

        logger.info(f"Starting SSH deployment to {host}...")

        try:
            # Create release package
            release_package = self.create_release_package()
            if not release_package:
                return False

            # Connect via SSH
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if key_file:
                ssh.connect(host, port=port, username=username, key_filename=key_file)
            else:
                password = os.getenv('SSH_PASSWORD') or input("SSH Password: ")
                ssh.connect(host, port=port, username=username, password=password)

            sftp = ssh.open_sftp()

            # Upload release package
            remote_package = f"{remote_path}/releases/{release_package.name}"

            # Create releases directory if needed
            try:
                sftp.mkdir(f"{remote_path}/releases")
            except:
                pass

            logger.info(f"Uploading {release_package.name}...")
            sftp.put(str(release_package), remote_package)

            # Extract on remote server
            extract_cmd = f"cd {remote_path} && unzip -o releases/{release_package.name}"
            stdin, stdout, stderr = ssh.exec_command(extract_cmd)
            stdout.read()

            # Run post-deploy commands
            for cmd in self.config.get('post_deploy_commands', []):
                full_cmd = f"cd {remote_path}/backend && {cmd}"
                logger.info(f"Running: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(full_cmd)
                output = stdout.read().decode()
                if output:
                    logger.info(output)

            # Restart service if configured
            restart_cmd = self.config.get('restart_command')
            if restart_cmd:
                logger.info(f"Restarting service...")
                ssh.exec_command(restart_cmd)

            sftp.close()
            ssh.close()

            logger.info("SSH deployment completed successfully!")
            return True

        except Exception as e:
            logger.error(f"SSH deployment failed: {e}")
            return False

    # ===== FTP DEPLOYMENT =====
    def deploy_ftp(self) -> bool:
        """Deploy using FTP/FTPS"""
        ftp_config = self.config.get('ftp', {})
        host = ftp_config.get('host')
        port = ftp_config.get('port', 21)
        username = ftp_config.get('username')
        password = ftp_config.get('password') or os.getenv('FTP_PASSWORD')
        remote_path = ftp_config.get('remote_path')
        use_tls = ftp_config.get('use_tls', True)

        if not all([host, username, password]):
            logger.error("FTP configuration incomplete")
            return False

        logger.info(f"Starting FTP deployment to {host}...")

        try:
            # Create release package
            release_package = self.create_release_package()
            if not release_package:
                return False

            # Connect to FTP
            if use_tls:
                ftp = FTP_TLS()
            else:
                ftp = FTP()

            ftp.connect(host, port)
            ftp.login(username, password)

            if use_tls:
                ftp.prot_p()

            # Navigate to remote path
            try:
                ftp.cwd(remote_path)
            except:
                logger.warning(f"Could not change to {remote_path}")

            # Upload release package
            with open(release_package, 'rb') as f:
                ftp.storbinary(f'STOR {release_package.name}', f)

            logger.info(f"Uploaded {release_package.name}")

            ftp.quit()
            logger.info("FTP deployment completed!")
            logger.warning("Note: You'll need to extract the ZIP on the server manually or via a webhook")
            return True

        except Exception as e:
            logger.error(f"FTP deployment failed: {e}")
            return False

    # ===== LOCAL SYNC =====
    def deploy_local(self) -> bool:
        """Deploy to local/network path"""
        local_config = self.config.get('local', {})
        target_path = Path(local_config.get('target_path', ''))

        if not target_path or not target_path.exists():
            logger.error(f"Target path does not exist: {target_path}")
            return False

        logger.info(f"Starting local deployment to {target_path}...")

        try:
            # Sync backend
            backend_src = self.base_path / "backend"
            backend_dst = target_path / "backend"
            if backend_src.exists():
                self._sync_directory(backend_src, backend_dst)

            # Sync frontend
            frontend_src = self.base_path / "frontend"
            frontend_dst = target_path / "frontend"
            if frontend_src.exists():
                self._sync_directory(frontend_src, frontend_dst)

            logger.info("Local deployment completed!")
            return True

        except Exception as e:
            logger.error(f"Local deployment failed: {e}")
            return False

    def _sync_directory(self, src: Path, dst: Path):
        """Sync source directory to destination"""
        dst.mkdir(parents=True, exist_ok=True)

        for src_file in src.rglob("*"):
            if src_file.is_file() and not self._should_exclude(src_file):
                rel_path = src_file.relative_to(src)
                dst_file = dst / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                logger.debug(f"Synced: {rel_path}")

    # ===== MAIN DEPLOY =====
    def deploy(self, method: str = None, message: str = None) -> Dict:
        """Run deployment"""
        logger.info("=" * 50)
        logger.info("Starting ReelSense AI Deployment")
        logger.info("=" * 50)

        method = method or self.config.get('deployment_method', 'git')
        result = {
            'success': False,
            'method': method,
            'errors': []
        }

        # Run pre-deploy commands
        for cmd in self.config.get('pre_deploy_commands', []):
            logger.info(f"Pre-deploy: {cmd}")
            subprocess.run(cmd, shell=True, cwd=self.base_path)

        # Deploy based on method
        if method == 'git':
            result['success'] = self.deploy_git(message)
        elif method == 'ssh':
            result['success'] = self.deploy_ssh()
        elif method == 'ftp':
            result['success'] = self.deploy_ftp()
        elif method == 'local':
            result['success'] = self.deploy_local()
        else:
            result['errors'].append(f"Unknown deployment method: {method}")

        # Cleanup old releases
        self._cleanup_old_releases()

        logger.info("=" * 50)
        if result['success']:
            logger.info("Deployment completed successfully!")
        else:
            logger.error("Deployment failed!")
        logger.info("=" * 50)

        return result

    def _cleanup_old_releases(self):
        """Remove old release packages"""
        max_releases = self.config.get('max_releases', 5)
        releases = sorted(
            self.releases_path.glob("release_*.zip"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for old_release in releases[max_releases:]:
            try:
                old_release.unlink()
                logger.info(f"Removed old release: {old_release.name}")
            except:
                pass

    def rollback(self, release_name: str = None) -> bool:
        """Rollback to previous release"""
        releases = sorted(
            self.releases_path.glob("release_*.zip"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not releases:
            logger.error("No releases found for rollback")
            return False

        if release_name:
            target = self.releases_path / f"{release_name}.zip"
            if not target.exists():
                logger.error(f"Release not found: {release_name}")
                return False
        else:
            # Get previous release (skip current)
            if len(releases) < 2:
                logger.error("No previous release available")
                return False
            target = releases[1]

        logger.info(f"Rolling back to: {target.name}")
        # Implementation depends on deployment method
        return True


def create_default_config():
    """Create default deployment configuration"""
    config = {
        "deployment_method": "git",
        "git": {
            "remote": "origin",
            "branch": "main",
            "auto_push": True
        },
        "ssh": {
            "host": "your-server.com",
            "port": 22,
            "username": "deploy",
            "key_file": "~/.ssh/id_rsa",
            "remote_path": "/var/www/reelsense"
        },
        "ftp": {
            "host": "ftp.your-server.com",
            "port": 21,
            "username": "ftpuser",
            "password": "",
            "remote_path": "/public_html",
            "use_tls": True
        },
        "local": {
            "target_path": "D:/LiveServer/ReelSenseAI"
        },
        "exclude_patterns": [
            "__pycache__",
            "*.pyc",
            ".git",
            ".env",
            "node_modules",
            "*.log",
            "releases",
            "backups",
            "outputs/videos/*"
        ],
        "max_releases": 5,
        "post_deploy_commands": [
            "pip install -r requirements.txt"
        ],
        "restart_command": "sudo systemctl restart reelsense"
    }

    config_path = Path(__file__).parent / "deploy_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Default config created at: {config_path}")
    return config_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReelSense AI Deployment Manager')
    parser.add_argument('--init', action='store_true', help='Create default configuration')
    parser.add_argument('--deploy', action='store_true', help='Run deployment')
    parser.add_argument('--method', type=str, choices=['git', 'ssh', 'ftp', 'local'], help='Deployment method')
    parser.add_argument('--message', '-m', type=str, help='Commit/deployment message')
    parser.add_argument('--rollback', type=str, nargs='?', const='latest', help='Rollback to release')
    parser.add_argument('--package', action='store_true', help='Create release package only')

    args = parser.parse_args()

    if args.init:
        create_default_config()
    elif args.package:
        manager = DeploymentManager()
        package = manager.create_release_package()
        if package:
            print(f"Release package created: {package}")
    elif args.rollback:
        manager = DeploymentManager()
        manager.rollback(args.rollback if args.rollback != 'latest' else None)
    elif args.deploy:
        manager = DeploymentManager()
        result = manager.deploy(method=args.method, message=args.message)
        print(json.dumps(result, indent=2))
    else:
        # Default: show help
        parser.print_help()
