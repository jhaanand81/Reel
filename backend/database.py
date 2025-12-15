"""
SQLite Database Module for Reel Sense
Production-ready persistent storage with WAL mode for concurrent access

RAILWAY DEPLOYMENT:
- Requires Railway Volume mounted at /app/backend/data
- Configure in Railway Dashboard: Settings → Volumes → Add Volume
- Mount path: /app/backend/data
- Size: 1GB minimum recommended
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading
import logging
import shutil

logger = logging.getLogger(__name__)

# Database path - Railway volume MUST be mounted at /app/backend/data
# Use ABSOLUTE path based on this file's location for 100% reliability
_BACKEND_DIR = Path(__file__).parent.resolve()
_DEFAULT_DB_PATH = _BACKEND_DIR / 'data' / 'reelsense.db'
DB_PATH = Path(os.getenv('DATABASE_PATH', str(_DEFAULT_DB_PATH)))
API_KEY_PREFIX = 'rsk_'

# Log database path on module load for debugging
print(f"[DATABASE] Path configured: {DB_PATH}")
print(f"[DATABASE] Backend dir: {_BACKEND_DIR}")
print(f"[DATABASE] Path exists: {DB_PATH.exists()}")
print(f"[DATABASE] Parent exists: {DB_PATH.parent.exists()}")

# Thread-local storage for connections
_local = threading.local()
_db_initialized = False

def get_db_connection() -> sqlite3.Connection:
    """Get thread-local database connection with WAL mode for concurrency"""
    if not hasattr(_local, 'connection') or _local.connection is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.connection = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            timeout=30.0  # 30 second timeout for busy database
        )
        _local.connection.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrent read/write performance
        # WAL mode is persistent - only needs to be set once per database file
        _local.connection.execute('PRAGMA journal_mode=WAL')
        _local.connection.execute('PRAGMA synchronous=NORMAL')  # Faster, still safe
        _local.connection.execute('PRAGMA cache_size=-64000')  # 64MB cache
        _local.connection.execute('PRAGMA busy_timeout=30000')  # 30 second busy timeout

        logger.info(f"[DB] Connected to {DB_PATH} (WAL mode enabled)")
    return _local.connection


def check_database_health() -> Dict[str, Any]:
    """Check database health and return status"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if database is accessible
        cursor.execute("SELECT 1")

        # Get database info
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]

        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]

        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]

        db_size_mb = (page_count * page_size) / (1024 * 1024)

        # Count records
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM video_jobs")
        video_count = cursor.fetchone()[0]

        return {
            'healthy': True,
            'path': str(DB_PATH),
            'exists': DB_PATH.exists(),
            'journal_mode': journal_mode,
            'size_mb': round(db_size_mb, 2),
            'user_count': user_count,
            'video_count': video_count
        }
    except Exception as e:
        logger.error(f"[DB] Health check failed: {e}")
        return {
            'healthy': False,
            'error': str(e),
            'path': str(DB_PATH)
        }


def backup_database(backup_path: str = None) -> Optional[str]:
    """Create a backup of the database using SQLite's online backup API"""
    try:
        if backup_path is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = str(DB_PATH.parent / f"reelsense_backup_{timestamp}.db")

        source = get_db_connection()
        dest = sqlite3.connect(backup_path)

        # Use SQLite's backup API (safe for concurrent access)
        source.backup(dest)
        dest.close()

        logger.info(f"[DB] Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"[DB] Backup failed: {e}")
        return None

@contextmanager
def get_db():
    """Context manager for database operations"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e

def init_database():
    """Initialize database tables"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                api_key TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        ''')

        # Sessions table (for token tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_jti TEXT UNIQUE NOT NULL,
                token_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                revoked INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Video jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                provider TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                prompt TEXT,
                duration INTEGER,
                video_url TEXT,
                thumbnail_url TEXT,
                file_path TEXT,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                expires_at TIMESTAMP,
                deleted_at TIMESTAMP,
                error TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Request logs (for analytics)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                duration_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # User credits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_credits (
                user_id TEXT PRIMARY KEY,
                balance INTEGER DEFAULT 100,
                total_used INTEGER DEFAULT 0,
                total_given INTEGER DEFAULT 100,
                total_purchased INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Credit transactions table (audit log)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                admin_id TEXT,
                project_id TEXT,
                balance_after INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Audit logs table (for tracking admin actions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id TEXT NOT NULL,
                admin_email TEXT,
                action TEXT NOT NULL,
                target_type TEXT,
                target_id TEXT,
                details TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users(id)
            )
        ''')

        # System settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        ''')

        # Notifications table (announcements)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT DEFAULT 'info',
                target_role TEXT DEFAULT 'all',
                is_active INTEGER DEFAULT 1,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        ''')

        # User notification reads (track who has seen what)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notification_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                read_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (notification_id) REFERENCES notifications(id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(notification_id, user_id)
            )
        ''')

        # Support tickets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_email TEXT,
                subject TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                priority TEXT DEFAULT 'normal',
                category TEXT DEFAULT 'general',
                assigned_to TEXT,
                admin_notes TEXT,
                resolved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (assigned_to) REFERENCES users(id)
            )
        ''')

        # Ticket replies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticket_replies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES support_tickets(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_jti ON sessions(token_jti)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_jobs_user ON video_jobs(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_logs_user ON request_logs(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_credit_transactions_user ON credit_transactions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_admin ON audit_logs(admin_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_active ON notifications(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_support_tickets_user ON support_tickets(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_support_tickets_status ON support_tickets(status)')

        # Insert default system settings if not exist
        default_settings = [
            ('default_free_credits', '100', 'Free credits for new users'),
            ('max_video_duration', '60', 'Maximum video duration in seconds'),
            ('api_rate_limit', '100', 'API requests per hour per user'),
            ('maintenance_mode', 'false', 'Enable maintenance mode'),
            ('registration_enabled', 'true', 'Allow new user registration'),
            ('video_generation_enabled', 'true', 'Enable video generation'),
            ('video_retention_days', '3', 'Video retention period in days before auto-delete'),
        ]
        for key, value, desc in default_settings:
            cursor.execute('''
                INSERT OR IGNORE INTO system_settings (key, value, description)
                VALUES (?, ?, ?)
            ''', (key, value, desc))

        logger.info("Database initialized successfully")

# ==============================================================================
# USER OPERATIONS
# ==============================================================================

def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = os.getenv('PASSWORD_SALT', 'reelsense-production-salt')
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def generate_user_id(email: str) -> str:
    """Generate deterministic user ID from email"""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

def generate_api_key() -> str:
    """Generate unique API key"""
    return f"{API_KEY_PREFIX}{secrets.token_hex(32)}"

def create_user(email: str, password: str, name: str, role: str = 'user') -> Optional[Dict]:
    """Create a new user"""
    user_id = generate_user_id(email)
    api_key = generate_api_key()
    password_hash = hash_password(password)

    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (id, email, name, password_hash, role, api_key)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, email.lower(), name, password_hash, role, api_key))

            return {
                'id': user_id,
                'email': email.lower(),
                'name': name,
                'role': role,
                'api_key': api_key,
                'created_at': datetime.utcnow().isoformat()
            }
        except sqlite3.IntegrityError:
            return None  # User already exists

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticate user with email and password"""
    password_hash = hash_password(password)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, name, role, api_key, created_at
            FROM users
            WHERE email = ? AND password_hash = ? AND is_active = 1
        ''', (email.lower(), password_hash))

        row = cursor.fetchone()
        if row:
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?',
                         (datetime.utcnow(), row['id']))
            return dict(row)
        return None

def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, name, role, api_key, created_at
            FROM users WHERE id = ? AND is_active = 1
        ''', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_user_by_api_key(api_key: str) -> Optional[Dict]:
    """Get user by API key"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, name, role, api_key, created_at
            FROM users WHERE api_key = ? AND is_active = 1
        ''', (api_key,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, name, role, api_key, created_at
            FROM users WHERE email = ? AND is_active = 1
        ''', (email.lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_all_users() -> List[Dict]:
    """Get all users"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, name, role, api_key, created_at, last_login
            FROM users WHERE is_active = 1
        ''')
        return [dict(row) for row in cursor.fetchall()]

def count_users() -> int:
    """Count total users"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        return cursor.fetchone()[0]

def update_user_api_key(user_id: str, new_api_key: str) -> bool:
    """Update a user's API key"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET api_key = ? WHERE id = ? AND is_active = 1
        ''', (new_api_key, user_id))
        return cursor.rowcount > 0

def update_user_password(user_id: str, new_password: str) -> bool:
    """Update a user's password"""
    new_password_hash = hash_password(new_password)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET password_hash = ? WHERE id = ? AND is_active = 1
        ''', (new_password_hash, user_id))
        return cursor.rowcount > 0

# ==============================================================================
# SESSION OPERATIONS
# ==============================================================================

def create_session(user_id: str, token_jti: str, token_type: str, expires_at: datetime) -> bool:
    """Create a new session"""
    session_id = secrets.token_hex(16)

    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO sessions (id, user_id, token_jti, token_type, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, user_id, token_jti, token_type, expires_at))
            return True
        except sqlite3.IntegrityError:
            return False

def revoke_session(token_jti: str) -> bool:
    """Revoke a session by JTI"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE sessions SET revoked = 1 WHERE token_jti = ?', (token_jti,))
        return cursor.rowcount > 0

def is_session_revoked(token_jti: str) -> bool:
    """Check if session is revoked. Returns False if session not found (token is valid until explicitly revoked)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT revoked FROM sessions WHERE token_jti = ?', (token_jti,))
        row = cursor.fetchone()
        # Return False if no session record exists - token is valid by default
        # Only return True if session exists AND is explicitly revoked
        return row['revoked'] == 1 if row else False

# ==============================================================================
# VIDEO JOB OPERATIONS
# ==============================================================================

def create_video_job(job_id: str, user_id: str, provider: str, prompt: str, duration: int) -> bool:
    """Create a video job record"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO video_jobs (id, user_id, provider, prompt, duration)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_id, user_id, provider, prompt, duration))
        return True

def update_video_job(job_id: str, status: str, video_url: str = None, error: str = None):
    """Update video job status"""
    with get_db() as conn:
        cursor = conn.cursor()
        if status == 'completed':
            cursor.execute('''
                UPDATE video_jobs
                SET status = ?, video_url = ?, completed_at = ?
                WHERE id = ?
            ''', (status, video_url, datetime.utcnow(), job_id))
        elif status == 'failed':
            cursor.execute('''
                UPDATE video_jobs SET status = ?, error = ? WHERE id = ?
            ''', (status, error, job_id))
        else:
            cursor.execute('UPDATE video_jobs SET status = ? WHERE id = ?', (status, job_id))

def get_user_jobs(user_id: str, limit: int = 10) -> List[Dict]:
    """Get user's video jobs"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM video_jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]

def get_user_videos(user_id: str, include_expired: bool = False) -> List[Dict]:
    """Get user's videos for My Videos page"""
    with get_db() as conn:
        cursor = conn.cursor()
        if include_expired:
            cursor.execute('''
                SELECT id, user_id, provider, status, prompt, duration,
                       video_url, thumbnail_url, file_path, file_size,
                       created_at, completed_at, expires_at, deleted_at, error
                FROM video_jobs
                WHERE user_id = ? AND deleted_at IS NULL AND status = 'completed'
                ORDER BY created_at DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT id, user_id, provider, status, prompt, duration,
                       video_url, thumbnail_url, file_path, file_size,
                       created_at, completed_at, expires_at, deleted_at, error
                FROM video_jobs
                WHERE user_id = ? AND deleted_at IS NULL AND status = 'completed'
                      AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY created_at DESC
            ''', (user_id,))
        return [dict(row) for row in cursor.fetchall()]

def soft_delete_video(video_id: str, user_id: str) -> bool:
    """Soft delete a video (mark as deleted)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE video_jobs
            SET deleted_at = datetime('now')
            WHERE id = ? AND user_id = ?
        ''', (video_id, user_id))
        return cursor.rowcount > 0

def get_all_videos_admin(limit: int = 100, include_deleted: bool = True) -> List[Dict]:
    """Get all videos for admin dashboard (includes deleted videos)"""
    with get_db() as conn:
        cursor = conn.cursor()
        if include_deleted:
            # Admin sees ALL videos including deleted (for audit/tracking)
            cursor.execute('''
                SELECT v.id, v.user_id, v.provider, v.status, v.prompt, v.duration,
                       v.video_url, v.thumbnail_url, v.file_path, v.file_size,
                       v.created_at, v.completed_at, v.expires_at, v.deleted_at, v.error,
                       u.email as user_email, u.name as user_name
                FROM video_jobs v
                LEFT JOIN users u ON v.user_id = u.id
                ORDER BY v.created_at DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT v.id, v.user_id, v.provider, v.status, v.prompt, v.duration,
                       v.video_url, v.thumbnail_url, v.file_path, v.file_size,
                       v.created_at, v.completed_at, v.expires_at, v.deleted_at, v.error,
                       u.email as user_email, u.name as user_name
                FROM video_jobs v
                LEFT JOIN users u ON v.user_id = u.id
                WHERE v.deleted_at IS NULL
                ORDER BY v.created_at DESC
                LIMIT ?
            ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

def update_video_thumbnail(video_id: str, thumbnail_url: str) -> bool:
    """Update thumbnail URL for a video"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE video_jobs SET thumbnail_url = ? WHERE id = ?
        ''', (thumbnail_url, video_id))
        return cursor.rowcount > 0

def get_video_job_info(job_id: str) -> Optional[Dict]:
    """Get video job info by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, user_id, provider, status, prompt, duration,
                   video_url, created_at, completed_at
            FROM video_jobs WHERE id = ?
        ''', (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def set_video_expiry(job_id: str, retention_days: int = 3):
    """Set video expiration date based on retention period"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE video_jobs
            SET expires_at = datetime(completed_at, '+' || ? || ' days')
            WHERE id = ? AND completed_at IS NOT NULL
        ''', (retention_days, job_id))

def cleanup_expired_videos() -> int:
    """Delete expired videos (soft delete in DB + delete actual files) and return count"""
    import shutil
    from pathlib import Path

    with get_db() as conn:
        cursor = conn.cursor()

        # First, get the list of expired videos to delete their files
        cursor.execute('''
            SELECT id, video_url, file_path FROM video_jobs
            WHERE expires_at IS NOT NULL
              AND expires_at < datetime('now')
              AND deleted_at IS NULL
        ''')
        expired_videos = cursor.fetchall()

        # Delete actual video files (but keep thumbnail for admin reference)
        for video in expired_videos:
            video_id = video['id'] if isinstance(video, dict) else video[0]
            try:
                # Video directory path
                video_dir = Path(__file__).parent / 'outputs' / 'videos' / video_id
                if video_dir.exists():
                    # Delete video files but keep thumbnail.jpg for admin
                    for file in video_dir.iterdir():
                        if file.name != 'thumbnail.jpg':
                            file.unlink(missing_ok=True)
                    logger.info(f"[CLEANUP] Deleted video files for {video_id}")
            except Exception as e:
                logger.warning(f"[CLEANUP] Failed to delete files for {video_id}: {e}")

        # Soft delete in database (keeps record for admin dashboard)
        cursor.execute('''
            UPDATE video_jobs
            SET deleted_at = datetime('now')
            WHERE expires_at IS NOT NULL
              AND expires_at < datetime('now')
              AND deleted_at IS NULL
        ''')
        return cursor.rowcount

def get_retention_setting() -> int:
    """Get video retention days from system settings"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT value FROM system_settings WHERE key = 'video_retention_days'
        ''')
        row = cursor.fetchone()
        return int(row[0]) if row else 3  # Default 3 days

def set_retention_setting(days: int) -> bool:
    """Set video retention days"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO system_settings (key, value, description, updated_at)
            VALUES ('video_retention_days', ?, 'Video retention period in days', datetime('now'))
        ''', (str(days),))
        return True

# ==============================================================================
# REQUEST LOGGING
# ==============================================================================

def log_request(user_id: str, endpoint: str, method: str, status_code: int, duration_ms: float):
    """Log API request for analytics"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO request_logs (user_id, endpoint, method, status_code, duration_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, endpoint, method, status_code, duration_ms))

def get_request_stats(minutes: int = 60) -> Dict:
    """Get request statistics"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                COUNT(*) as total_requests,
                AVG(duration_ms) as avg_duration,
                MAX(duration_ms) as max_duration,
                MIN(duration_ms) as min_duration,
                COUNT(DISTINCT user_id) as unique_users
            FROM request_logs
            WHERE created_at > datetime('now', ? || ' minutes')
        ''', (-minutes,))
        return dict(cursor.fetchone())

# ==============================================================================
# CREDIT OPERATIONS
# ==============================================================================

DEFAULT_CREDITS = 100  # Free credits for new users

def calculate_video_credits(duration: int, quality: str = 'standard') -> int:
    """
    Calculate credits required for video generation.
    POC Model: 1 credit = 1 second of video

    Args:
        duration: Video duration in seconds
        quality: 'standard' or 'high' (for future use)

    Returns:
        Number of credits required
    """
    # Simple 1:1 ratio for POC
    # 5 sec = 5 credits, 15 sec = 15 credits, etc.
    credits = max(5, duration)  # Minimum 5 credits

    # Future: Add quality multiplier if needed
    # if quality == 'high':
    #     credits = int(credits * 1.5)

    return credits

def check_user_credits(user_id: str, required_credits: int) -> Dict:
    """
    Check if user has enough credits for video generation.

    Returns:
        Dict with 'has_enough', 'balance', 'required', 'shortfall'
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT balance FROM user_credits WHERE user_id = ?
        ''', (user_id,))
        row = cursor.fetchone()

        balance = row['balance'] if row else DEFAULT_CREDITS
        has_enough = balance >= required_credits

        return {
            'has_enough': has_enough,
            'balance': balance,
            'required': required_credits,
            'shortfall': max(0, required_credits - balance)
        }

def init_user_credits(user_id: str, initial_credits: int = DEFAULT_CREDITS) -> bool:
    """Initialize credits for a new user"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO user_credits (user_id, balance, total_given)
                VALUES (?, ?, ?)
            ''', (user_id, initial_credits, initial_credits))

            if cursor.rowcount > 0:
                # Log the initial credit grant
                cursor.execute('''
                    INSERT INTO credit_transactions (user_id, amount, type, description, balance_after)
                    VALUES (?, ?, 'signup_bonus', 'Welcome bonus credits', ?)
                ''', (user_id, initial_credits, initial_credits))
            return True
        except Exception as e:
            logger.error(f"Failed to init credits: {e}")
            return False

def get_user_credits(user_id: str) -> Dict:
    """Get user's credit balance and stats"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT balance, total_used, total_given, total_purchased, updated_at
            FROM user_credits WHERE user_id = ?
        ''', (user_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        else:
            # Initialize if not exists
            init_user_credits(user_id)
            return {
                'balance': DEFAULT_CREDITS,
                'total_used': 0,
                'total_given': DEFAULT_CREDITS,
                'total_purchased': 0
            }

def add_credits(user_id: str, amount: int, reason: str, admin_id: str = None) -> Dict:
    """Add credits to user (admin gift or purchase)"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Ensure user has credits record
        cursor.execute('SELECT balance FROM user_credits WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if not row:
            init_user_credits(user_id, 0)
            current_balance = 0
        else:
            current_balance = row['balance']

        new_balance = current_balance + amount

        # Update balance
        cursor.execute('''
            UPDATE user_credits
            SET balance = ?, total_given = total_given + ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (new_balance, amount, user_id))

        # Log transaction
        tx_type = 'admin_gift' if admin_id else 'purchase'
        cursor.execute('''
            INSERT INTO credit_transactions (user_id, amount, type, description, admin_id, balance_after)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, amount, tx_type, reason, admin_id, new_balance))

        return {'balance': new_balance, 'added': amount}

def deduct_credits(user_id: str, amount: int, reason: str, project_id: str = None) -> Dict:
    """Deduct credits for video generation"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get current balance
        cursor.execute('SELECT balance FROM user_credits WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if not row:
            init_user_credits(user_id)
            current_balance = DEFAULT_CREDITS
        else:
            current_balance = row['balance']

        if current_balance < amount:
            return {'error': 'Insufficient credits', 'balance': current_balance, 'required': amount}

        new_balance = current_balance - amount

        # Update balance
        cursor.execute('''
            UPDATE user_credits
            SET balance = ?, total_used = total_used + ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (new_balance, amount, user_id))

        # Log transaction
        cursor.execute('''
            INSERT INTO credit_transactions (user_id, amount, type, description, project_id, balance_after)
            VALUES (?, ?, 'usage', ?, ?, ?)
        ''', (user_id, -amount, reason, project_id, new_balance))

        return {'balance': new_balance, 'deducted': amount}

def get_credit_history(user_id: str, limit: int = 50) -> List[Dict]:
    """Get user's credit transaction history"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, amount, type, description, project_id, balance_after, created_at
            FROM credit_transactions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]

def get_all_users_with_credits() -> List[Dict]:
    """Get all users with their credit balances (for admin)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                u.id, u.email, u.name, u.role, u.created_at, u.last_login,
                COALESCE(c.balance, 100) as credits_balance,
                COALESCE(c.total_used, 0) as credits_used,
                COALESCE(c.total_given, 100) as credits_given,
                (SELECT COUNT(*) FROM video_jobs WHERE user_id = u.id) as video_count
            FROM users u
            LEFT JOIN user_credits c ON u.id = c.user_id
            WHERE u.is_active = 1
            ORDER BY u.created_at DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

def update_user_role(user_id: str, new_role: str) -> bool:
    """Update user's role (admin only)"""
    valid_roles = ['user', 'editor', 'viewer', 'admin', 'api']
    if new_role not in valid_roles:
        return False

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
        return cursor.rowcount > 0

def delete_user(user_id: str) -> bool:
    """Soft delete user (set inactive)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = 0 WHERE id = ?', (user_id,))
        return cursor.rowcount > 0

def get_all_videos(limit: int = 100) -> List[Dict]:
    """Get all videos with user info (for admin)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                v.id, v.user_id, v.provider, v.status, v.prompt,
                v.duration, v.video_url, v.created_at, v.completed_at, v.error,
                u.email as user_email, u.name as user_name
            FROM video_jobs v
            LEFT JOIN users u ON v.user_id = u.id
            ORDER BY v.created_at DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

def delete_video(video_id: str) -> bool:
    """Delete a video job"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM video_jobs WHERE id = ?', (video_id,))
        return cursor.rowcount > 0

def get_admin_stats() -> Dict:
    """Get dashboard statistics for admin"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Total users
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        total_users = cursor.fetchone()[0]

        # Total videos
        cursor.execute('SELECT COUNT(*) FROM video_jobs')
        total_videos = cursor.fetchone()[0]

        # Completed videos
        cursor.execute("SELECT COUNT(*) FROM video_jobs WHERE status = 'completed'")
        completed_videos = cursor.fetchone()[0]

        # Failed videos
        cursor.execute("SELECT COUNT(*) FROM video_jobs WHERE status = 'failed'")
        failed_videos = cursor.fetchone()[0]

        # Active users (logged in last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM users
            WHERE is_active = 1 AND last_login > datetime('now', '-7 days')
        ''')
        active_users = cursor.fetchone()[0]

        # Total credits given
        cursor.execute('SELECT COALESCE(SUM(total_given), 0) FROM user_credits')
        total_credits_given = cursor.fetchone()[0]

        # Total credits used
        cursor.execute('SELECT COALESCE(SUM(total_used), 0) FROM user_credits')
        total_credits_used = cursor.fetchone()[0]

        # Videos today
        cursor.execute('''
            SELECT COUNT(*) FROM video_jobs
            WHERE created_at > datetime('now', '-1 day')
        ''')
        videos_today = cursor.fetchone()[0]

        # New users today
        cursor.execute('''
            SELECT COUNT(*) FROM users
            WHERE created_at > datetime('now', '-1 day')
        ''')
        new_users_today = cursor.fetchone()[0]

        return {
            'total_users': total_users,
            'active_users': active_users,
            'new_users_today': new_users_today,
            'total_videos': total_videos,
            'completed_videos': completed_videos,
            'failed_videos': failed_videos,
            'videos_today': videos_today,
            'success_rate': round(completed_videos / total_videos * 100, 1) if total_videos > 0 else 0,
            'total_credits_given': total_credits_given,
            'total_credits_used': total_credits_used
        }

# ==============================================================================
# ANALYTICS FUNCTIONS
# ==============================================================================

def get_analytics_data(days: int = 30) -> Dict:
    """Get analytics data for charts"""
    with get_db() as conn:
        cursor = conn.cursor()

        # User growth over time
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM users
            WHERE created_at > datetime('now', ? || ' days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', (f'-{days}',))
        user_growth = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Video generation over time
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM video_jobs
            WHERE created_at > datetime('now', ? || ' days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', (f'-{days}',))
        video_trend = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Credit usage over time
        cursor.execute('''
            SELECT DATE(created_at) as date,
                   SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as used,
                   SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as given
            FROM credit_transactions
            WHERE created_at > datetime('now', ? || ' days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', (f'-{days}',))
        credit_trend = [{'date': row[0], 'used': row[1], 'given': row[2]} for row in cursor.fetchall()]

        # User roles distribution
        cursor.execute('''
            SELECT role, COUNT(*) as count
            FROM users WHERE is_active = 1
            GROUP BY role
        ''')
        role_distribution = [{'role': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Video status distribution
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM video_jobs
            GROUP BY status
        ''')
        video_status = [{'status': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Top users by video count
        cursor.execute('''
            SELECT u.name, u.email, COUNT(v.id) as video_count
            FROM users u
            LEFT JOIN video_jobs v ON u.id = v.user_id
            WHERE u.is_active = 1
            GROUP BY u.id
            ORDER BY video_count DESC
            LIMIT 10
        ''')
        top_users = [{'name': row[0], 'email': row[1], 'videos': row[2]} for row in cursor.fetchall()]

        return {
            'user_growth': user_growth,
            'video_trend': video_trend,
            'credit_trend': credit_trend,
            'role_distribution': role_distribution,
            'video_status': video_status,
            'top_users': top_users
        }

# ==============================================================================
# AUDIT LOG FUNCTIONS
# ==============================================================================

def log_audit(admin_id: str, admin_email: str, action: str, target_type: str = None,
              target_id: str = None, details: str = None, ip_address: str = None) -> int:
    """Log an admin action"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (admin_id, admin_email, action, target_type, target_id, details, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (admin_id, admin_email, action, target_type, target_id, details, ip_address))
        return cursor.lastrowid

def get_audit_logs(limit: int = 100, offset: int = 0, admin_id: str = None, action: str = None) -> List[Dict]:
    """Get audit logs with optional filters"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = 'SELECT * FROM audit_logs WHERE 1=1'
        params = []

        if admin_id:
            query += ' AND admin_id = ?'
            params.append(admin_id)
        if action:
            query += ' AND action LIKE ?'
            params.append(f'%{action}%')

        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def get_audit_log_count(admin_id: str = None, action: str = None) -> int:
    """Get total audit log count"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = 'SELECT COUNT(*) FROM audit_logs WHERE 1=1'
        params = []

        if admin_id:
            query += ' AND admin_id = ?'
            params.append(admin_id)
        if action:
            query += ' AND action LIKE ?'
            params.append(f'%{action}%')

        cursor.execute(query, params)
        return cursor.fetchone()[0]

# ==============================================================================
# SYSTEM SETTINGS FUNCTIONS
# ==============================================================================

def get_all_settings() -> Dict[str, Any]:
    """Get all system settings"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT key, value, description, updated_at FROM system_settings')
        return {row[0]: {'value': row[1], 'description': row[2], 'updated_at': row[3]}
                for row in cursor.fetchall()}

def get_setting(key: str, default: str = None) -> str:
    """Get a specific setting value"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM system_settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row[0] if row else default

def update_setting(key: str, value: str, updated_by: str = None) -> bool:
    """Update a system setting"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE system_settings
            SET value = ?, updated_at = CURRENT_TIMESTAMP, updated_by = ?
            WHERE key = ?
        ''', (value, updated_by, key))
        return cursor.rowcount > 0

def create_setting(key: str, value: str, description: str = None) -> bool:
    """Create a new system setting"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO system_settings (key, value, description)
                VALUES (?, ?, ?)
            ''', (key, value, description))
            return True
        except sqlite3.IntegrityError:
            return False

# ==============================================================================
# NOTIFICATION FUNCTIONS
# ==============================================================================

def create_notification(title: str, message: str, type: str = 'info',
                       target_role: str = 'all', created_by: str = None,
                       expires_at: str = None) -> int:
    """Create a new notification/announcement"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO notifications (title, message, type, target_role, created_by, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, message, type, target_role, created_by, expires_at))
        return cursor.lastrowid

def get_notifications(include_inactive: bool = False, limit: int = 50) -> List[Dict]:
    """Get all notifications (for admin)"""
    with get_db() as conn:
        cursor = conn.cursor()
        if include_inactive:
            cursor.execute('''
                SELECT n.*, u.email as creator_email
                FROM notifications n
                LEFT JOIN users u ON n.created_by = u.id
                ORDER BY n.created_at DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT n.*, u.email as creator_email
                FROM notifications n
                LEFT JOIN users u ON n.created_by = u.id
                WHERE n.is_active = 1
                AND (n.expires_at IS NULL OR n.expires_at > datetime('now'))
                ORDER BY n.created_at DESC
                LIMIT ?
            ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

def get_user_notifications(user_id: str, user_role: str) -> List[Dict]:
    """Get notifications for a specific user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT n.*,
                   CASE WHEN nr.id IS NOT NULL THEN 1 ELSE 0 END as is_read
            FROM notifications n
            LEFT JOIN notification_reads nr ON n.id = nr.notification_id AND nr.user_id = ?
            WHERE n.is_active = 1
            AND (n.expires_at IS NULL OR n.expires_at > datetime('now'))
            AND (n.target_role = 'all' OR n.target_role = ?)
            ORDER BY n.created_at DESC
        ''', (user_id, user_role))
        return [dict(row) for row in cursor.fetchall()]

def mark_notification_read(notification_id: int, user_id: str) -> bool:
    """Mark a notification as read by user"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO notification_reads (notification_id, user_id)
                VALUES (?, ?)
            ''', (notification_id, user_id))
            return True
        except sqlite3.IntegrityError:
            return False  # Already marked as read

def update_notification(notification_id: int, is_active: bool) -> bool:
    """Activate/deactivate a notification"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE notifications SET is_active = ? WHERE id = ?',
                      (1 if is_active else 0, notification_id))
        return cursor.rowcount > 0

def delete_notification(notification_id: int) -> bool:
    """Delete a notification"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM notification_reads WHERE notification_id = ?', (notification_id,))
        cursor.execute('DELETE FROM notifications WHERE id = ?', (notification_id,))
        return cursor.rowcount > 0

# ==============================================================================
# SUPPORT TICKET FUNCTIONS
# ==============================================================================

def create_ticket(user_id: str, user_email: str, subject: str, message: str,
                 category: str = 'general', priority: str = 'normal') -> int:
    """Create a new support ticket"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO support_tickets (user_id, user_email, subject, message, category, priority)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, user_email, subject, message, category, priority))
        return cursor.lastrowid

def get_all_tickets(status: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get all support tickets (for admin)"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = '''
            SELECT t.*,
                   u.name as user_name,
                   a.email as assigned_email,
                   (SELECT COUNT(*) FROM ticket_replies WHERE ticket_id = t.id) as reply_count
            FROM support_tickets t
            LEFT JOIN users u ON t.user_id = u.id
            LEFT JOIN users a ON t.assigned_to = a.id
        '''
        params = []

        if status:
            query += ' WHERE t.status = ?'
            params.append(status)

        query += ' ORDER BY t.created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def get_user_tickets(user_id: str) -> List[Dict]:
    """Get tickets for a specific user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT t.*,
                   (SELECT COUNT(*) FROM ticket_replies WHERE ticket_id = t.id) as reply_count
            FROM support_tickets t
            WHERE t.user_id = ?
            ORDER BY t.created_at DESC
        ''', (user_id,))
        return [dict(row) for row in cursor.fetchall()]

def get_ticket_by_id(ticket_id: int) -> Optional[Dict]:
    """Get a specific ticket with replies"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT t.*, u.name as user_name, u.email as user_email_display
            FROM support_tickets t
            LEFT JOIN users u ON t.user_id = u.id
            WHERE t.id = ?
        ''', (ticket_id,))
        ticket = cursor.fetchone()
        if not ticket:
            return None

        ticket_dict = dict(ticket)

        # Get replies
        cursor.execute('''
            SELECT r.*, u.name as author_name, u.email as author_email
            FROM ticket_replies r
            LEFT JOIN users u ON r.user_id = u.id
            WHERE r.ticket_id = ?
            ORDER BY r.created_at ASC
        ''', (ticket_id,))
        ticket_dict['replies'] = [dict(row) for row in cursor.fetchall()]

        return ticket_dict

def update_ticket(ticket_id: int, status: str = None, priority: str = None,
                 assigned_to: str = None, admin_notes: str = None) -> bool:
    """Update ticket status/assignment"""
    with get_db() as conn:
        cursor = conn.cursor()
        updates = ['updated_at = CURRENT_TIMESTAMP']
        params = []

        if status:
            updates.append('status = ?')
            params.append(status)
            if status == 'resolved':
                updates.append('resolved_at = CURRENT_TIMESTAMP')
        if priority:
            updates.append('priority = ?')
            params.append(priority)
        if assigned_to is not None:
            updates.append('assigned_to = ?')
            params.append(assigned_to if assigned_to else None)
        if admin_notes is not None:
            updates.append('admin_notes = ?')
            params.append(admin_notes)

        params.append(ticket_id)
        cursor.execute(f'UPDATE support_tickets SET {", ".join(updates)} WHERE id = ?', params)
        return cursor.rowcount > 0

def add_ticket_reply(ticket_id: int, user_id: str, message: str, is_admin: bool = False) -> int:
    """Add a reply to a ticket"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ticket_replies (ticket_id, user_id, message, is_admin)
            VALUES (?, ?, ?, ?)
        ''', (ticket_id, user_id, message, 1 if is_admin else 0))

        # Update ticket timestamp
        cursor.execute('UPDATE support_tickets SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                      (ticket_id,))
        return cursor.lastrowid

def get_ticket_stats() -> Dict:
    """Get ticket statistics"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM support_tickets WHERE status = "open"')
        open_tickets = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM support_tickets WHERE status = "in_progress"')
        in_progress = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM support_tickets WHERE status = "resolved"')
        resolved = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM support_tickets WHERE priority = "urgent"')
        urgent = cursor.fetchone()[0]

        return {
            'open': open_tickets,
            'in_progress': in_progress,
            'resolved': resolved,
            'urgent': urgent,
            'total': open_tickets + in_progress + resolved
        }

# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_users_data() -> List[Dict]:
    """Export all users data for CSV"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                u.id, u.email, u.name, u.role, u.created_at, u.last_login, u.is_active,
                COALESCE(c.balance, 0) as credit_balance,
                COALESCE(c.total_used, 0) as credits_used,
                COALESCE(c.total_given, 0) as credits_given,
                (SELECT COUNT(*) FROM video_jobs WHERE user_id = u.id) as video_count
            FROM users u
            LEFT JOIN user_credits c ON u.id = c.user_id
            ORDER BY u.created_at DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

def export_transactions_data(days: int = 30) -> List[Dict]:
    """Export credit transactions for CSV"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                ct.id, ct.user_id, u.email as user_email, u.name as user_name,
                ct.amount, ct.type, ct.description, ct.balance_after, ct.created_at
            FROM credit_transactions ct
            LEFT JOIN users u ON ct.user_id = u.id
            WHERE ct.created_at > datetime('now', ? || ' days')
            ORDER BY ct.created_at DESC
        ''', (f'-{days}',))
        return [dict(row) for row in cursor.fetchall()]

def export_videos_data(days: int = 30) -> List[Dict]:
    """Export video jobs for CSV"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                v.id, v.user_id, u.email as user_email, u.name as user_name,
                v.provider, v.status, v.prompt, v.duration,
                v.created_at, v.completed_at, v.error
            FROM video_jobs v
            LEFT JOIN users u ON v.user_id = u.id
            WHERE v.created_at > datetime('now', ? || ' days')
            ORDER BY v.created_at DESC
        ''', (f'-{days}',))
        return [dict(row) for row in cursor.fetchall()]

# Initialize database on import
init_database()
