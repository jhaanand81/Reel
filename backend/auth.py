"""
JWT Authentication Module for Reel Sense API
Production-ready authentication with refresh tokens
"""

import os
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Tuple
from flask import request, jsonify, g, current_app
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
JWT_ALGORITHM = 'HS256'
JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))  # 1 hour
JWT_REFRESH_TOKEN_EXPIRES = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRES', 604800))  # 7 days

# API Key settings
API_KEY_PREFIX = 'rsk_'  # Reel Sense Key prefix
API_KEY_LENGTH = 32

# Beta Access / Invite Code System
# Set BETA_MODE=true and BETA_INVITE_CODE to enable invite-only registration
BETA_MODE = os.getenv('BETA_MODE', 'true').lower() == 'true'
BETA_INVITE_CODE = os.getenv('BETA_INVITE_CODE', 'REELSENSE2025')  # Change this!
BETA_INVITE_CODES = set(filter(None, os.getenv('BETA_INVITE_CODES', '').split(',')))  # Multiple codes
if BETA_INVITE_CODE:
    BETA_INVITE_CODES.add(BETA_INVITE_CODE)

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class User:
    """User model for authentication"""
    id: str
    email: str
    name: str
    role: str = 'user'  # user, admin, api
    api_key: Optional[str] = None
    created_at: Optional[Any] = None  # Can be datetime or string from DB

    def to_dict(self) -> Dict[str, Any]:
        # Handle created_at - could be datetime object or string from SQLite
        created_at_str = None
        if self.created_at:
            if isinstance(self.created_at, str):
                created_at_str = self.created_at
            elif hasattr(self.created_at, 'isoformat'):
                created_at_str = self.created_at.isoformat()
            else:
                created_at_str = str(self.created_at)

        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'created_at': created_at_str
        }

@dataclass
class TokenPayload:
    """JWT Token payload"""
    user_id: str
    email: str
    role: str
    token_type: str  # 'access' or 'refresh'
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation

# ==============================================================================
# DATABASE-BACKED USER STORE (Production Ready)
# ==============================================================================

# Import database functions
try:
    from database import (
        create_user as db_create_user,
        authenticate_user as db_authenticate_user,
        get_user_by_id as db_get_user_by_id,
        get_user_by_email as db_get_user_by_email,
        get_user_by_api_key as db_get_user_by_api_key,
        create_session as db_create_session,
        revoke_session as db_revoke_session,
        is_session_revoked as db_is_session_revoked,
        init_database
    )
    DATABASE_AVAILABLE = True
    logger.info("[OK] Database module loaded for auth")
except ImportError:
    try:
        from backend.database import (
            create_user as db_create_user,
            authenticate_user as db_authenticate_user,
            get_user_by_id as db_get_user_by_id,
            get_user_by_email as db_get_user_by_email,
            get_user_by_api_key as db_get_user_by_api_key,
            create_session as db_create_session,
            revoke_session as db_revoke_session,
            is_session_revoked as db_is_session_revoked,
            init_database
        )
        DATABASE_AVAILABLE = True
        logger.info("[OK] Database module loaded for auth (backend prefix)")
    except ImportError as e:
        DATABASE_AVAILABLE = False
        logger.warning(f"[WARN] Database not available, using in-memory store: {e}")


class UserStore:
    """
    Production user store with SQLite database backend.
    Falls back to in-memory storage if database is not available.
    """

    def __init__(self):
        self._memory_users: Dict[str, Dict] = {}
        self._memory_api_keys: Dict[str, str] = {}
        self._revoked_tokens: set = set()

        # Initialize database if available
        if DATABASE_AVAILABLE:
            try:
                init_database()
                logger.info("[OK] Database initialized for auth")
            except Exception as e:
                logger.error(f"[ERROR] Database init failed: {e}")

        # Create default admin user
        self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        admin_email = os.getenv('ADMIN_EMAIL', 'admin@reelsense.ai')
        admin_password = os.getenv('ADMIN_PASSWORD', 'changeme123')

        if admin_email and admin_password:
            # Check if admin already exists
            existing = self.get_user_by_email(admin_email)
            if not existing:
                user = self.create_user(admin_email, admin_password, 'Admin', 'admin')
                if user:
                    logger.info(f"[OK] Default admin user created: {admin_email}")
                else:
                    logger.info(f"[INFO] Admin user already exists: {admin_email}")

    def _generate_user_id(self, email: str) -> str:
        """Generate deterministic user ID from email"""
        return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

    def _generate_api_key(self) -> str:
        """Generate API key with prefix"""
        return f"{API_KEY_PREFIX}{secrets.token_hex(API_KEY_LENGTH)}"

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = os.getenv('PASSWORD_SALT', 'reelsense-production-salt')
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

    def _dict_to_user(self, data: Dict) -> Optional[User]:
        """Convert dict to User object"""
        if not data:
            return None
        return User(
            id=data.get('id'),
            email=data.get('email'),
            name=data.get('name'),
            role=data.get('role', 'user'),
            api_key=data.get('api_key'),
            created_at=data.get('created_at')
        )

    def create_user(self, email: str, password: str, name: str, role: str = 'user') -> Optional[User]:
        """Create a new user - uses database if available"""
        # Validate email
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
            return None

        if DATABASE_AVAILABLE:
            try:
                user_data = db_create_user(email, password, name, role)
                return self._dict_to_user(user_data)
            except Exception as e:
                logger.error(f"Database create_user failed: {e}")
                # Fall through to memory store

        # Fallback to in-memory
        user_id = self._generate_user_id(email)
        if user_id in self._memory_users:
            return None

        api_key = self._generate_api_key()
        self._memory_users[user_id] = {
            'id': user_id,
            'email': email.lower(),
            'name': name,
            'password_hash': self._hash_password(password),
            'role': role,
            'api_key': api_key,
            'created_at': datetime.utcnow()
        }
        self._memory_api_keys[api_key] = user_id

        return User(
            id=user_id,
            email=email.lower(),
            name=name,
            role=role,
            api_key=api_key,
            created_at=self._memory_users[user_id]['created_at']
        )

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        if DATABASE_AVAILABLE:
            try:
                user_data = db_authenticate_user(email, password)
                return self._dict_to_user(user_data)
            except Exception as e:
                logger.error(f"Database authenticate failed: {e}")

        # Fallback to in-memory
        user_id = self._generate_user_id(email)
        if user_id not in self._memory_users:
            return None

        user_data = self._memory_users[user_id]
        if user_data['password_hash'] != self._hash_password(password):
            return None

        return self._dict_to_user(user_data)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        if DATABASE_AVAILABLE:
            try:
                user_data = db_get_user_by_id(user_id)
                if user_data:
                    return self._dict_to_user(user_data)
            except Exception as e:
                logger.error(f"Database get_user_by_id failed: {e}")

        # Fallback to in-memory
        if user_id in self._memory_users:
            return self._dict_to_user(self._memory_users[user_id])
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        if DATABASE_AVAILABLE:
            try:
                user_data = db_get_user_by_email(email)
                if user_data:
                    return self._dict_to_user(user_data)
            except Exception as e:
                logger.error(f"Database get_user_by_email failed: {e}")

        # Fallback to in-memory using ID generation
        user_id = self._generate_user_id(email)
        if user_id in self._memory_users:
            return self._dict_to_user(self._memory_users[user_id])
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        if DATABASE_AVAILABLE:
            try:
                user_data = db_get_user_by_api_key(api_key)
                if user_data:
                    return self._dict_to_user(user_data)
            except Exception as e:
                logger.error(f"Database get_user_by_api_key failed: {e}")

        # Fallback to in-memory
        if api_key in self._memory_api_keys:
            user_id = self._memory_api_keys[api_key]
            return self.get_user_by_id(user_id)
        return None

    def revoke_token(self, jti: str):
        """Revoke a token by JTI"""
        if DATABASE_AVAILABLE:
            try:
                db_revoke_session(jti)
            except Exception as e:
                logger.error(f"Database revoke_session failed: {e}")
        self._revoked_tokens.add(jti)

    def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        if DATABASE_AVAILABLE:
            try:
                return db_is_session_revoked(jti)
            except Exception as e:
                logger.error(f"Database is_session_revoked failed: {e}")
        return jti in self._revoked_tokens

    def regenerate_user_api_key(self, user_id: str) -> Optional[str]:
        """Regenerate API key for a user"""
        new_api_key = self._generate_api_key()

        if DATABASE_AVAILABLE:
            try:
                # Import update function
                try:
                    from database import update_user_api_key
                except ImportError:
                    from backend.database import update_user_api_key

                if update_user_api_key(user_id, new_api_key):
                    return new_api_key
            except Exception as e:
                logger.error(f"Database regenerate_api_key failed: {e}")

        # Fallback to in-memory
        if user_id in self._memory_users:
            old_key = self._memory_users[user_id].get('api_key')
            if old_key and old_key in self._memory_api_keys:
                del self._memory_api_keys[old_key]

            self._memory_users[user_id]['api_key'] = new_api_key
            self._memory_api_keys[new_api_key] = user_id
            return new_api_key

        return None

# Global user store instance
user_store = UserStore()

# ==============================================================================
# JWT TOKEN MANAGEMENT
# ==============================================================================

def create_access_token(user: User) -> str:
    """Create JWT access token"""
    now = datetime.utcnow()
    jti = secrets.token_hex(16)
    expires_at = now + timedelta(seconds=JWT_ACCESS_TOKEN_EXPIRES)

    payload = {
        'user_id': user.id,
        'email': user.email,
        'role': user.role,
        'token_type': 'access',
        'exp': expires_at,
        'iat': now,
        'jti': jti
    }

    # Create session record in database for token revocation tracking
    if DATABASE_AVAILABLE:
        try:
            db_create_session(user.id, jti, 'access', expires_at)
        except Exception as e:
            logger.error(f"Failed to create session record: {e}")

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_refresh_token(user: User) -> str:
    """Create JWT refresh token"""
    now = datetime.utcnow()
    jti = secrets.token_hex(16)
    expires_at = now + timedelta(seconds=JWT_REFRESH_TOKEN_EXPIRES)

    payload = {
        'user_id': user.id,
        'email': user.email,
        'role': user.role,
        'token_type': 'refresh',
        'exp': expires_at,
        'iat': now,
        'jti': jti
    }

    # Create session record in database
    if DATABASE_AVAILABLE:
        try:
            db_create_session(user.id, jti, 'refresh', expires_at)
        except Exception as e:
            logger.error(f"Failed to create refresh session record: {e}")

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Check if token is revoked
        if user_store.is_token_revoked(payload.get('jti', '')):
            return None

        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None

def refresh_access_token(refresh_token: str) -> Optional[Tuple[str, str]]:
    """Refresh access token using refresh token"""
    payload = decode_token(refresh_token)

    if not payload:
        return None

    if payload.get('token_type') != 'refresh':
        return None

    user = user_store.get_user_by_id(payload['user_id'])
    if not user:
        return None

    # Revoke old refresh token
    user_store.revoke_token(payload['jti'])

    # Create new tokens
    new_access_token = create_access_token(user)
    new_refresh_token = create_refresh_token(user)

    return new_access_token, new_refresh_token

# ==============================================================================
# AUTHENTICATION DECORATORS
# ==============================================================================

def get_token_from_request() -> Optional[str]:
    """Extract token from request (header or query param)"""
    # Check Authorization header
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]

    # Check X-API-Key header
    api_key = request.headers.get('X-API-Key', '')
    if api_key.startswith(API_KEY_PREFIX):
        return api_key

    # Check query parameter (for downloads)
    token = request.args.get('token', '')
    if token:
        return token

    return None

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()

        if not token:
            return jsonify({
                'status': 'error',
                'error': 'Authentication required',
                'error_code': 'AUTH_REQUIRED'
            }), 401

        # Check if it's an API key
        if token.startswith(API_KEY_PREFIX):
            user = user_store.get_user_by_api_key(token)
            if not user:
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid API key',
                    'error_code': 'INVALID_API_KEY'
                }), 401
            g.current_user = user
            g.auth_type = 'api_key'
        else:
            # JWT token
            payload = decode_token(token)
            if not payload:
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid or expired token',
                    'error_code': 'INVALID_TOKEN'
                }), 401

            if payload.get('token_type') != 'access':
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid token type',
                    'error_code': 'INVALID_TOKEN_TYPE'
                }), 401

            user = user_store.get_user_by_id(payload['user_id'])
            if not user:
                return jsonify({
                    'status': 'error',
                    'error': 'User not found',
                    'error_code': 'USER_NOT_FOUND'
                }), 401

            g.current_user = user
            g.auth_type = 'jwt'
            g.token_payload = payload

        return f(*args, **kwargs)

    return decorated

def require_admin(f):
    """Decorator to require admin role (admin or superadmin)"""
    @wraps(f)
    @require_auth
    def decorated(*args, **kwargs):
        if g.current_user.role not in ['admin', 'superadmin']:
            return jsonify({
                'status': 'error',
                'error': 'Admin access required',
                'error_code': 'ADMIN_REQUIRED'
            }), 403

        return f(*args, **kwargs)

    return decorated

def optional_auth(f):
    """Decorator for optional authentication (public endpoints)"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()

        if token:
            if token.startswith(API_KEY_PREFIX):
                user = user_store.get_user_by_api_key(token)
                if user:
                    g.current_user = user
                    g.auth_type = 'api_key'
            else:
                payload = decode_token(token)
                if payload and payload.get('token_type') == 'access':
                    user = user_store.get_user_by_id(payload['user_id'])
                    if user:
                        g.current_user = user
                        g.auth_type = 'jwt'
                        g.token_payload = payload

        return f(*args, **kwargs)

    return decorated

# ==============================================================================
# AUTH BLUEPRINT
# ==============================================================================

from flask import Blueprint

auth_bp = Blueprint('auth', __name__, url_prefix='/api/v1/auth')

@auth_bp.route('/beta-status', methods=['GET'])
def beta_status():
    """Check if beta mode is enabled (for frontend to show/hide invite code field)"""
    return jsonify({
        'status': 'success',
        'data': {
            'beta_mode': BETA_MODE,
            'message': 'Invite code required for registration' if BETA_MODE else 'Open registration'
        }
    })

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'error': 'Request body required',
                'error_code': 'INVALID_REQUEST'
            }), 400

        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        invite_code = data.get('invite_code', '').strip().upper()

        # Validate invite code in beta mode
        if BETA_MODE:
            if not invite_code:
                return jsonify({
                    'status': 'error',
                    'error': 'Invite code is required for beta access',
                    'error_code': 'INVITE_CODE_REQUIRED'
                }), 400

            if invite_code not in BETA_INVITE_CODES:
                logger.warning(f"Invalid invite code attempt: {invite_code} by {email}")
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid invite code. Please contact admin for access.',
                    'error_code': 'INVALID_INVITE_CODE'
                }), 403

        if not email or not password or not name:
            return jsonify({
                'status': 'error',
                'error': 'Email, password, and name are required',
                'error_code': 'MISSING_FIELDS'
            }), 400

        if len(password) < 8:
            return jsonify({
                'status': 'error',
                'error': 'Password must be at least 8 characters',
                'error_code': 'WEAK_PASSWORD'
            }), 400

        user = user_store.create_user(email, password, name)

        if not user:
            return jsonify({
                'status': 'error',
                'error': 'User already exists or invalid email',
                'error_code': 'USER_EXISTS'
            }), 409

        # Create tokens
        access_token = create_access_token(user)
        refresh_token = create_refresh_token(user)

        logger.info(f"User registered: {email}")

        return jsonify({
            'status': 'success',
            'data': {
                'user': user.to_dict(),
                'access_token': access_token,
                'refresh_token': refresh_token,
                'api_key': user.api_key,
                'expires_in': JWT_ACCESS_TOKEN_EXPIRES
            }
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Registration failed: {str(e)}',
            'error_code': 'REGISTRATION_FAILED'
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login with email and password"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'error': 'Request body required',
                'error_code': 'INVALID_REQUEST'
            }), 400

        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({
                'status': 'error',
                'error': 'Email and password are required',
                'error_code': 'MISSING_FIELDS'
            }), 400

        # First check if user exists
        existing_user = user_store.get_user_by_email(email)
        if not existing_user:
            return jsonify({
                'status': 'error',
                'error': 'No account found with this email. Please create an account first.',
                'error_code': 'USER_NOT_FOUND'
            }), 404

        user = user_store.authenticate(email, password)

        if not user:
            return jsonify({
                'status': 'error',
                'error': 'Incorrect password. Please try again.',
                'error_code': 'INVALID_PASSWORD'
            }), 401

        # Create tokens
        access_token = create_access_token(user)
        refresh_token = create_refresh_token(user)

        logger.info(f"User logged in: {email}")

        return jsonify({
            'status': 'success',
            'data': {
                'user': user.to_dict(),
                'access_token': access_token,
                'refresh_token': refresh_token,
                'api_key': user.api_key,
                'expires_in': JWT_ACCESS_TOKEN_EXPIRES
            }
        })

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Login failed: {str(e)}',
            'error_code': 'LOGIN_FAILED'
        }), 500

@auth_bp.route('/refresh', methods=['POST'])
def refresh():
    """Refresh access token"""
    data = request.get_json()

    if not data:
        return jsonify({
            'status': 'error',
            'error': 'Request body required',
            'error_code': 'INVALID_REQUEST'
        }), 400

    refresh_token = data.get('refresh_token', '')

    if not refresh_token:
        return jsonify({
            'status': 'error',
            'error': 'Refresh token required',
            'error_code': 'MISSING_TOKEN'
        }), 400

    result = refresh_access_token(refresh_token)

    if not result:
        return jsonify({
            'status': 'error',
            'error': 'Invalid or expired refresh token',
            'error_code': 'INVALID_REFRESH_TOKEN'
        }), 401

    new_access_token, new_refresh_token = result

    return jsonify({
        'status': 'success',
        'data': {
            'access_token': new_access_token,
            'refresh_token': new_refresh_token,
            'expires_in': JWT_ACCESS_TOKEN_EXPIRES
        }
    })

@auth_bp.route('/logout', methods=['POST'])
@require_auth
def logout():
    """Logout (revoke tokens)"""
    if hasattr(g, 'token_payload') and g.token_payload:
        user_store.revoke_token(g.token_payload['jti'])

    logger.info(f"User logged out: {g.current_user.email}")

    return jsonify({
        'status': 'success',
        'message': 'Successfully logged out'
    })

@auth_bp.route('/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user info"""
    return jsonify({
        'status': 'success',
        'data': {
            'user': g.current_user.to_dict(),
            'auth_type': g.auth_type
        }
    })

@auth_bp.route('/api-key/regenerate', methods=['POST'])
@require_auth
def regenerate_api_key():
    """Regenerate API key"""
    user = g.current_user

    # Generate new API key
    new_api_key = user_store.regenerate_user_api_key(user.id)

    if not new_api_key:
        return jsonify({
            'status': 'error',
            'error': 'Failed to regenerate API key',
            'error_code': 'REGENERATE_FAILED'
        }), 500

    logger.info(f"API key regenerated for user: {user.email}")

    return jsonify({
        'status': 'success',
        'data': {
            'api_key': new_api_key
        }
    })

# ==============================================================================
# QR CODE AUTHENTICATION FOR MOBILE APP
# ==============================================================================

# QR Token storage (token -> {user_id, created_at, expires_at, used})
_qr_tokens: Dict[str, Dict] = {}

QR_TOKEN_EXPIRES = 300  # 5 minutes

@auth_bp.route('/generate-qr', methods=['POST'])
@require_auth
def generate_qr_token():
    """
    Generate a QR code token for mobile app login.
    Web user generates this, mobile app scans and logs in.
    """
    user = g.current_user

    # Generate unique QR token
    qr_token = secrets.token_urlsafe(32)
    now = datetime.utcnow()

    # Store token with user association
    _qr_tokens[qr_token] = {
        'user_id': user.id,
        'created_at': now,
        'expires_at': now + timedelta(seconds=QR_TOKEN_EXPIRES),
        'used': False
    }

    # Clean up expired tokens
    expired = [t for t, data in _qr_tokens.items()
               if data['expires_at'] < now or data['used']]
    for t in expired:
        del _qr_tokens[t]

    logger.info(f"QR token generated for user: {user.email}")

    return jsonify({
        'status': 'success',
        'data': {
            'qr_token': qr_token,
            'qr_value': f'reelsense:{qr_token}',
            'expires_in': QR_TOKEN_EXPIRES,
            'expires_at': (_qr_tokens[qr_token]['expires_at']).isoformat()
        }
    })

@auth_bp.route('/qr-login', methods=['POST'])
def qr_login():
    """
    Login using QR code token (for mobile app).
    Mobile app sends the scanned QR token to authenticate.
    """
    data = request.get_json()

    if not data:
        return jsonify({
            'status': 'error',
            'error': 'Request body required',
            'error_code': 'INVALID_REQUEST'
        }), 400

    qr_token = data.get('qr_token', '')

    if not qr_token:
        return jsonify({
            'status': 'error',
            'error': 'QR token required',
            'error_code': 'MISSING_QR_TOKEN'
        }), 400

    # Validate QR token
    if qr_token not in _qr_tokens:
        return jsonify({
            'status': 'error',
            'error': 'Invalid QR token',
            'error_code': 'INVALID_QR_TOKEN'
        }), 401

    token_data = _qr_tokens[qr_token]

    # Check if expired
    if token_data['expires_at'] < datetime.utcnow():
        del _qr_tokens[qr_token]
        return jsonify({
            'status': 'error',
            'error': 'QR token expired',
            'error_code': 'QR_TOKEN_EXPIRED'
        }), 401

    # Check if already used
    if token_data['used']:
        return jsonify({
            'status': 'error',
            'error': 'QR token already used',
            'error_code': 'QR_TOKEN_USED'
        }), 401

    # Mark as used
    _qr_tokens[qr_token]['used'] = True

    # Get user
    user = user_store.get_user_by_id(token_data['user_id'])

    if not user:
        return jsonify({
            'status': 'error',
            'error': 'User not found',
            'error_code': 'USER_NOT_FOUND'
        }), 404

    # Create tokens for mobile app
    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user)

    logger.info(f"QR login successful for user: {user.email}")

    return jsonify({
        'status': 'success',
        'data': {
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token,
            'api_key': user.api_key,
            'expires_in': JWT_ACCESS_TOKEN_EXPIRES
        }
    })

@auth_bp.route('/qr-status/<qr_token>', methods=['GET'])
def qr_status(qr_token):
    """
    Check QR token status (for web polling).
    Web app can poll this to know when mobile has scanned.
    """
    if qr_token not in _qr_tokens:
        return jsonify({
            'status': 'error',
            'error': 'Invalid QR token',
            'error_code': 'INVALID_QR_TOKEN'
        }), 404

    token_data = _qr_tokens[qr_token]

    return jsonify({
        'status': 'success',
        'data': {
            'used': token_data['used'],
            'expired': token_data['expires_at'] < datetime.utcnow()
        }
    })

# ==============================================================================
# PASSWORD RESET
# ==============================================================================

# Password reset token storage (token -> {email, created_at, expires_at, used})
_reset_tokens: Dict[str, Dict] = {}
RESET_TOKEN_EXPIRES = 3600  # 1 hour

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """
    Request password reset. Generates a reset token.
    In production, this would send an email. For beta, returns the token directly.
    """
    data = request.get_json()

    if not data:
        return jsonify({
            'status': 'error',
            'error': 'Request body required',
            'error_code': 'INVALID_REQUEST'
        }), 400

    email = data.get('email', '').strip().lower()

    if not email:
        return jsonify({
            'status': 'error',
            'error': 'Email is required',
            'error_code': 'MISSING_EMAIL'
        }), 400

    # Check if user exists
    user = user_store.get_user_by_email(email)
    if not user:
        # Don't reveal if email exists for security
        # But still return success to prevent enumeration
        logger.warning(f"Password reset requested for non-existent email: {email}")
        return jsonify({
            'status': 'success',
            'message': 'If an account exists with this email, a reset link has been sent.'
        })

    # Generate reset token
    reset_token = secrets.token_urlsafe(32)
    now = datetime.utcnow()

    _reset_tokens[reset_token] = {
        'email': email,
        'user_id': user.id,
        'created_at': now,
        'expires_at': now + timedelta(seconds=RESET_TOKEN_EXPIRES),
        'used': False
    }

    # Clean up expired tokens
    expired = [t for t, d in _reset_tokens.items()
               if d['expires_at'] < now or d['used']]
    for t in expired:
        del _reset_tokens[t]

    logger.info(f"Password reset token generated for: {email}")

    # For beta: Return token directly (in production, would send email)
    # The reset URL would be: /auth.html?reset_token=<token>
    reset_url = f"/auth.html?reset_token={reset_token}"

    return jsonify({
        'status': 'success',
        'message': 'Password reset link generated.',
        'data': {
            'reset_token': reset_token,
            'reset_url': reset_url,
            'expires_in': RESET_TOKEN_EXPIRES,
            'note': 'In production, this would be sent via email.'
        }
    })

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """
    Reset password using reset token.
    """
    data = request.get_json()

    if not data:
        return jsonify({
            'status': 'error',
            'error': 'Request body required',
            'error_code': 'INVALID_REQUEST'
        }), 400

    reset_token = data.get('reset_token', '').strip()
    new_password = data.get('new_password', '')

    if not reset_token or not new_password:
        return jsonify({
            'status': 'error',
            'error': 'Reset token and new password are required',
            'error_code': 'MISSING_FIELDS'
        }), 400

    if len(new_password) < 8:
        return jsonify({
            'status': 'error',
            'error': 'Password must be at least 8 characters',
            'error_code': 'WEAK_PASSWORD'
        }), 400

    # Validate reset token
    if reset_token not in _reset_tokens:
        return jsonify({
            'status': 'error',
            'error': 'Invalid or expired reset token',
            'error_code': 'INVALID_RESET_TOKEN'
        }), 401

    token_data = _reset_tokens[reset_token]

    # Check if expired
    if token_data['expires_at'] < datetime.utcnow():
        del _reset_tokens[reset_token]
        return jsonify({
            'status': 'error',
            'error': 'Reset token has expired. Please request a new one.',
            'error_code': 'RESET_TOKEN_EXPIRED'
        }), 401

    # Check if already used
    if token_data['used']:
        return jsonify({
            'status': 'error',
            'error': 'Reset token has already been used',
            'error_code': 'RESET_TOKEN_USED'
        }), 401

    # Update password in database
    try:
        if DATABASE_AVAILABLE:
            try:
                from database import update_user_password
            except ImportError:
                from backend.database import update_user_password

            if update_user_password(token_data['user_id'], new_password):
                # Mark token as used
                _reset_tokens[reset_token]['used'] = True
                logger.info(f"Password reset successful for: {token_data['email']}")

                return jsonify({
                    'status': 'success',
                    'message': 'Password has been reset successfully. You can now sign in.'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to update password',
                    'error_code': 'UPDATE_FAILED'
                }), 500
        else:
            # In-memory fallback
            user_id = token_data['user_id']
            if user_id in user_store._memory_users:
                user_store._memory_users[user_id]['password_hash'] = user_store._hash_password(new_password)
                _reset_tokens[reset_token]['used'] = True
                logger.info(f"Password reset successful (in-memory) for: {token_data['email']}")

                return jsonify({
                    'status': 'success',
                    'message': 'Password has been reset successfully. You can now sign in.'
                })

        return jsonify({
            'status': 'error',
            'error': 'User not found',
            'error_code': 'USER_NOT_FOUND'
        }), 404

    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Password reset failed: {str(e)}',
            'error_code': 'RESET_FAILED'
        }), 500

@auth_bp.route('/validate-reset-token/<token>', methods=['GET'])
def validate_reset_token(token):
    """
    Validate a reset token (for frontend to check before showing form).
    """
    if token not in _reset_tokens:
        return jsonify({
            'status': 'error',
            'error': 'Invalid reset token',
            'error_code': 'INVALID_RESET_TOKEN'
        }), 404

    token_data = _reset_tokens[token]

    if token_data['expires_at'] < datetime.utcnow():
        return jsonify({
            'status': 'error',
            'error': 'Reset token has expired',
            'error_code': 'RESET_TOKEN_EXPIRED'
        }), 401

    if token_data['used']:
        return jsonify({
            'status': 'error',
            'error': 'Reset token has already been used',
            'error_code': 'RESET_TOKEN_USED'
        }), 401

    return jsonify({
        'status': 'success',
        'data': {
            'valid': True,
            'email': token_data['email']
        }
    })

# ==============================================================================
# USER VIDEOS (My Videos Page)
# ==============================================================================

@auth_bp.route('/user/videos', methods=['GET'])
@require_auth
def get_user_videos_endpoint():
    """Get current user's videos for My Videos page"""
    try:
        if DATABASE_AVAILABLE:
            try:
                from database import get_user_videos, get_retention_setting
            except ImportError:
                from backend.database import get_user_videos, get_retention_setting

            include_expired = request.args.get('include_expired', 'false').lower() == 'true'
            videos = get_user_videos(g.current_user.id, include_expired=include_expired)
            retention_days = get_retention_setting()

            return jsonify({
                'status': 'success',
                'data': {
                    'videos': videos,
                    'retention_days': retention_days,
                    'count': len(videos)
                }
            })
        else:
            return jsonify({
                'status': 'success',
                'data': {
                    'videos': [],
                    'retention_days': 3,
                    'count': 0
                }
            })
    except Exception as e:
        logger.error(f"Error fetching user videos: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Failed to load videos: {str(e)}',
            'error_code': 'LOAD_VIDEOS_FAILED'
        }), 500

@auth_bp.route('/user/videos/<video_id>', methods=['DELETE'])
@require_auth
def delete_user_video(video_id):
    """Delete a user's video (soft delete)"""
    try:
        if DATABASE_AVAILABLE:
            try:
                from database import soft_delete_video
            except ImportError:
                from backend.database import soft_delete_video

            if soft_delete_video(video_id, g.current_user.id):
                logger.info(f"Video {video_id} deleted by user {g.current_user.email}")
                return jsonify({
                    'status': 'success',
                    'message': 'Video deleted successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Video not found or already deleted',
                    'error_code': 'VIDEO_NOT_FOUND'
                }), 404
        else:
            return jsonify({
                'status': 'error',
                'error': 'Database not available',
                'error_code': 'DATABASE_UNAVAILABLE'
            }), 500
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Failed to delete video: {str(e)}',
            'error_code': 'DELETE_FAILED'
        }), 500

@auth_bp.route('/user/credits', methods=['GET'])
@require_auth
def get_user_credits():
    """Get current user's credit balance"""
    try:
        if DATABASE_AVAILABLE:
            try:
                from database import get_user_credits
            except ImportError:
                from backend.database import get_user_credits

            credits = get_user_credits(g.current_user.id)
            return jsonify({
                'status': 'success',
                'data': credits
            })
        else:
            return jsonify({
                'status': 'success',
                'data': {
                    'balance': 100,
                    'total_used': 0,
                    'total_given': 100
                }
            })
    except Exception as e:
        logger.error(f"Error fetching credits: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to load credits',
            'error_code': 'LOAD_CREDITS_FAILED'
        }), 500
