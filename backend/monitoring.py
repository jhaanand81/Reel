"""
Monitoring Module for Reel Sense API
Prometheus metrics + Sentry error tracking
"""

import os
import time
import logging
from functools import wraps
from typing import Optional
from flask import Flask, request, g

logger = logging.getLogger(__name__)

# ==============================================================================
# PROMETHEUS METRICS
# ==============================================================================

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics disabled.")

if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        'reelsense_http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code']
    )

    REQUEST_LATENCY = Histogram(
        'reelsense_http_request_duration_seconds',
        'HTTP request latency',
        ['method', 'endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
    )

    # Video generation metrics
    VIDEO_GENERATION_COUNT = Counter(
        'reelsense_video_generations_total',
        'Total video generation requests',
        ['provider', 'status', 'reel_type']
    )

    VIDEO_GENERATION_DURATION = Histogram(
        'reelsense_video_generation_duration_seconds',
        'Video generation duration',
        ['provider'],
        buckets=[30, 60, 120, 180, 300, 420, 600, 900]
    )

    ACTIVE_VIDEO_JOBS = Gauge(
        'reelsense_active_video_jobs',
        'Currently active video generation jobs',
        ['provider']
    )

    # Script generation metrics
    SCRIPT_GENERATION_COUNT = Counter(
        'reelsense_script_generations_total',
        'Total script generation requests',
        ['status']
    )

    SCRIPT_GENERATION_DURATION = Histogram(
        'reelsense_script_generation_duration_seconds',
        'Script generation duration',
        buckets=[0.5, 1, 2, 5, 10, 30]
    )

    # TTS metrics
    TTS_GENERATION_COUNT = Counter(
        'reelsense_tts_generations_total',
        'Total TTS generation requests',
        ['provider', 'status']
    )

    # Auth metrics
    AUTH_REQUESTS = Counter(
        'reelsense_auth_requests_total',
        'Authentication requests',
        ['action', 'status']
    )

    # Error metrics
    ERROR_COUNT = Counter(
        'reelsense_errors_total',
        'Total errors',
        ['error_type', 'endpoint']
    )

    # Service health
    SERVICE_STATUS = Gauge(
        'reelsense_service_status',
        'Service availability status (1=up, 0=down)',
        ['service']
    )

    # System info
    APP_INFO = Info('reelsense_app', 'Application information')

# ==============================================================================
# SENTRY ERROR TRACKING
# ==============================================================================

try:
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.redis import RedisIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logger.warning("sentry_sdk not installed. Error tracking disabled.")

def init_sentry(app: Flask):
    """Initialize Sentry error tracking"""
    if not SENTRY_AVAILABLE:
        logger.warning("Sentry SDK not available")
        return False

    sentry_dsn = os.getenv('SENTRY_DSN', '')

    if not sentry_dsn or sentry_dsn.startswith('https://your'):
        logger.warning("Sentry DSN not configured")
        return False

    try:
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[
                FlaskIntegration(),
                CeleryIntegration(),
                RedisIntegration(),
            ],
            # Performance monitoring
            traces_sample_rate=float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', 0.1)),
            # Profile sampling
            profiles_sample_rate=float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', 0.1)),
            # Environment
            environment=os.getenv('FLASK_ENV', 'production'),
            # Release tracking
            release=os.getenv('APP_VERSION', '2.0.0'),
            # Server name
            server_name=os.getenv('SERVER_NAME', 'reelsense-api'),
            # Send default PII
            send_default_pii=False,
            # Before send hook for filtering
            before_send=_sentry_before_send,
        )

        logger.info("Sentry initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        return False

def _sentry_before_send(event, hint):
    """Filter sensitive data before sending to Sentry"""
    # Remove sensitive headers
    if 'request' in event and 'headers' in event['request']:
        headers = event['request']['headers']
        sensitive_headers = ['Authorization', 'X-API-Key', 'Cookie']
        for header in sensitive_headers:
            if header in headers:
                headers[header] = '[FILTERED]'

    # Remove sensitive data from breadcrumbs
    if 'breadcrumbs' in event:
        for breadcrumb in event['breadcrumbs'].get('values', []):
            if 'data' in breadcrumb:
                for key in ['password', 'api_key', 'token', 'secret']:
                    if key in breadcrumb['data']:
                        breadcrumb['data'][key] = '[FILTERED]'

    return event

def capture_exception(exception: Exception, extra: dict = None):
    """Capture exception to Sentry"""
    if SENTRY_AVAILABLE:
        with sentry_sdk.push_scope() as scope:
            if extra:
                for key, value in extra.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(exception)

def capture_message(message: str, level: str = 'info', extra: dict = None):
    """Capture message to Sentry"""
    if SENTRY_AVAILABLE:
        with sentry_sdk.push_scope() as scope:
            if extra:
                for key, value in extra.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_message(message, level=level)

def set_user_context(user_id: str, email: str = None, username: str = None):
    """Set user context for Sentry"""
    if SENTRY_AVAILABLE:
        sentry_sdk.set_user({
            'id': user_id,
            'email': email,
            'username': username
        })

# ==============================================================================
# FLASK MIDDLEWARE
# ==============================================================================

def init_monitoring(app: Flask):
    """Initialize all monitoring for Flask app"""

    # Initialize Sentry
    sentry_enabled = init_sentry(app)

    # Set app info for Prometheus
    if PROMETHEUS_AVAILABLE:
        APP_INFO.info({
            'version': os.getenv('APP_VERSION', '2.0.0'),
            'environment': os.getenv('FLASK_ENV', 'production'),
            'sentry_enabled': str(sentry_enabled),
            'python_version': os.popen('python --version').read().strip()
        })

    # Request timing middleware
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.request_id = request.headers.get('X-Request-ID', '')

    @app.after_request
    def after_request(response):
        # Calculate request duration
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time

            # Get endpoint name
            endpoint = request.endpoint or 'unknown'

            # Record Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=response.status_code
                ).inc()

                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=endpoint
                ).observe(duration)

        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Global exception handler with monitoring"""
        endpoint = request.endpoint or 'unknown'

        # Record error in Prometheus
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(
                error_type=type(e).__name__,
                endpoint=endpoint
            ).inc()

        # Capture in Sentry
        capture_exception(e, extra={
            'endpoint': endpoint,
            'method': request.method,
            'url': request.url,
            'request_id': getattr(g, 'request_id', 'unknown')
        })

        # Re-raise for Flask to handle
        raise e

    logger.info("Monitoring middleware initialized")

# ==============================================================================
# METRIC DECORATORS
# ==============================================================================

def track_video_generation(provider: str):
    """Decorator to track video generation metrics"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if PROMETHEUS_AVAILABLE:
                ACTIVE_VIDEO_JOBS.labels(provider=provider).inc()

            start_time = time.time()
            status = 'success'

            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failed'
                raise e
            finally:
                duration = time.time() - start_time

                if PROMETHEUS_AVAILABLE:
                    ACTIVE_VIDEO_JOBS.labels(provider=provider).dec()
                    VIDEO_GENERATION_COUNT.labels(
                        provider=provider,
                        status=status,
                        reel_type=kwargs.get('reel_type', 'unknown')
                    ).inc()
                    VIDEO_GENERATION_DURATION.labels(
                        provider=provider
                    ).observe(duration)

        return decorated
    return decorator

def track_script_generation(f):
    """Decorator to track script generation metrics"""
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = time.time()
        status = 'success'

        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            status = 'failed'
            raise e
        finally:
            duration = time.time() - start_time

            if PROMETHEUS_AVAILABLE:
                SCRIPT_GENERATION_COUNT.labels(status=status).inc()
                SCRIPT_GENERATION_DURATION.observe(duration)

    return decorated

def track_tts_generation(provider: str):
    """Decorator to track TTS generation metrics"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            status = 'success'

            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failed'
                raise e
            finally:
                if PROMETHEUS_AVAILABLE:
                    TTS_GENERATION_COUNT.labels(
                        provider=provider,
                        status=status
                    ).inc()

        return decorated
    return decorator

def track_auth(action: str):
    """Decorator to track auth metrics"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            status = 'success'

            try:
                result = f(*args, **kwargs)
                # Check response status
                if hasattr(result, '__iter__') and len(result) > 1:
                    if result[1] >= 400:
                        status = 'failed'
                return result
            except Exception as e:
                status = 'failed'
                raise e
            finally:
                if PROMETHEUS_AVAILABLE:
                    AUTH_REQUESTS.labels(
                        action=action,
                        status=status
                    ).inc()

        return decorated
    return decorator

# ==============================================================================
# SERVICE HEALTH TRACKING
# ==============================================================================

def update_service_status(service: str, is_healthy: bool):
    """Update service health status"""
    if PROMETHEUS_AVAILABLE:
        SERVICE_STATUS.labels(service=service).set(1 if is_healthy else 0)

def check_all_services():
    """Check and update status of all services"""
    services = {
        'groq': bool(os.getenv('GROQ_API_KEY')),
        'replicate': bool(os.getenv('REPLICATE_API_TOKEN')),
        'runpod': bool(os.getenv('RUNPOD_API_KEY') and os.getenv('RUNPOD_ENDPOINT_ID')),
        'redis': _check_redis(),
        'ffmpeg': _check_ffmpeg(),
    }

    for service, is_healthy in services.items():
        update_service_status(service, is_healthy)

    return services

def _check_redis():
    """Check if Redis is available"""
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        return True
    except:
        return False

def _check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# ==============================================================================
# METRICS ENDPOINT BLUEPRINT
# ==============================================================================

from flask import Blueprint, Response

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    if not PROMETHEUS_AVAILABLE:
        return Response("Prometheus not available", status=503)

    # Update service health before returning metrics
    check_all_services()

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

@metrics_bp.route('/health')
def health():
    """Health check endpoint for load balancers"""
    return {
        'status': 'healthy',
        'timestamp': time.time()
    }

@metrics_bp.route('/readiness')
def readiness():
    """Readiness check for Kubernetes"""
    services = check_all_services()

    # Check critical services
    critical_services = ['groq', 'ffmpeg']
    all_ready = all(services.get(s, False) for s in critical_services)

    if all_ready:
        return {
            'status': 'ready',
            'services': services
        }
    else:
        return {
            'status': 'not_ready',
            'services': services
        }, 503

@metrics_bp.route('/liveness')
def liveness():
    """Liveness check for Kubernetes"""
    return {
        'status': 'alive',
        'timestamp': time.time()
    }
