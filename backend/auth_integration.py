"""
Auth Integration Helper
Add this to your main.py create_app() function after CORS initialization:

# === ADD THESE LINES TO main.py create_app() function ===

# Register Auth Blueprint
try:
    from backend.auth import auth_bp
    app.register_blueprint(auth_bp)
    print("[OK] Auth blueprint registered")
except ImportError:
    from auth import auth_bp
    app.register_blueprint(auth_bp)
    print("[OK] Auth blueprint registered")

# Initialize Monitoring
try:
    from backend.monitoring import init_monitoring, metrics_bp
    init_monitoring(app)
    app.register_blueprint(metrics_bp)
    print("[OK] Monitoring initialized")
except ImportError:
    from monitoring import init_monitoring, metrics_bp
    init_monitoring(app)
    app.register_blueprint(metrics_bp)
    print("[OK] Monitoring initialized")

# Also update CORS allow_headers to include "X-API-Key":
# allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-API-Key"]
"""

def register_auth_and_monitoring(app, logger=None):
    """
    Helper function to register auth and monitoring blueprints.

    Usage in main.py:
        from auth_integration import register_auth_and_monitoring
        register_auth_and_monitoring(app, logger)
    """
    log = logger.info if logger else print
    warn = logger.warning if logger else print

    # Register Auth Blueprint
    try:
        from backend.auth import auth_bp
        app.register_blueprint(auth_bp)
        log("[OK] Auth blueprint registered")
    except ImportError:
        try:
            from auth import auth_bp
            app.register_blueprint(auth_bp)
            log("[OK] Auth blueprint registered")
        except ImportError as e:
            warn(f"[WARN] Auth blueprint not available: {e}")

    # Initialize Monitoring
    try:
        from backend.monitoring import init_monitoring, metrics_bp
        init_monitoring(app)
        app.register_blueprint(metrics_bp)
        log("[OK] Monitoring initialized")
    except ImportError:
        try:
            from monitoring import init_monitoring, metrics_bp
            init_monitoring(app)
            app.register_blueprint(metrics_bp)
            log("[OK] Monitoring initialized")
        except ImportError as e:
            warn(f"[WARN] Monitoring not available: {e}")
