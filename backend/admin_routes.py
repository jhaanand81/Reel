"""
Admin Routes for ReelSense AI
Super Admin Dashboard API endpoints
"""

from flask import Blueprint, request, jsonify, g
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Create admin blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/api/v1/admin')

# Import database functions
try:
    from database import (
        get_admin_stats,
        get_all_users_with_credits,
        get_user_by_id,
        get_user_credits,
        add_credits,
        get_credit_history,
        update_user_role,
        delete_user,
        get_all_videos,
        delete_video,
        get_user_jobs,
        # Analytics
        get_analytics_data,
        # Audit logs
        log_audit,
        get_audit_logs,
        get_audit_log_count,
        # Settings
        get_all_settings,
        get_setting,
        update_setting,
        create_setting,
        # Notifications
        create_notification,
        get_notifications,
        get_user_notifications,
        mark_notification_read,
        update_notification,
        delete_notification,
        # Support tickets
        create_ticket,
        get_all_tickets,
        get_user_tickets,
        get_ticket_by_id,
        update_ticket,
        add_ticket_reply,
        get_ticket_stats,
        # Export
        export_users_data,
        export_transactions_data,
        export_videos_data
    )
    DATABASE_AVAILABLE = True
except ImportError:
    try:
        from backend.database import (
            get_admin_stats,
            get_all_users_with_credits,
            get_user_by_id,
            get_user_credits,
            add_credits,
            get_credit_history,
            update_user_role,
            delete_user,
            get_all_videos,
            delete_video,
            get_user_jobs,
            # Analytics
            get_analytics_data,
            # Audit logs
            log_audit,
            get_audit_logs,
            get_audit_log_count,
            # Settings
            get_all_settings,
            get_setting,
            update_setting,
            create_setting,
            # Notifications
            create_notification,
            get_notifications,
            get_user_notifications,
            mark_notification_read,
            update_notification,
            delete_notification,
            # Support tickets
            create_ticket,
            get_all_tickets,
            get_user_tickets,
            get_ticket_by_id,
            update_ticket,
            add_ticket_reply,
            get_ticket_stats,
            # Export
            export_users_data,
            export_transactions_data,
            export_videos_data
        )
        DATABASE_AVAILABLE = True
    except ImportError:
        DATABASE_AVAILABLE = False
        logger.error("Database module not available for admin routes")

# Import auth functions
try:
    from auth import require_auth, require_admin, user_store
except ImportError:
    from backend.auth import require_auth, require_admin, user_store


# ==============================================================================
# RBAC PERMISSION SYSTEM
# ==============================================================================

PERMISSIONS = {
    'superadmin': ['*'],  # All permissions
    'admin': ['view_users', 'view_videos', 'view_stats', 'manage_users', 'manage_credits', 'delete_videos'],
    'editor': ['view_videos', 'generate_videos', 'delete_own_videos'],
    'viewer': ['view_videos'],
    'api': ['generate_videos', 'view_own_videos'],
    'user': ['generate_videos', 'view_own_videos']
}

def has_permission(role: str, permission: str) -> bool:
    """Check if role has specific permission"""
    perms = PERMISSIONS.get(role, [])
    return '*' in perms or permission in perms

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        @require_auth
        def wrapper(*args, **kwargs):
            user_role = g.current_user.role
            if not has_permission(user_role, permission):
                return jsonify({
                    'status': 'error',
                    'error': f'Permission denied. Required: {permission}',
                    'error_code': 'PERMISSION_DENIED'
                }), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ==============================================================================
# DASHBOARD STATS
# ==============================================================================

@admin_bp.route('/stats', methods=['GET'])
@require_admin
def get_stats():
    """Get dashboard statistics"""
    try:
        if not DATABASE_AVAILABLE:
            return jsonify({
                'status': 'error',
                'error': 'Database not available',
                'error_code': 'DB_UNAVAILABLE'
            }), 500

        stats = get_admin_stats()
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'STATS_ERROR'
        }), 500


# ==============================================================================
# USER MANAGEMENT
# ==============================================================================

@admin_bp.route('/users', methods=['GET'])
@require_admin
def list_users():
    """Get all users with credits info"""
    try:
        if not DATABASE_AVAILABLE:
            return jsonify({
                'status': 'error',
                'error': 'Database not available',
                'error_code': 'DB_UNAVAILABLE'
            }), 500

        users = get_all_users_with_credits()
        return jsonify({
            'status': 'success',
            'data': {
                'users': users,
                'total': len(users)
            }
        })
    except Exception as e:
        logger.error(f"List users error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'LIST_USERS_ERROR'
        }), 500


@admin_bp.route('/users/<user_id>', methods=['GET'])
@require_admin
def get_user_detail(user_id):
    """Get detailed user info including credits and videos"""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({
                'status': 'error',
                'error': 'User not found',
                'error_code': 'USER_NOT_FOUND'
            }), 404

        credits = get_user_credits(user_id)
        history = get_credit_history(user_id, limit=20)
        videos = get_user_jobs(user_id, limit=20)

        return jsonify({
            'status': 'success',
            'data': {
                'user': user,
                'credits': credits,
                'credit_history': history,
                'videos': videos
            }
        })
    except Exception as e:
        logger.error(f"Get user detail error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'USER_DETAIL_ERROR'
        }), 500


@admin_bp.route('/users/<user_id>/role', methods=['PUT'])
@require_admin
def change_user_role(user_id):
    """Change user's role"""
    try:
        data = request.get_json()
        new_role = data.get('role')

        if not new_role:
            return jsonify({
                'status': 'error',
                'error': 'Role is required',
                'error_code': 'MISSING_ROLE'
            }), 400

        valid_roles = ['user', 'editor', 'viewer', 'admin', 'api']
        if new_role not in valid_roles:
            return jsonify({
                'status': 'error',
                'error': f'Invalid role. Must be one of: {", ".join(valid_roles)}',
                'error_code': 'INVALID_ROLE'
            }), 400

        # Prevent changing own role
        if user_id == g.current_user.id:
            return jsonify({
                'status': 'error',
                'error': 'Cannot change your own role',
                'error_code': 'SELF_ROLE_CHANGE'
            }), 400

        success = update_user_role(user_id, new_role)
        if success:
            logger.info(f"Admin {g.current_user.email} changed role of {user_id} to {new_role}")
            return jsonify({
                'status': 'success',
                'message': f'Role updated to {new_role}'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to update role',
                'error_code': 'UPDATE_FAILED'
            }), 500

    except Exception as e:
        logger.error(f"Change role error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'ROLE_CHANGE_ERROR'
        }), 500


@admin_bp.route('/users/<user_id>', methods=['DELETE'])
@require_admin
def remove_user(user_id):
    """Delete (deactivate) a user"""
    try:
        # Prevent self-deletion
        if user_id == g.current_user.id:
            return jsonify({
                'status': 'error',
                'error': 'Cannot delete yourself',
                'error_code': 'SELF_DELETE'
            }), 400

        success = delete_user(user_id)
        if success:
            logger.info(f"Admin {g.current_user.email} deleted user {user_id}")
            return jsonify({
                'status': 'success',
                'message': 'User deleted'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'User not found or already deleted',
                'error_code': 'DELETE_FAILED'
            }), 404

    except Exception as e:
        logger.error(f"Delete user error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'DELETE_ERROR'
        }), 500


# ==============================================================================
# CREDITS MANAGEMENT
# ==============================================================================

@admin_bp.route('/users/<user_id>/credits', methods=['POST'])
@require_admin
def add_user_credits(user_id):
    """Add credits to a user"""
    try:
        data = request.get_json()
        amount = data.get('amount')
        reason = data.get('reason', 'Admin credit grant')

        if not amount or not isinstance(amount, int) or amount <= 0:
            return jsonify({
                'status': 'error',
                'error': 'Valid positive amount is required',
                'error_code': 'INVALID_AMOUNT'
            }), 400

        if amount > 10000:
            return jsonify({
                'status': 'error',
                'error': 'Maximum 10,000 credits per transaction',
                'error_code': 'AMOUNT_TOO_HIGH'
            }), 400

        result = add_credits(user_id, amount, reason, admin_id=g.current_user.id)

        logger.info(f"Admin {g.current_user.email} added {amount} credits to {user_id}: {reason}")

        return jsonify({
            'status': 'success',
            'data': result,
            'message': f'Added {amount} credits'
        })

    except Exception as e:
        logger.error(f"Add credits error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'ADD_CREDITS_ERROR'
        }), 500


@admin_bp.route('/users/<user_id>/credits/history', methods=['GET'])
@require_admin
def get_user_credit_history(user_id):
    """Get user's credit transaction history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = get_credit_history(user_id, limit=limit)

        return jsonify({
            'status': 'success',
            'data': {
                'history': history,
                'total': len(history)
            }
        })

    except Exception as e:
        logger.error(f"Credit history error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'HISTORY_ERROR'
        }), 500


# ==============================================================================
# VIDEO MANAGEMENT
# ==============================================================================

@admin_bp.route('/videos', methods=['GET'])
@require_admin
def list_videos():
    """Get all videos"""
    try:
        limit = request.args.get('limit', 100, type=int)
        videos = get_all_videos(limit=limit)

        return jsonify({
            'status': 'success',
            'data': {
                'videos': videos,
                'total': len(videos)
            }
        })

    except Exception as e:
        logger.error(f"List videos error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'LIST_VIDEOS_ERROR'
        }), 500


@admin_bp.route('/videos/<video_id>', methods=['DELETE'])
@require_admin
def remove_video(video_id):
    """Delete a video"""
    try:
        success = delete_video(video_id)
        if success:
            logger.info(f"Admin {g.current_user.email} deleted video {video_id}")
            return jsonify({
                'status': 'success',
                'message': 'Video deleted'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Video not found',
                'error_code': 'VIDEO_NOT_FOUND'
            }), 404

    except Exception as e:
        logger.error(f"Delete video error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'DELETE_VIDEO_ERROR'
        }), 500


# ==============================================================================
# CURRENT USER CREDITS (for regular users)
# ==============================================================================

@admin_bp.route('/my-credits', methods=['GET'])
@require_auth
def get_my_credits():
    """Get current user's credits (for any authenticated user)"""
    try:
        user_id = g.current_user.id
        credits = get_user_credits(user_id)
        history = get_credit_history(user_id, limit=10)

        return jsonify({
            'status': 'success',
            'data': {
                'credits': credits,
                'recent_history': history
            }
        })

    except Exception as e:
        logger.error(f"Get my credits error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'CREDITS_ERROR'
        }), 500


# ==============================================================================
# ANALYTICS
# ==============================================================================

@admin_bp.route('/analytics', methods=['GET'])
@require_admin
def get_analytics():
    """Get analytics data for charts"""
    try:
        days = request.args.get('days', 30, type=int)
        if days > 365:
            days = 365

        data = get_analytics_data(days=days)
        return jsonify({
            'status': 'success',
            'data': data
        })

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'ANALYTICS_ERROR'
        }), 500


# ==============================================================================
# AUDIT LOGS
# ==============================================================================

@admin_bp.route('/audit-logs', methods=['GET'])
@require_admin
def list_audit_logs():
    """Get audit logs with pagination and filters"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        admin_id = request.args.get('admin_id')
        action = request.args.get('action')

        logs = get_audit_logs(limit=limit, offset=offset, admin_id=admin_id, action=action)
        total = get_audit_log_count(admin_id=admin_id, action=action)

        return jsonify({
            'status': 'success',
            'data': {
                'logs': logs,
                'total': total,
                'limit': limit,
                'offset': offset
            }
        })

    except Exception as e:
        logger.error(f"Audit logs error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'AUDIT_LOGS_ERROR'
        }), 500


# ==============================================================================
# SYSTEM SETTINGS
# ==============================================================================

@admin_bp.route('/settings', methods=['GET'])
@require_admin
def list_settings():
    """Get all system settings"""
    try:
        settings = get_all_settings()
        return jsonify({
            'status': 'success',
            'data': settings
        })

    except Exception as e:
        logger.error(f"Get settings error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'SETTINGS_ERROR'
        }), 500


@admin_bp.route('/settings/<key>', methods=['PUT'])
@require_admin
def modify_setting(key):
    """Update a system setting"""
    try:
        data = request.get_json()
        value = data.get('value')

        if value is None:
            return jsonify({
                'status': 'error',
                'error': 'Value is required',
                'error_code': 'MISSING_VALUE'
            }), 400

        success = update_setting(key, str(value), updated_by=g.current_user.id)

        if success:
            # Log the action
            log_audit(
                admin_id=g.current_user.id,
                admin_email=g.current_user.email,
                action='update_setting',
                target_type='setting',
                target_id=key,
                details=f'Changed to: {value}',
                ip_address=request.remote_addr
            )

            return jsonify({
                'status': 'success',
                'message': f'Setting {key} updated'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Setting not found',
                'error_code': 'SETTING_NOT_FOUND'
            }), 404

    except Exception as e:
        logger.error(f"Update setting error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'UPDATE_SETTING_ERROR'
        }), 500


# ==============================================================================
# NOTIFICATIONS
# ==============================================================================

@admin_bp.route('/notifications', methods=['GET'])
@require_admin
def list_notifications():
    """Get all notifications (admin view)"""
    try:
        include_inactive = request.args.get('include_inactive', 'false').lower() == 'true'
        notifications = get_notifications(include_inactive=include_inactive)

        return jsonify({
            'status': 'success',
            'data': {
                'notifications': notifications,
                'total': len(notifications)
            }
        })

    except Exception as e:
        logger.error(f"List notifications error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'NOTIFICATIONS_ERROR'
        }), 500


@admin_bp.route('/notifications', methods=['POST'])
@require_admin
def add_notification():
    """Create a new notification/announcement"""
    try:
        data = request.get_json()
        title = data.get('title')
        message = data.get('message')
        notif_type = data.get('type', 'info')
        target_role = data.get('target_role', 'all')
        expires_at = data.get('expires_at')

        if not title or not message:
            return jsonify({
                'status': 'error',
                'error': 'Title and message are required',
                'error_code': 'MISSING_FIELDS'
            }), 400

        notification_id = create_notification(
            title=title,
            message=message,
            type=notif_type,
            target_role=target_role,
            created_by=g.current_user.id,
            expires_at=expires_at
        )

        # Log the action
        log_audit(
            admin_id=g.current_user.id,
            admin_email=g.current_user.email,
            action='create_notification',
            target_type='notification',
            target_id=str(notification_id),
            details=f'Title: {title}',
            ip_address=request.remote_addr
        )

        return jsonify({
            'status': 'success',
            'data': {'id': notification_id},
            'message': 'Notification created'
        })

    except Exception as e:
        logger.error(f"Create notification error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'CREATE_NOTIFICATION_ERROR'
        }), 500


@admin_bp.route('/notifications/<int:notification_id>', methods=['PUT'])
@require_admin
def toggle_notification(notification_id):
    """Activate/deactivate a notification"""
    try:
        data = request.get_json()
        is_active = data.get('is_active', True)

        success = update_notification(notification_id, is_active)

        if success:
            return jsonify({
                'status': 'success',
                'message': 'Notification updated'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Notification not found',
                'error_code': 'NOTIFICATION_NOT_FOUND'
            }), 404

    except Exception as e:
        logger.error(f"Update notification error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'UPDATE_NOTIFICATION_ERROR'
        }), 500


@admin_bp.route('/notifications/<int:notification_id>', methods=['DELETE'])
@require_admin
def remove_notification(notification_id):
    """Delete a notification"""
    try:
        success = delete_notification(notification_id)

        if success:
            return jsonify({
                'status': 'success',
                'message': 'Notification deleted'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Notification not found',
                'error_code': 'NOTIFICATION_NOT_FOUND'
            }), 404

    except Exception as e:
        logger.error(f"Delete notification error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'DELETE_NOTIFICATION_ERROR'
        }), 500


# User endpoint to get their notifications
@admin_bp.route('/my-notifications', methods=['GET'])
@require_auth
def get_my_notifications():
    """Get notifications for current user"""
    try:
        notifications = get_user_notifications(
            user_id=g.current_user.id,
            user_role=g.current_user.role
        )

        return jsonify({
            'status': 'success',
            'data': notifications
        })

    except Exception as e:
        logger.error(f"Get my notifications error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'MY_NOTIFICATIONS_ERROR'
        }), 500


@admin_bp.route('/my-notifications/<int:notification_id>/read', methods=['POST'])
@require_auth
def mark_my_notification_read(notification_id):
    """Mark a notification as read"""
    try:
        mark_notification_read(notification_id, g.current_user.id)
        return jsonify({
            'status': 'success',
            'message': 'Notification marked as read'
        })

    except Exception as e:
        logger.error(f"Mark notification read error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'MARK_READ_ERROR'
        }), 500


# ==============================================================================
# SUPPORT TICKETS
# ==============================================================================

@admin_bp.route('/tickets', methods=['GET'])
@require_admin
def list_tickets():
    """Get all support tickets (admin view)"""
    try:
        status = request.args.get('status')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)

        tickets = get_all_tickets(status=status, limit=limit, offset=offset)
        stats = get_ticket_stats()

        return jsonify({
            'status': 'success',
            'data': {
                'tickets': tickets,
                'stats': stats,
                'total': len(tickets)
            }
        })

    except Exception as e:
        logger.error(f"List tickets error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'TICKETS_ERROR'
        }), 500


@admin_bp.route('/tickets/<int:ticket_id>', methods=['GET'])
@require_admin
def get_ticket(ticket_id):
    """Get ticket details with replies"""
    try:
        ticket = get_ticket_by_id(ticket_id)

        if not ticket:
            return jsonify({
                'status': 'error',
                'error': 'Ticket not found',
                'error_code': 'TICKET_NOT_FOUND'
            }), 404

        return jsonify({
            'status': 'success',
            'data': ticket
        })

    except Exception as e:
        logger.error(f"Get ticket error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'GET_TICKET_ERROR'
        }), 500


@admin_bp.route('/tickets/<int:ticket_id>', methods=['PUT'])
@require_admin
def modify_ticket(ticket_id):
    """Update ticket status/priority/assignment"""
    try:
        data = request.get_json()
        status = data.get('status')
        priority = data.get('priority')
        assigned_to = data.get('assigned_to')
        admin_notes = data.get('admin_notes')

        success = update_ticket(
            ticket_id=ticket_id,
            status=status,
            priority=priority,
            assigned_to=assigned_to,
            admin_notes=admin_notes
        )

        if success:
            # Log the action
            log_audit(
                admin_id=g.current_user.id,
                admin_email=g.current_user.email,
                action='update_ticket',
                target_type='ticket',
                target_id=str(ticket_id),
                details=f'Status: {status}, Priority: {priority}',
                ip_address=request.remote_addr
            )

            return jsonify({
                'status': 'success',
                'message': 'Ticket updated'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Ticket not found',
                'error_code': 'TICKET_NOT_FOUND'
            }), 404

    except Exception as e:
        logger.error(f"Update ticket error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'UPDATE_TICKET_ERROR'
        }), 500


@admin_bp.route('/tickets/<int:ticket_id>/reply', methods=['POST'])
@require_admin
def reply_to_ticket(ticket_id):
    """Add admin reply to ticket"""
    try:
        data = request.get_json()
        message = data.get('message')

        if not message:
            return jsonify({
                'status': 'error',
                'error': 'Message is required',
                'error_code': 'MISSING_MESSAGE'
            }), 400

        reply_id = add_ticket_reply(
            ticket_id=ticket_id,
            user_id=g.current_user.id,
            message=message,
            is_admin=True
        )

        return jsonify({
            'status': 'success',
            'data': {'id': reply_id},
            'message': 'Reply added'
        })

    except Exception as e:
        logger.error(f"Reply to ticket error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'REPLY_ERROR'
        }), 500


# User endpoints for tickets
@admin_bp.route('/my-tickets', methods=['GET'])
@require_auth
def get_my_tickets():
    """Get current user's tickets"""
    try:
        tickets = get_user_tickets(g.current_user.id)
        return jsonify({
            'status': 'success',
            'data': tickets
        })

    except Exception as e:
        logger.error(f"Get my tickets error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'MY_TICKETS_ERROR'
        }), 500


@admin_bp.route('/my-tickets', methods=['POST'])
@require_auth
def submit_ticket():
    """Submit a new support ticket"""
    try:
        data = request.get_json()
        subject = data.get('subject')
        message = data.get('message')
        category = data.get('category', 'general')
        priority = data.get('priority', 'normal')

        if not subject or not message:
            return jsonify({
                'status': 'error',
                'error': 'Subject and message are required',
                'error_code': 'MISSING_FIELDS'
            }), 400

        ticket_id = create_ticket(
            user_id=g.current_user.id,
            user_email=g.current_user.email,
            subject=subject,
            message=message,
            category=category,
            priority=priority
        )

        return jsonify({
            'status': 'success',
            'data': {'id': ticket_id},
            'message': 'Ticket submitted successfully'
        })

    except Exception as e:
        logger.error(f"Submit ticket error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'SUBMIT_TICKET_ERROR'
        }), 500


@admin_bp.route('/my-tickets/<int:ticket_id>/reply', methods=['POST'])
@require_auth
def user_reply_to_ticket(ticket_id):
    """User reply to their own ticket"""
    try:
        data = request.get_json()
        message = data.get('message')

        if not message:
            return jsonify({
                'status': 'error',
                'error': 'Message is required',
                'error_code': 'MISSING_MESSAGE'
            }), 400

        # Verify ticket belongs to user
        ticket = get_ticket_by_id(ticket_id)
        if not ticket or ticket['user_id'] != g.current_user.id:
            return jsonify({
                'status': 'error',
                'error': 'Ticket not found',
                'error_code': 'TICKET_NOT_FOUND'
            }), 404

        reply_id = add_ticket_reply(
            ticket_id=ticket_id,
            user_id=g.current_user.id,
            message=message,
            is_admin=False
        )

        return jsonify({
            'status': 'success',
            'data': {'id': reply_id},
            'message': 'Reply added'
        })

    except Exception as e:
        logger.error(f"User reply error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'USER_REPLY_ERROR'
        }), 500


# ==============================================================================
# EXPORT DATA
# ==============================================================================

@admin_bp.route('/export/users', methods=['GET'])
@require_admin
def export_users():
    """Export users data as JSON (can be converted to CSV on frontend)"""
    try:
        data = export_users_data()

        # Log the action
        log_audit(
            admin_id=g.current_user.id,
            admin_email=g.current_user.email,
            action='export_users',
            target_type='export',
            details=f'Exported {len(data)} users',
            ip_address=request.remote_addr
        )

        return jsonify({
            'status': 'success',
            'data': data,
            'total': len(data)
        })

    except Exception as e:
        logger.error(f"Export users error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'EXPORT_ERROR'
        }), 500


@admin_bp.route('/export/transactions', methods=['GET'])
@require_admin
def export_transactions():
    """Export credit transactions"""
    try:
        days = request.args.get('days', 30, type=int)
        data = export_transactions_data(days=days)

        # Log the action
        log_audit(
            admin_id=g.current_user.id,
            admin_email=g.current_user.email,
            action='export_transactions',
            target_type='export',
            details=f'Exported {len(data)} transactions (last {days} days)',
            ip_address=request.remote_addr
        )

        return jsonify({
            'status': 'success',
            'data': data,
            'total': len(data)
        })

    except Exception as e:
        logger.error(f"Export transactions error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'EXPORT_ERROR'
        }), 500


@admin_bp.route('/export/videos', methods=['GET'])
@require_admin
def export_videos():
    """Export video jobs data"""
    try:
        days = request.args.get('days', 30, type=int)
        data = export_videos_data(days=days)

        # Log the action
        log_audit(
            admin_id=g.current_user.id,
            admin_email=g.current_user.email,
            action='export_videos',
            target_type='export',
            details=f'Exported {len(data)} videos (last {days} days)',
            ip_address=request.remote_addr
        )

        return jsonify({
            'status': 'success',
            'data': data,
            'total': len(data)
        })

    except Exception as e:
        logger.error(f"Export videos error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'error_code': 'EXPORT_ERROR'
        }), 500
