/**
 * ReelSense AI - Admin Dashboard
 * Super Admin functionality for user, video, and credits management
 */

// ==============================================================================
// CONFIGURATION
// ==============================================================================

const API_BASE = window.APP_CONFIG?.API_BASE || 'http://localhost:5000/api/v1';
const ADMIN_API = `${API_BASE}/admin`;

// ==============================================================================
// STATE
// ==============================================================================

let currentUser = null;
let usersData = [];
let videosData = [];

// ==============================================================================
// INITIALIZATION
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    initTabs();
    initSearch();
    initMobileMenu();
});

/**
 * Check authentication and admin status
 */
async function checkAuth() {
    const token = localStorage.getItem('authToken');
    const user = JSON.parse(localStorage.getItem('user') || 'null');

    if (!token || !user) {
        window.location.href = 'auth.html';
        return;
    }

    // Verify admin role
    if (!['admin', 'superadmin'].includes(user.role)) {
        alert('Access denied. Admin privileges required.');
        window.location.href = 'index.html';
        return;
    }

    currentUser = user;

    // Update UI
    document.getElementById('adminEmail').textContent = user.email;
    document.getElementById('adminRole').textContent = user.role.toUpperCase();

    // Load dashboard data
    loadDashboardStats();
    loadUsers();
}

/**
 * Initialize tab navigation
 */
function initTabs() {
    const tabs = document.querySelectorAll('.admin-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Add active to clicked tab
            tab.classList.add('active');

            // Hide all panels
            document.querySelectorAll('.admin-panel').forEach(p => p.classList.remove('active'));

            // Show selected panel
            const tabName = tab.dataset.tab;
            document.getElementById(`${tabName}Panel`).classList.add('active');

            // Load data for tab
            if (tabName === 'users') loadUsers();
            else if (tabName === 'videos') loadVideos();
            else if (tabName === 'credits') loadCreditsOverview();
            else if (tabName === 'analytics') loadAnalytics();
            else if (tabName === 'settings') loadSettings();
            else if (tabName === 'notifications') loadNotifications();
            else if (tabName === 'tickets') loadTickets();
            else if (tabName === 'logs') loadAuditLogs();
        });
    });
}

/**
 * Initialize search functionality
 */
function initSearch() {
    // User search
    const userSearch = document.getElementById('userSearch');
    if (userSearch) {
        userSearch.addEventListener('input', debounce(() => {
            filterUsers(userSearch.value);
        }, 300));
    }

    // Video search
    const videoSearch = document.getElementById('videoSearch');
    if (videoSearch) {
        videoSearch.addEventListener('input', debounce(() => {
            filterVideos(videoSearch.value);
        }, 300));
    }
}

/**
 * Initialize mobile menu
 */
function initMobileMenu() {
    const toggle = document.getElementById('mobileToggle');
    const menu = document.querySelector('.nav-menu');

    if (toggle && menu) {
        toggle.addEventListener('click', () => {
            menu.classList.toggle('active');
        });
    }

    // Logout button
    document.getElementById('logoutBtn').addEventListener('click', (e) => {
        e.preventDefault();
        logout();
    });
}

// ==============================================================================
// API HELPERS
// ==============================================================================

/**
 * Make authenticated API request
 */
async function apiRequest(endpoint, options = {}) {
    const token = localStorage.getItem('authToken');

    const config = {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
            ...options.headers
        }
    };

    try {
        const response = await fetch(`${ADMIN_API}${endpoint}`, config);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ==============================================================================
// DASHBOARD STATS
// ==============================================================================

/**
 * Load dashboard statistics
 */
async function loadDashboardStats() {
    try {
        const response = await apiRequest('/stats');
        const stats = response.data;

        document.getElementById('statTotalUsers').textContent = stats.total_users || 0;
        document.getElementById('statTotalVideos').textContent = stats.total_videos || 0;
        document.getElementById('statTotalCredits').textContent = formatNumber(stats.total_credits_issued || 0);
        document.getElementById('statActiveToday').textContent = stats.active_today || 0;
    } catch (error) {
        console.error('Failed to load stats:', error);
        showToast('Failed to load dashboard stats', 'error');
    }
}

/**
 * Format large numbers
 */
function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

// ==============================================================================
// USER MANAGEMENT
// ==============================================================================

/**
 * Load all users
 */
async function loadUsers() {
    const container = document.getElementById('usersTableContainer');
    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        const response = await apiRequest('/users');
        usersData = response.data.users || [];
        renderUsersTable(usersData);
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>Failed to load users: ${error.message}</p></div>`;
    }
}

/**
 * Refresh users list
 */
function refreshUsers() {
    loadUsers();
}

/**
 * Filter users by search term
 */
function filterUsers(searchTerm) {
    const filtered = usersData.filter(user => {
        const search = searchTerm.toLowerCase();
        return user.email.toLowerCase().includes(search) ||
               (user.name && user.name.toLowerCase().includes(search)) ||
               user.role.toLowerCase().includes(search);
    });
    renderUsersTable(filtered);
}

/**
 * Render users table
 */
function renderUsersTable(users) {
    const container = document.getElementById('usersTableContainer');

    if (!users || users.length === 0) {
        container.innerHTML = '<div class="empty-state"><i class="fas fa-users"></i><p>No users found</p></div>';
        return;
    }

    const html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>User</th>
                    <th>Role</th>
                    <th>Credits</th>
                    <th>Videos</th>
                    <th>Last Login</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${users.map(user => `
                    <tr>
                        <td>
                            <div class="user-cell">
                                <div class="user-avatar">${getInitials(user.email)}</div>
                                <div class="user-info">
                                    <span class="user-name">${user.name || 'No name'}</span>
                                    <span class="user-email">${user.email}</span>
                                </div>
                            </div>
                        </td>
                        <td><span class="role-badge ${user.role}">${user.role}</span></td>
                        <td>
                            <div class="credits-display">
                                <i class="fas fa-coins"></i>
                                <span>${user.credits || 0}</span>
                            </div>
                        </td>
                        <td>${user.video_count || 0}</td>
                        <td>${formatDate(user.last_login)}</td>
                        <td>
                            <span class="status-badge ${user.is_active ? 'active' : 'inactive'}">
                                ${user.is_active ? 'Active' : 'Inactive'}
                            </span>
                        </td>
                        <td>
                            <div class="action-btns">
                                <button class="action-btn view" onclick="viewUser('${user.id}')" title="View Details">
                                    <i class="fas fa-eye"></i>
                                </button>
                                <button class="action-btn edit" onclick="openRoleModal('${user.id}', '${user.email}', '${user.role}')" title="Change Role">
                                    <i class="fas fa-user-tag"></i>
                                </button>
                                <button class="action-btn credits" onclick="openCreditsModal('${user.id}', '${user.email}')" title="Add Credits">
                                    <i class="fas fa-coins"></i>
                                </button>
                                <button class="action-btn delete" onclick="openDeleteModal('${user.id}', 'user', '${user.email}')" title="Delete User">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

/**
 * View user details
 */
async function viewUser(userId) {
    const modal = document.getElementById('userDetailModal');
    const content = document.getElementById('userDetailContent');

    content.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';
    modal.classList.add('active');

    try {
        const response = await apiRequest(`/users/${userId}`);
        const { user, credits, credit_history, videos } = response.data;

        content.innerHTML = `
            <div class="user-detail-card">
                <div class="user-detail-header">
                    <div class="user-detail-avatar">${getInitials(user.email)}</div>
                    <div class="user-detail-info">
                        <h4>${user.name || user.email}</h4>
                        <p>${user.email}</p>
                        <span class="role-badge ${user.role}">${user.role}</span>
                    </div>
                </div>
                <div class="user-stats-grid">
                    <div class="user-stat-item">
                        <div class="value">${credits?.balance || 0}</div>
                        <div class="label">Credits</div>
                    </div>
                    <div class="user-stat-item">
                        <div class="value">${credits?.total_used || 0}</div>
                        <div class="label">Used</div>
                    </div>
                    <div class="user-stat-item">
                        <div class="value">${videos?.length || 0}</div>
                        <div class="label">Videos</div>
                    </div>
                </div>
            </div>

            <h4 style="margin-bottom: var(--space-sm);">Recent Credit History</h4>
            <div class="credit-history">
                ${credit_history && credit_history.length > 0 ? credit_history.map(h => `
                    <div class="credit-history-item">
                        <div>
                            <div>${h.description || h.type}</div>
                            <small style="color: var(--gray);">${formatDate(h.created_at)}</small>
                        </div>
                        <span class="credit-amount ${h.amount > 0 ? 'positive' : 'negative'}">
                            ${h.amount > 0 ? '+' : ''}${h.amount}
                        </span>
                    </div>
                `).join('') : '<p style="color: var(--gray); text-align: center;">No credit history</p>'}
            </div>
        `;
    } catch (error) {
        content.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>${error.message}</p></div>`;
    }
}

/**
 * Get initials from email
 */
function getInitials(email) {
    if (!email) return '?';
    const parts = email.split('@')[0].split(/[._-]/);
    if (parts.length >= 2) {
        return (parts[0][0] + parts[1][0]).toUpperCase();
    }
    return email.substring(0, 2).toUpperCase();
}

/**
 * Format date
 */
function formatDate(dateStr) {
    if (!dateStr) return 'Never';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;

    // Less than 24 hours
    if (diff < 86400000) {
        const hours = Math.floor(diff / 3600000);
        if (hours < 1) {
            const mins = Math.floor(diff / 60000);
            return mins <= 1 ? 'Just now' : `${mins}m ago`;
        }
        return `${hours}h ago`;
    }

    // Less than 7 days
    if (diff < 604800000) {
        const days = Math.floor(diff / 86400000);
        return `${days}d ago`;
    }

    // Format as date
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

// ==============================================================================
// VIDEO MANAGEMENT
// ==============================================================================

/**
 * Load all videos
 */
async function loadVideos() {
    const container = document.getElementById('videosContainer');
    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        const response = await apiRequest('/videos');
        videosData = response.data.videos || [];
        renderVideosGrid(videosData);
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>Failed to load videos: ${error.message}</p></div>`;
    }
}

/**
 * Refresh videos list
 */
function refreshVideos() {
    loadVideos();
}

/**
 * Filter videos by search term
 */
function filterVideos(searchTerm) {
    const filtered = videosData.filter(video => {
        const search = searchTerm.toLowerCase();
        return (video.prompt && video.prompt.toLowerCase().includes(search)) ||
               (video.user_email && video.user_email.toLowerCase().includes(search)) ||
               (video.status && video.status.toLowerCase().includes(search));
    });
    renderVideosGrid(filtered);
}

/**
 * Render videos grid
 */
function renderVideosGrid(videos) {
    const container = document.getElementById('videosContainer');

    if (!videos || videos.length === 0) {
        container.innerHTML = '<div class="empty-state"><i class="fas fa-video"></i><p>No videos found</p></div>';
        return;
    }

    const html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Preview</th>
                    <th>Prompt</th>
                    <th>User</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${videos.map(video => {
                    // Determine status display (show if deleted/expired)
                    const isDeleted = video.deleted_at !== null;
                    const statusText = isDeleted ? 'expired' : video.status;
                    const statusClass = isDeleted ? 'expired' : video.status;

                    // Use thumbnail URL for preview
                    const thumbnailUrl = video.thumbnail_url || `/api/v1/videos/${video.id}/thumbnail.jpg`;
                    const videoUrl = video.video_url || video.output_url;

                    return `
                    <tr style="${isDeleted ? 'opacity: 0.6;' : ''}">
                        <td>
                            <div style="width: 80px; height: 45px; background: var(--dark); border-radius: 4px; overflow: hidden; position: relative;">
                                <img src="${thumbnailUrl}"
                                     style="width: 100%; height: 100%; object-fit: cover;"
                                     onerror="this.style.display='none'; this.parentElement.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;width:100%;height:100%;color:var(--gray);\\'><i class=\\'fas fa-film\\'></i></div>';"
                                     alt="Video thumbnail">
                                ${isDeleted ? '<div style="position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;color:#ff6b6b;font-size:10px;">EXPIRED</div>' : ''}
                            </div>
                        </td>
                        <td>
                            <div style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${(video.prompt || video.script || 'No prompt').replace(/"/g, '&quot;')}">
                                ${video.prompt || video.script || 'No prompt'}
                            </div>
                        </td>
                        <td>
                            <span style="font-size: var(--font-size-sm);">${video.user_email || 'Unknown'}</span>
                        </td>
                        <td>
                            <span class="status-badge ${statusClass}">${statusText}</span>
                        </td>
                        <td>${formatDate(video.created_at)}</td>
                        <td>
                            <div class="action-btns">
                                ${videoUrl && !isDeleted ? `
                                    <button class="action-btn view" onclick="window.open('${videoUrl}', '_blank')" title="View Video">
                                        <i class="fas fa-external-link-alt"></i>
                                    </button>
                                ` : ''}
                                ${!isDeleted ? `
                                    <button class="action-btn delete" onclick="openDeleteModal('${video.id}', 'video', 'this video')" title="Delete Video">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                ` : '<span style="color: var(--gray); font-size: 11px;">Deleted</span>'}
                            </div>
                        </td>
                    </tr>
                `}).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

// ==============================================================================
// CREDITS MANAGEMENT
// ==============================================================================

/**
 * Load credits overview
 */
async function loadCreditsOverview() {
    const container = document.getElementById('creditsTableContainer');
    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        const response = await apiRequest('/users');
        const users = response.data.users || [];

        // Sort by credits descending
        users.sort((a, b) => (b.credits || 0) - (a.credits || 0));

        renderCreditsTable(users);
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>Failed to load credits: ${error.message}</p></div>`;
    }
}

/**
 * Render credits table
 */
function renderCreditsTable(users) {
    const container = document.getElementById('creditsTableContainer');

    if (!users || users.length === 0) {
        container.innerHTML = '<div class="empty-state"><i class="fas fa-coins"></i><p>No users found</p></div>';
        return;
    }

    const html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>User</th>
                    <th>Balance</th>
                    <th>Total Used</th>
                    <th>Total Given</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${users.map(user => `
                    <tr>
                        <td>
                            <div class="user-cell">
                                <div class="user-avatar">${getInitials(user.email)}</div>
                                <div class="user-info">
                                    <span class="user-name">${user.email}</span>
                                    <span class="role-badge ${user.role}" style="font-size: 10px;">${user.role}</span>
                                </div>
                            </div>
                        </td>
                        <td>
                            <div class="credits-display">
                                <i class="fas fa-coins"></i>
                                <span style="font-weight: 700; font-size: 1.1rem;">${user.credits || 0}</span>
                            </div>
                        </td>
                        <td style="color: var(--danger);">${user.credits_used || 0}</td>
                        <td style="color: var(--success);">${user.credits_given || 0}</td>
                        <td>
                            <button class="btn btn-primary" style="padding: 6px 12px; font-size: 12px;" onclick="openCreditsModal('${user.id}', '${user.email}')">
                                <i class="fas fa-plus"></i> Add
                            </button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

// ==============================================================================
// MODAL HANDLERS
// ==============================================================================

/**
 * Open add credits modal
 */
function openCreditsModal(userId, userEmail) {
    document.getElementById('creditsUserId').value = userId;
    document.getElementById('creditsUserDisplay').value = userEmail;
    document.getElementById('creditsAmount').value = 100;
    document.getElementById('creditsReason').value = 'Admin credit grant';
    document.getElementById('addCreditsModal').classList.add('active');
}

/**
 * Submit add credits
 */
async function submitAddCredits() {
    const userId = document.getElementById('creditsUserId').value;
    const amount = parseInt(document.getElementById('creditsAmount').value);
    const reason = document.getElementById('creditsReason').value;

    if (!amount || amount < 1) {
        showToast('Please enter a valid amount', 'error');
        return;
    }

    try {
        await apiRequest(`/users/${userId}/credits`, {
            method: 'POST',
            body: JSON.stringify({ amount, reason })
        });

        showToast(`Added ${amount} credits successfully`, 'success');
        closeModal('addCreditsModal');

        // Refresh data
        loadDashboardStats();
        loadUsers();
        if (document.getElementById('creditsPanel').classList.contains('active')) {
            loadCreditsOverview();
        }
    } catch (error) {
        showToast(error.message, 'error');
    }
}

/**
 * Open role change modal
 */
function openRoleModal(userId, userEmail, currentRole) {
    document.getElementById('roleUserId').value = userId;
    document.getElementById('roleUserDisplay').value = userEmail;
    document.getElementById('newRole').value = currentRole;
    document.getElementById('changeRoleModal').classList.add('active');
}

/**
 * Submit role change
 */
async function submitChangeRole() {
    const userId = document.getElementById('roleUserId').value;
    const newRole = document.getElementById('newRole').value;

    try {
        await apiRequest(`/users/${userId}/role`, {
            method: 'PUT',
            body: JSON.stringify({ role: newRole })
        });

        showToast(`Role updated to ${newRole}`, 'success');
        closeModal('changeRoleModal');
        loadUsers();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

/**
 * Open delete confirmation modal
 */
function openDeleteModal(itemId, itemType, displayName) {
    document.getElementById('deleteItemId').value = itemId;
    document.getElementById('deleteItemType').value = itemType;
    document.getElementById('deleteConfirmText').textContent =
        `Are you sure you want to delete ${displayName}? This action cannot be undone.`;
    document.getElementById('confirmDeleteModal').classList.add('active');
}

/**
 * Confirm and execute delete
 */
async function confirmDelete() {
    const itemId = document.getElementById('deleteItemId').value;
    const itemType = document.getElementById('deleteItemType').value;

    try {
        if (itemType === 'user') {
            await apiRequest(`/users/${itemId}`, { method: 'DELETE' });
            showToast('User deleted successfully', 'success');
            loadUsers();
            loadDashboardStats();
        } else if (itemType === 'video') {
            await apiRequest(`/videos/${itemId}`, { method: 'DELETE' });
            showToast('Video deleted successfully', 'success');
            loadVideos();
            loadDashboardStats();
        }

        closeModal('confirmDeleteModal');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

/**
 * Open bulk credits modal (placeholder for future)
 */
function openBulkCreditsModal() {
    showToast('Bulk credits feature coming soon!', 'info');
}

/**
 * Close modal by ID
 */
function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Remove existing toasts
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 9999;
        animation: slideIn 0.3s ease;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;

    // Set background based on type
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };
    toast.style.background = colors[type] || colors.info;

    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}" style="margin-right: 8px;"></i>
        ${message}
    `;

    document.body.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Add toast animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

/**
 * Logout function
 */
function logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = 'auth.html';
}

// ==============================================================================
// ANALYTICS
// ==============================================================================

// Chart instances for cleanup
let userGrowthChart = null;
let videoTrendChart = null;
let creditTrendChart = null;
let roleDistChart = null;

/**
 * Load analytics data and render charts
 */
async function loadAnalytics() {
    const days = document.getElementById('analyticsDays')?.value || 30;
    const container = document.getElementById('topUsersContainer');
    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        const response = await apiRequest(`/analytics?days=${days}`);
        const data = response.data;

        // Render charts
        renderUserGrowthChart(data.user_growth || []);
        renderVideoTrendChart(data.video_trend || []);
        renderCreditTrendChart(data.credit_trend || []);
        renderRoleDistChart(data.role_distribution || {});

        // Render top users
        renderTopUsers(data.top_users || []);
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>Failed to load analytics: ${error.message}</p></div>`;
    }
}

/**
 * Render user growth chart
 */
function renderUserGrowthChart(data) {
    const canvas = document.getElementById('userGrowthChart');
    if (!canvas) return;

    // Show message if insufficient data
    if (!data || data.length < 2) {
        canvas.parentElement.innerHTML = `
            <h4>User Growth</h4>
            <div style="display: flex; align-items: center; justify-content: center; height: 200px; color: var(--gray);">
                <div style="text-align: center;">
                    <i class="fas fa-chart-line" style="font-size: 2rem; margin-bottom: 8px; opacity: 0.5;"></i>
                    <p>Not enough data yet</p>
                    <small>Chart will appear as users sign up</small>
                </div>
            </div>`;
        return;
    }

    const ctx = canvas.getContext('2d');
    if (userGrowthChart) userGrowthChart.destroy();

    const labels = data.map(d => d.date);
    const values = data.map(d => d.count);

    userGrowthChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'New Users',
                data: values,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { stepSize: 1 } }
            }
        }
    });
}

/**
 * Render video trend chart
 */
function renderVideoTrendChart(data) {
    const canvas = document.getElementById('videoTrendChart');
    if (!canvas) return;

    // Show message if insufficient data
    if (!data || data.length < 2) {
        canvas.parentElement.innerHTML = `
            <h4>Videos Created</h4>
            <div style="display: flex; align-items: center; justify-content: center; height: 200px; color: var(--gray);">
                <div style="text-align: center;">
                    <i class="fas fa-video" style="font-size: 2rem; margin-bottom: 8px; opacity: 0.5;"></i>
                    <p>Not enough data yet</p>
                    <small>Chart will appear as videos are created</small>
                </div>
            </div>`;
        return;
    }

    const ctx = canvas.getContext('2d');
    if (videoTrendChart) videoTrendChart.destroy();

    const labels = data.map(d => d.date);
    const values = data.map(d => d.count);

    videoTrendChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Videos Created',
                data: values,
                backgroundColor: '#10b981',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { stepSize: 1 } }
            }
        }
    });
}

/**
 * Render credit trend chart
 */
function renderCreditTrendChart(data) {
    const canvas = document.getElementById('creditTrendChart');
    if (!canvas) return;

    // Show message if insufficient data
    if (!data || data.length < 2) {
        canvas.parentElement.innerHTML = `
            <h4>Credit Usage</h4>
            <div style="display: flex; align-items: center; justify-content: center; height: 200px; color: var(--gray);">
                <div style="text-align: center;">
                    <i class="fas fa-coins" style="font-size: 2rem; margin-bottom: 8px; opacity: 0.5;"></i>
                    <p>Not enough data yet</p>
                    <small>Chart will appear as credits are used</small>
                </div>
            </div>`;
        return;
    }

    const ctx = canvas.getContext('2d');
    if (creditTrendChart) creditTrendChart.destroy();

    const labels = data.map(d => d.date);
    const values = data.map(d => d.credits);

    creditTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Credits Used',
                data: values,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

/**
 * Render role distribution chart
 */
function renderRoleDistChart(data) {
    const canvas = document.getElementById('roleDistChart');
    if (!canvas) return;

    const labels = Object.keys(data || {});
    const values = Object.values(data || {});

    // Show message if no data
    if (!labels.length || values.every(v => v === 0)) {
        canvas.parentElement.innerHTML = `
            <h4>User Roles</h4>
            <div style="display: flex; align-items: center; justify-content: center; height: 200px; color: var(--gray);">
                <div style="text-align: center;">
                    <i class="fas fa-users" style="font-size: 2rem; margin-bottom: 8px; opacity: 0.5;"></i>
                    <p>Not enough data yet</p>
                    <small>Chart will appear as users register</small>
                </div>
            </div>`;
        return;
    }

    const ctx = canvas.getContext('2d');
    if (roleDistChart) roleDistChart.destroy();

    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

    roleDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.slice(0, labels.length)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right' }
            }
        }
    });
}

/**
 * Render top users list
 */
function renderTopUsers(users) {
    const container = document.getElementById('topUsersContainer');

    if (!users || users.length === 0) {
        container.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 150px; color: var(--gray);">
                <div style="text-align: center;">
                    <i class="fas fa-trophy" style="font-size: 2rem; margin-bottom: 8px; opacity: 0.5;"></i>
                    <p>No top users yet</p>
                    <small>Leaderboard will appear as users create videos</small>
                </div>
            </div>`;
        return;
    }

    container.innerHTML = users.map((user, index) => `
        <div class="top-user-item">
            <div style="display: flex; align-items: center; gap: var(--space-sm);">
                <span style="font-weight: 700; color: ${index < 3 ? '#f59e0b' : 'var(--gray)'};">#${index + 1}</span>
                <div class="user-avatar" style="width: 32px; height: 32px; font-size: 12px;">${getInitials(user.email)}</div>
                <span>${user.email}</span>
            </div>
            <div style="display: flex; align-items: center; gap: var(--space-md);">
                <span style="color: var(--gray);"><i class="fas fa-video"></i> ${user.video_count || 0}</span>
                <span style="color: #f59e0b;"><i class="fas fa-coins"></i> ${user.credits || 0}</span>
            </div>
        </div>
    `).join('');
}

// ==============================================================================
// SETTINGS
// ==============================================================================

/**
 * Load system settings
 */
async function loadSettings() {
    try {
        const response = await apiRequest('/settings');
        const settings = response.data || {};

        // Update form values - settings is {key: {value: '...', description: '...'}}
        Object.keys(settings).forEach(key => {
            const element = document.getElementById(`setting_${key}`);
            if (element) {
                // Extract the actual value from the settings object
                const settingValue = settings[key]?.value ?? settings[key];
                if (element.tagName === 'SELECT') {
                    element.value = String(settingValue);
                } else {
                    element.value = settingValue;
                }
            }
        });

        showToast('Settings loaded', 'success');
    } catch (error) {
        showToast('Failed to load settings: ' + error.message, 'error');
    }
}

/**
 * Update a system setting
 */
async function updateSetting(key) {
    const element = document.getElementById(`setting_${key}`);
    if (!element) return;

    let value = element.value;

    // Convert boolean strings
    if (value === 'true') value = true;
    else if (value === 'false') value = false;
    // Convert numbers
    else if (!isNaN(value) && value !== '') value = Number(value);

    try {
        await apiRequest(`/settings/${key}`, {
            method: 'PUT',
            body: JSON.stringify({ value })
        });

        showToast(`Setting "${key}" updated successfully`, 'success');
    } catch (error) {
        showToast('Failed to update setting: ' + error.message, 'error');
    }
}

// ==============================================================================
// NOTIFICATIONS / ANNOUNCEMENTS
// ==============================================================================

/**
 * Load notifications/announcements
 */
async function loadNotifications() {
    const container = document.getElementById('notificationsContainer');
    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        const response = await apiRequest('/notifications');
        // API returns {notifications: [...], total: X}
        const notifications = response.data?.notifications || response.data || [];

        if (notifications.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-bell-slash"></i><p>No announcements yet</p></div>';
            return;
        }

        container.innerHTML = notifications.map(notif => `
            <div class="notification-item">
                <div class="notif-icon ${notif.type || 'info'}">
                    <i class="fas fa-${getNotifIcon(notif.type)}"></i>
                </div>
                <div class="notif-content">
                    <div class="notif-title">${notif.title}</div>
                    <div class="notif-message">${notif.message}</div>
                    <div class="notif-meta">
                        <span><i class="fas fa-user"></i> ${notif.admin_email || 'System'}</span>
                        <span style="margin-left: var(--space-md);"><i class="fas fa-clock"></i> ${formatDate(notif.created_at)}</span>
                        ${notif.expires_at ? `<span style="margin-left: var(--space-md);"><i class="fas fa-hourglass"></i> Expires: ${formatDate(notif.expires_at)}</span>` : ''}
                    </div>
                </div>
                <button class="action-btn delete" onclick="deleteNotification('${notif.id}')" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>${error.message}</p></div>`;
    }
}

/**
 * Get notification icon based on type
 */
function getNotifIcon(type) {
    const icons = {
        info: 'info-circle',
        warning: 'exclamation-triangle',
        success: 'check-circle',
        error: 'times-circle'
    };
    return icons[type] || icons.info;
}

/**
 * Open notification creation modal
 */
function openNotificationModal() {
    document.getElementById('notifTitle').value = '';
    document.getElementById('notifMessage').value = '';
    document.getElementById('notifType').value = 'info';
    document.getElementById('notifExpires').value = '';
    document.getElementById('createNotificationModal').classList.add('active');
}

/**
 * Submit new notification
 */
async function submitNotification() {
    const title = document.getElementById('notifTitle').value.trim();
    const message = document.getElementById('notifMessage').value.trim();
    const type = document.getElementById('notifType').value;
    const expires = document.getElementById('notifExpires').value;

    if (!title || !message) {
        showToast('Please fill in title and message', 'error');
        return;
    }

    try {
        await apiRequest('/notifications', {
            method: 'POST',
            body: JSON.stringify({
                title,
                message,
                type,
                expires_at: expires || null
            })
        });

        showToast('Announcement created successfully', 'success');
        closeModal('createNotificationModal');
        loadNotifications();
    } catch (error) {
        showToast('Failed to create announcement: ' + error.message, 'error');
    }
}

/**
 * Delete notification
 */
async function deleteNotification(notifId) {
    if (!confirm('Delete this announcement?')) return;

    try {
        await apiRequest(`/notifications/${notifId}`, { method: 'DELETE' });
        showToast('Announcement deleted', 'success');
        loadNotifications();
    } catch (error) {
        showToast('Failed to delete: ' + error.message, 'error');
    }
}

// ==============================================================================
// SUPPORT TICKETS
// ==============================================================================

/**
 * Load support tickets
 */
async function loadTickets() {
    const container = document.getElementById('ticketsContainer');
    const statsContainer = document.getElementById('ticketStats');
    const statusFilter = document.getElementById('ticketStatusFilter')?.value || '';

    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        let url = '/tickets';
        if (statusFilter) url += `?status=${statusFilter}`;

        const response = await apiRequest(url);
        const tickets = response.data?.tickets || [];
        const stats = response.data?.stats || {};

        // Render stats
        statsContainer.innerHTML = `
            <div class="ticket-stat open">Open: ${stats.open || 0}</div>
            <div class="ticket-stat in_progress">In Progress: ${stats.in_progress || 0}</div>
            <div class="ticket-stat resolved">Resolved: ${stats.resolved || 0}</div>
            <div class="ticket-stat urgent">High Priority: ${stats.high_priority || 0}</div>
        `;

        if (tickets.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-ticket-alt"></i><p>No tickets found</p></div>';
            return;
        }

        container.innerHTML = tickets.map(ticket => `
            <div class="ticket-item" onclick="openTicketDetail('${ticket.id}')">
                <div class="ticket-header">
                    <span class="ticket-subject">${ticket.subject}</span>
                    <div>
                        <span class="status-badge ${ticket.status}">${ticket.status.replace('_', ' ')}</span>
                        ${ticket.priority === 'high' ? '<span class="status-badge failed" style="margin-left: 4px;">High</span>' : ''}
                    </div>
                </div>
                <div class="ticket-meta">
                    <span><i class="fas fa-user"></i> ${ticket.user_email}</span>
                    <span style="margin-left: var(--space-md);"><i class="fas fa-clock"></i> ${formatDate(ticket.created_at)}</span>
                    <span style="margin-left: var(--space-md);"><i class="fas fa-comments"></i> ${ticket.reply_count || 0} replies</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>${error.message}</p></div>`;
    }
}

/**
 * Open ticket detail modal
 */
async function openTicketDetail(ticketId) {
    const modal = document.getElementById('ticketDetailModal');
    const content = document.getElementById('ticketDetailContent');

    content.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';
    modal.classList.add('active');

    try {
        const response = await apiRequest(`/tickets/${ticketId}`);
        const ticket = response.data;

        content.innerHTML = `
            <div style="margin-bottom: var(--space-lg);">
                <h4 style="margin-bottom: var(--space-xs);">${ticket.subject}</h4>
                <div style="display: flex; gap: var(--space-md); margin-bottom: var(--space-md);">
                    <span class="status-badge ${ticket.status}">${ticket.status.replace('_', ' ')}</span>
                    <span style="color: var(--gray);"><i class="fas fa-user"></i> ${ticket.user_email}</span>
                    <span style="color: var(--gray);"><i class="fas fa-clock"></i> ${formatDate(ticket.created_at)}</span>
                </div>
                <p style="background: var(--light); padding: var(--space-md); border-radius: var(--radius-md);">${ticket.message}</p>
            </div>

            <div style="margin-bottom: var(--space-lg);">
                <h5>Replies</h5>
                <div id="ticketReplies" style="max-height: 200px; overflow-y: auto;">
                    ${ticket.replies && ticket.replies.length > 0 ? ticket.replies.map(reply => `
                        <div style="padding: var(--space-sm); background: ${reply.is_admin ? 'rgba(59, 130, 246, 0.1)' : 'var(--light)'}; border-radius: var(--radius-md); margin-bottom: var(--space-sm);">
                            <div style="font-weight: 600; font-size: var(--font-size-sm);">
                                ${reply.is_admin ? '<i class="fas fa-shield-halved"></i> Admin' : '<i class="fas fa-user"></i> User'} - ${formatDate(reply.created_at)}
                            </div>
                            <p style="margin: var(--space-xs) 0 0;">${reply.message}</p>
                        </div>
                    `).join('') : '<p style="color: var(--gray);">No replies yet</p>'}
                </div>
            </div>

            <div style="display: flex; gap: var(--space-sm); margin-bottom: var(--space-md);">
                <select id="ticketStatus" style="flex: 1;">
                    <option value="open" ${ticket.status === 'open' ? 'selected' : ''}>Open</option>
                    <option value="in_progress" ${ticket.status === 'in_progress' ? 'selected' : ''}>In Progress</option>
                    <option value="resolved" ${ticket.status === 'resolved' ? 'selected' : ''}>Resolved</option>
                    <option value="closed" ${ticket.status === 'closed' ? 'selected' : ''}>Closed</option>
                </select>
                <button class="btn btn-primary" onclick="updateTicketStatus('${ticket.id}')">Update Status</button>
            </div>

            <div class="form-group">
                <textarea id="ticketReplyText" rows="3" placeholder="Write a reply..."></textarea>
            </div>
            <button class="btn btn-primary" onclick="submitTicketReply('${ticket.id}')">
                <i class="fas fa-paper-plane"></i> Send Reply
            </button>
        `;
    } catch (error) {
        content.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>${error.message}</p></div>`;
    }
}

/**
 * Update ticket status
 */
async function updateTicketStatus(ticketId) {
    const status = document.getElementById('ticketStatus').value;

    try {
        await apiRequest(`/tickets/${ticketId}`, {
            method: 'PUT',
            body: JSON.stringify({ status })
        });

        showToast('Ticket status updated', 'success');
        loadTickets();
    } catch (error) {
        showToast('Failed to update status: ' + error.message, 'error');
    }
}

/**
 * Submit ticket reply
 */
async function submitTicketReply(ticketId) {
    const message = document.getElementById('ticketReplyText').value.trim();

    if (!message) {
        showToast('Please enter a reply', 'error');
        return;
    }

    try {
        await apiRequest(`/tickets/${ticketId}/reply`, {
            method: 'POST',
            body: JSON.stringify({ message })
        });

        showToast('Reply sent', 'success');
        openTicketDetail(ticketId); // Refresh the detail view
    } catch (error) {
        showToast('Failed to send reply: ' + error.message, 'error');
    }
}

// ==============================================================================
// AUDIT LOGS
// ==============================================================================

/**
 * Load audit logs
 */
async function loadAuditLogs() {
    const container = document.getElementById('logsContainer');
    const filter = document.getElementById('logActionFilter')?.value || '';

    container.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i></div>';

    try {
        let url = '/audit-logs?limit=100';
        if (filter) url += `&action=${encodeURIComponent(filter)}`;

        const response = await apiRequest(url);
        // API returns {logs: [...], total: X, ...}
        const logs = response.data?.logs || response.data || [];

        if (logs.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-history"></i><p>No audit logs found</p></div>';
            return;
        }

        container.innerHTML = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Action</th>
                        <th>Admin</th>
                        <th>Target</th>
                        <th>Details</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    ${logs.map(log => `
                        <tr>
                            <td><span style="font-weight: 600;">${log.action}</span></td>
                            <td>${log.admin_email || 'System'}</td>
                            <td>${log.target_type ? `${log.target_type}: ${log.target_id || '-'}` : '-'}</td>
                            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${log.details || '-'}</td>
                            <td style="white-space: nowrap;">${formatDate(log.created_at)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>${error.message}</p></div>`;
    }
}

// ==============================================================================
// DATA EXPORT
// ==============================================================================

/**
 * Export data to CSV
 */
async function exportData(type) {
    try {
        showToast(`Exporting ${type} data...`, 'info');

        const response = await apiRequest(`/export/${type}`);
        const data = response.data || [];

        if (!data.length) {
            showToast('No data to export', 'warning');
            return;
        }

        // Convert JSON to CSV
        const headers = Object.keys(data[0]);
        const csvRows = [
            headers.join(','),
            ...data.map(row => headers.map(h => {
                let val = row[h];
                // Escape values with commas or quotes
                if (val === null || val === undefined) val = '';
                val = String(val).replace(/"/g, '""');
                if (val.includes(',') || val.includes('"') || val.includes('\n')) {
                    val = `"${val}"`;
                }
                return val;
            }).join(','))
        ];
        const csvContent = csvRows.join('\n');

        // Download as file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${type}_export_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        showToast(`${type} data exported successfully (${data.length} records)`, 'success');
    } catch (error) {
        showToast('Export failed: ' + error.message, 'error');
    }
}

// ==============================================================================
// KEYBOARD SHORTCUTS
// ==============================================================================

document.addEventListener('keydown', (e) => {
    // ESC to close modals
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }
});

// Close modal on overlay click
document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            overlay.classList.remove('active');
        }
    });
});
