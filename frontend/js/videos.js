/**
 * ReelSense AI - My Videos Page
 * Displays user's generated videos with expiration countdown
 */

// ==============================================================================
// CONFIGURATION
// ==============================================================================

const API_BASE = window.APP_CONFIG?.API_BASE || 'http://localhost:5000/api/v1';
let currentUser = null;
let videosData = [];

// ==============================================================================
// INITIALIZATION
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    initNavigation();
});

/**
 * Check authentication status
 */
function checkAuth() {
    const token = localStorage.getItem('authToken');
    const user = JSON.parse(localStorage.getItem('user') || 'null');

    if (!token || !user) {
        window.location.href = 'auth.html?redirect=videos';
        return;
    }

    currentUser = user;
    updateUserUI();
    loadVideos();
    loadRetentionSetting();
}

/**
 * Update user interface with user info
 */
function updateUserUI() {
    // Hide login, show user menu
    document.getElementById('loginBtn').style.display = 'none';
    document.getElementById('userMenu').style.display = 'block';
    document.getElementById('navCredits').style.display = 'flex';

    // Set user info
    const initials = getInitials(currentUser.email);
    document.getElementById('userInitials').textContent = initials;
    document.getElementById('userName').textContent = currentUser.name || currentUser.email.split('@')[0];
    document.getElementById('userEmail').textContent = currentUser.email;

    // Show admin link if admin
    if (['admin', 'superadmin'].includes(currentUser.role)) {
        document.getElementById('adminLink').style.display = 'block';
        document.getElementById('adminLink').href = 'admin.html';
    }

    // Load credits
    loadCredits();
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
 * Load user credits
 */
async function loadCredits() {
    try {
        const response = await apiRequest('/auth/user/credits');
        if (response.data) {
            document.getElementById('creditBalance').textContent = response.data.balance || 0;
        }
    } catch (error) {
        console.error('Failed to load credits:', error);
    }
}

/**
 * Load retention setting
 */
async function loadRetentionSetting() {
    // Retention will be set from loadVideos API response
    // Default to 3 days initially
    const retentionEl = document.getElementById('retentionDays');
    if (retentionEl && retentionEl.textContent === '3') {
        // Already set or will be updated by loadVideos
    }
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

    const response = await fetch(`${API_BASE}${endpoint}`, config);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || 'API request failed');
    }

    return data;
}

// ==============================================================================
// LOAD VIDEOS
// ==============================================================================

/**
 * Load user's videos
 */
async function loadVideos() {
    const container = document.getElementById('videosContainer');

    try {
        const response = await apiRequest('/auth/user/videos');
        videosData = response.data?.videos || [];

        // Update retention days from API
        const retentionDays = response.data?.retention_days || 3;
        const retentionEl = document.getElementById('retentionDays');
        if (retentionEl) {
            retentionEl.textContent = retentionDays;
        }

        // Update stats
        updateStats(videosData);

        // Render videos
        renderVideos(videosData);

    } catch (error) {
        console.error('Failed to load videos:', error);
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-circle"></i>
                <h3>Failed to load videos</h3>
                <p>${error.message}</p>
                <button class="btn btn-primary" onclick="loadVideos()">
                    <i class="fas fa-refresh"></i> Try Again
                </button>
            </div>
        `;
    }
}

/**
 * Update video stats
 */
function updateStats(videos) {
    document.getElementById('totalVideos').textContent = videos.length;

    // Count expiring today
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const expiringToday = videos.filter(v => {
        if (!v.expires_at) return false;
        const expiry = new Date(v.expires_at);
        return expiry <= tomorrow;
    }).length;

    document.getElementById('expiringToday').textContent = expiringToday;
}

/**
 * Render videos grid
 */
function renderVideos(videos) {
    const container = document.getElementById('videosContainer');

    if (!videos || videos.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-video-slash"></i>
                <h3>No videos yet</h3>
                <p>Your generated videos will appear here</p>
                <a href="index.html#generator" class="btn btn-primary">
                    <i class="fas fa-plus"></i> Create Your First Video
                </a>
            </div>
        `;
        return;
    }

    // Sort by creation date (newest first)
    videos.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    container.innerHTML = `
        <div class="videos-grid">
            ${videos.map(video => renderVideoCard(video)).join('')}
        </div>
    `;
}

/**
 * Render single video card
 */
function renderVideoCard(video) {
    const expiryInfo = getExpiryInfo(video.expires_at);
    const createdDate = formatDate(video.created_at);
    const videoUrl = video.video_url || '';
    // Use thumbnail endpoint - this generates thumbnails on-demand if missing
    const thumbnailUrl = video.thumbnail_url || `/api/v1/videos/${video.id}/thumbnail.jpg`;

    return `
        <div class="video-card" data-id="${video.id}">
            <div class="video-thumbnail" onclick="playVideo('${videoUrl}')">
                <img src="${thumbnailUrl}"
                     alt="Video thumbnail"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                     style="width: 100%; height: 100%; object-fit: cover;">
                <div class="thumbnail-fallback" style="display: none; width: 100%; height: 100%; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
                    <i class="fas fa-film" style="font-size: 3rem; color: rgba(255,255,255,0.5);"></i>
                </div>
                <div class="play-overlay">
                    <i class="fas fa-play-circle"></i>
                </div>
            </div>
            <div class="video-info">
                <p class="video-prompt">${escapeHtml(video.prompt || 'No description')}</p>
                <div class="video-meta">
                    <span><i class="fas fa-clock"></i> ${createdDate}</span>
                    <span>${video.duration || 5}s</span>
                </div>
                <div class="video-expiry ${expiryInfo.class}">
                    <i class="fas fa-hourglass-half"></i>
                    <span>${expiryInfo.text}</span>
                </div>
                <div class="video-actions">
                    <button class="btn btn-download" onclick="downloadVideo('${video.id}', '${videoUrl}')">
                        <i class="fas fa-download"></i> Download
                    </button>
                    <button class="btn btn-delete" onclick="deleteVideo('${video.id}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
}

/**
 * Get expiry information
 */
function getExpiryInfo(expiresAt) {
    if (!expiresAt) {
        return { text: 'No expiry set', class: 'expiring-later' };
    }

    const now = new Date();
    const expiry = new Date(expiresAt);
    const diff = expiry - now;

    if (diff <= 0) {
        return { text: 'Expired', class: 'expiring-soon' };
    }

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;

    if (days === 0) {
        if (hours <= 6) {
            return { text: `Expires in ${hours}h`, class: 'expiring-soon' };
        }
        return { text: `Expires in ${hours}h`, class: 'expiring-medium' };
    }

    if (days === 1) {
        return { text: `Expires in 1d ${remainingHours}h`, class: 'expiring-medium' };
    }

    return { text: `Expires in ${days}d ${remainingHours}h`, class: 'expiring-later' };
}

/**
 * Format date
 */
function formatDate(dateStr) {
    if (!dateStr) return 'Unknown';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;

    if (diff < 86400000) { // Less than 24 hours
        const hours = Math.floor(diff / 3600000);
        if (hours < 1) {
            const mins = Math.floor(diff / 60000);
            return mins <= 1 ? 'Just now' : `${mins}m ago`;
        }
        return `${hours}h ago`;
    }

    if (diff < 604800000) { // Less than 7 days
        const days = Math.floor(diff / 86400000);
        return `${days}d ago`;
    }

    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==============================================================================
// VIDEO ACTIONS
// ==============================================================================

/**
 * Play video in modal
 */
function playVideo(videoUrl) {
    if (!videoUrl) {
        showToast('Video not available', 'error');
        return;
    }

    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');

    video.src = videoUrl;
    modal.classList.add('active');
    video.play();
}

/**
 * Close video modal
 */
function closeVideoModal() {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');

    video.pause();
    video.src = '';
    modal.classList.remove('active');
}

/**
 * Download video
 */
async function downloadVideo(videoId, videoUrl) {
    if (!videoUrl) {
        showToast('Video not available for download', 'error');
        return;
    }

    try {
        showToast('Starting download...', 'info');

        const response = await fetch(videoUrl);
        const blob = await response.blob();

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `reelsense_video_${videoId}.mp4`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        showToast('Download started!', 'success');
    } catch (error) {
        console.error('Download failed:', error);
        showToast('Download failed', 'error');
    }
}

/**
 * Delete video
 */
async function deleteVideo(videoId) {
    if (!confirm('Are you sure you want to delete this video? This cannot be undone.')) {
        return;
    }

    try {
        await apiRequest(`/auth/user/videos/${videoId}`, { method: 'DELETE' });
        showToast('Video deleted', 'success');
        loadVideos(); // Refresh list
    } catch (error) {
        console.error('Delete failed:', error);
        showToast('Failed to delete video: ' + error.message, 'error');
    }
}

// ==============================================================================
// NAVIGATION
// ==============================================================================

/**
 * Initialize navigation
 */
function initNavigation() {
    // Mobile menu toggle
    const toggle = document.getElementById('mobileToggle');
    const menu = document.getElementById('navMenu');

    if (toggle && menu) {
        toggle.addEventListener('click', () => {
            menu.classList.toggle('active');
        });
    }

    // User menu dropdown
    const userMenuBtn = document.getElementById('userMenuBtn');
    const userDropdown = document.getElementById('userDropdown');

    if (userMenuBtn && userDropdown) {
        userMenuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            userDropdown.classList.toggle('active');
        });

        document.addEventListener('click', () => {
            userDropdown.classList.remove('active');
        });
    }

    // Logout
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            logout();
        });
    }

    // Close video modal on click outside
    const videoModal = document.getElementById('videoModal');
    if (videoModal) {
        videoModal.addEventListener('click', (e) => {
            if (e.target === videoModal) {
                closeVideoModal();
            }
        });
    }

    // Close modal on ESC
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeVideoModal();
        }
    });
}

/**
 * Logout
 */
function logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = 'auth.html';
}

// ==============================================================================
// TOAST NOTIFICATIONS
// ==============================================================================

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
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

    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };
    toast.style.background = colors[type] || colors.info;

    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };

    toast.innerHTML = `
        <i class="fas fa-${icons[type]}" style="margin-right: 8px;"></i>
        ${message}
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Add animation styles
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
