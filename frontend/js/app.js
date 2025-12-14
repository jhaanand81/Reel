/**
 * REELCRAFT AI - Fully Consolidated Frontend Application
 * Complete standalone JavaScript with all functionality embedded
 * No external dependencies required
 */

// === CONFIGURATION ===
const CONFIG = {
    API_VERSION: 'v1',
    API_BASE_URL: window.REELSENSE_CONFIG?.API_BASE_URL || window.__API_BASE_URL__ || `${location.origin}/api`,
    DEFAULT_TIMEOUT: 60000, // 60 seconds for initial requests
    POLL_INTERVAL: 800, // Start polling every 800ms for SUPER FAST updates
    POLL_INTERVAL_SLOW: 3000, // Slow down to 3s after 1 minute
    POLL_SLOWDOWN_AFTER: 60000, // Slow down after 60 seconds
    MAX_POLL_DURATION: 900000, // 15 minutes
    AUTO_SAVE_INTERVAL: 30000,
    NOTIFICATION_DURATION: 4000,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
    PEXELS_API_KEY: 'YOUR_PEXELS_API_KEY', // Add your key here
    PIXABAY_API_KEY: 'YOUR_PIXABAY_API_KEY' // Add your key here
};

// === PRODUCTION-GRADE API CLIENT ===
// Features: Single-flight 401 refresh, exponential backoff + jitter, GET dedup,
// ETag/TTL caching, idempotency keys, offline queue, concurrency limiter
class ApiClient {
    constructor(baseUrl = CONFIG.API_BASE_URL, version = CONFIG.API_VERSION, opts = {}) {
        this.baseUrl = baseUrl;
        this.version = version;

        // Retries and backoff
        this.maxRetries = opts.maxRetries ?? 3;
        this.baseDelayMs = opts.baseDelayMs ?? 400;
        this.jitterMs = opts.jitterMs ?? 300;

        // Concurrency control (prevents browser stampede)
        this.maxConcurrent = opts.maxConcurrent ?? 12;
        this._permits = this.maxConcurrent;
        this._waiters = [];

        // Caching / dedup
        this._etagCache = new Map(); // url -> { etag, data, ts }
        this._ttlCache = new Map();  // url -> { data, ts, ttlMs }
        this._inflight = new Map();  // dedupKey -> Promise

        // Single-flight auth refresh
        this._refreshPromise = null;

        // Offline queue
        this._offlineKey = 'api_offline_queue_v1';
        window.addEventListener('online', () => this._drainOfflineQueue());

        console.log('[ApiClient] Production client initialized with: concurrency=' + this.maxConcurrent + ', retries=' + this.maxRetries);
    }

    // === Token Management ===
    setApiKey(apiKey) {
        if (apiKey) localStorage.setItem('api_key', apiKey);
        else localStorage.removeItem('api_key');
    }
    setAccessToken(token) {
        if (token) localStorage.setItem('access_token', token);
        else localStorage.removeItem('access_token');
    }
    get accessToken() { return localStorage.getItem('access_token'); }
    get refreshToken() { return localStorage.getItem('refresh_token'); }
    get apiKey() { return localStorage.getItem('api_key'); }

    // === Core Request ===
    async request(endpoint, options = {}, timeout = CONFIG.DEFAULT_TIMEOUT) {
        const method = (options.method || 'GET').toUpperCase();
        const url = `${this.baseUrl}/${this.version}${endpoint}`;
        const requestId = this._genReqId();
        const idempotencyKey = options.idempotencyKey || (method !== 'GET' ? this._genIdemKey(method, endpoint, options.body) : undefined);
        const cacheTTL = options.cacheTTL; // ms
        const deduplicate = options.deduplicate ?? (method === 'GET');
        const useETag = options.useETag ?? (method === 'GET');

        // Offline queue for write ops
        if (!navigator.onLine && options.queueIfOffline && method !== 'GET') {
            this._enqueueOffline({ url, method, body: options.body, headers: options.headers, idempotencyKey });
            const err = new Error('Offline: request queued');
            err.code = 'OFFLINE_QUEUED';
            throw err;
        }

        // Build headers
        const headers = {
            'Content-Type': 'application/json',
            'X-Request-ID': requestId,
            ...(this.apiKey ? { 'X-API-Key': this.apiKey } : {}),
            ...(this.accessToken ? { 'Authorization': `Bearer ${this.accessToken}` } : {}),
            ...(method !== 'GET' && idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {}),
            ...(options.headers || {})
        };

        // Dedup key (GET only)
        const dedupKey = deduplicate ? `${method}|${url}|${options.body || ''}` : null;

        // TTL cache hit
        if (method === 'GET' && cacheTTL) {
            const entry = this._ttlCache.get(url);
            if (entry && (Date.now() - entry.ts) < entry.ttlMs) {
                return entry.data;
            }
        }

        // ETag conditional request
        if (method === 'GET' && useETag) {
            const etagEntry = this._etagCache.get(url);
            if (etagEntry?.etag) {
                headers['If-None-Match'] = etagEntry.etag;
            }
        }

        const attemptFetch = async (attempt) => {
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), timeout);

            try {
                const res = await this._withPermit(() => fetch(url, {
                    method,
                    signal: controller.signal,
                    headers,
                    body: options.body,
                    cache: 'no-store',
                    credentials: 'same-origin'
                }));

                clearTimeout(t);

                // 401 handling with single-flight refresh
                if (res.status === 401 && this.refreshToken && !options.noRetryOnAuth) {
                    const refreshed = await this._refreshAccessTokenSingleFlight();
                    if (refreshed) {
                        // Update auth header and retry
                        headers['Authorization'] = `Bearer ${this.accessToken}`;
                        return attemptFetch(attempt + 1);
                    }
                }

                // 304 Not Modified (ETag)
                if (res.status === 304 && method === 'GET' && useETag) {
                    const e = this._etagCache.get(url);
                    if (e?.data) return e.data;
                }

                // Parse JSON safely
                const text = await res.text();
                let data = text ? (() => { try { return JSON.parse(text); } catch { return { raw: text }; } })() : {};

                if (!res.ok) {
                    const err = new Error(data.error || res.statusText || 'Request failed');
                    err.code = data.error_code || (res.status === 429 ? 'RATE_LIMIT_EXCEEDED' : 'REQUEST_FAILED');
                    err.status = res.status;
                    err.requestId = data.request_id || requestId;
                    err.retryAfter = res.headers.get('retry-after') ? parseInt(res.headers.get('retry-after'), 10) : undefined;
                    throw err;
                }

                // Store ETag
                if (method === 'GET' && useETag) {
                    const etag = res.headers.get('etag');
                    if (etag) {
                        this._etagCache.set(url, { etag, data, ts: Date.now() });
                    }
                }

                // Store TTL cache
                if (method === 'GET' && cacheTTL) {
                    this._ttlCache.set(url, { data, ts: Date.now(), ttlMs: cacheTTL });
                }

                return data;

            } catch (error) {
                clearTimeout(t);
                if (error?.name === 'AbortError') {
                    const timeoutError = new Error('Request timed out');
                    timeoutError.code = 'TIMEOUT';
                    throw timeoutError;
                }
                if (!error?.code && !error?.status) {
                    error = Object.assign(new Error('Cannot connect to server. Please check your connection.'), { code: 'NETWORK_ERROR' });
                }
                throw error;
            }
        };

        // GET de-dup
        if (deduplicate) {
            const existing = this._inflight.get(dedupKey);
            if (existing) return existing;
        }

        const exec = async () => {
            let lastErr = null;
            for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
                try {
                    return await attemptFetch(attempt);
                } catch (err) {
                    lastErr = err;

                    // Respect Retry-After for 429
                    if (err.status === 429 && typeof err.retryAfter === 'number') {
                        console.log(`[ApiClient] Rate limited, waiting ${err.retryAfter}s`);
                        await this._sleep(err.retryAfter * 1000);
                        continue;
                    }

                    // Retry on 408/5xx/network/timeouts
                    const retryable = err.code === 'NETWORK_ERROR' || err.code === 'TIMEOUT' ||
                        (err.status && [408, 429, 500, 502, 503, 504].includes(err.status));

                    if (!retryable || attempt === this.maxRetries) break;

                    const backoff = this._backoffWithJitter(attempt);
                    console.log(`[ApiClient] Retry ${attempt + 1}/${this.maxRetries} in ${backoff}ms`);
                    await this._sleep(backoff);
                }
            }
            throw lastErr;
        };

        const p = exec();
        if (deduplicate) this._inflight.set(dedupKey, p);
        try {
            return await p;
        } finally {
            if (deduplicate) this._inflight.delete(dedupKey);
        }
    }

    // === API Methods ===
    async checkHealth() {
        return this.request('/health', { method: 'GET', cacheTTL: 3000, useETag: true });
    }

    async generateConcept(data) {
        return this.request('/generate-concept', {
            method: 'POST',
            body: JSON.stringify(data)
        }, 60000);
    }

    async generateScript(data) {
        return this.request('/generate-script', {
            method: 'POST',
            body: JSON.stringify(data),
            queueIfOffline: true
        }, 60000);
    }

    async generateVoiceover(data) {
        return this.request('/generate-voiceover', {
            method: 'POST',
            body: JSON.stringify(data),
            queueIfOffline: true
        }, 60000);
    }

    async generateVideo(data) {
        return this.request('/generate-video', {
            method: 'POST',
            body: JSON.stringify(data),
            queueIfOffline: true
        }, 60000);
    }

    async checkVideoStatus(jobId, projectId = null) {
        const params = projectId ? `?projectId=${encodeURIComponent(projectId)}` : '';
        // Disable dedup for status polls to ensure fresh data
        return this.request(`/video-status/${encodeURIComponent(jobId)}${params}`, {
            method: 'GET',
            deduplicate: false
        }, 20000);
    }

    async composeVideo(data) {
        return this.request('/compose-video', {
            method: 'POST',
            body: JSON.stringify(data),
            queueIfOffline: true
        }, 60000);
    }

    async addCaptions(data) {
        return this.request('/add-captions', {
            method: 'POST',
            body: JSON.stringify(data),
            queueIfOffline: true
        }, 60000);
    }

    async generateVideoFromImage(data) {
        return this.request('/generate-video-from-image', {
            method: 'POST',
            body: JSON.stringify(data),
            queueIfOffline: true
        }, 60000);
    }

    async enhancePrompt(prompt, context = null) {
        return this.request('/enhance-prompt', {
            method: 'POST',
            body: JSON.stringify({ prompt, context })
        }, 15000);  // 15s timeout for LLM call
    }

    async downloadVideo(projectId, format = 'mp4') {
        const url = `${this.baseUrl}/${this.version}/download/${encodeURIComponent(projectId)}?format=${encodeURIComponent(format)}`;
        window.open(url, '_blank');
    }

    async testService(service) {
        return this.request(`/test/${service}`, { method: 'GET' });
    }

    // === Internal Helpers ===
    _genReqId() {
        return `req_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    }

    _genIdemKey(method, endpoint, body) {
        const seed = `${method}|${endpoint}|${typeof body === 'string' ? body : JSON.stringify(body || {})}`;
        const hash = btoa(unescape(encodeURIComponent(seed))).replace(/=+$/, '').slice(0, 16);
        return `idem_${Date.now()}_${Math.random().toString(36).slice(2, 7)}_${hash}`;
    }

    _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    _backoffWithJitter(attempt) {
        const base = this.baseDelayMs * Math.pow(2, attempt);
        const jitter = Math.floor(Math.random() * this.jitterMs);
        return base + jitter;
    }

    // Single-flight token refresh (prevents thundering herd)
    async _refreshAccessTokenSingleFlight() {
        if (this._refreshPromise) return this._refreshPromise;
        if (!this.refreshToken) return false;

        this._refreshPromise = (async () => {
            try {
                console.log('[ApiClient] Refreshing access token...');
                const url = `${this.baseUrl}/${this.version}/auth/refresh`;
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-Request-ID': this._genReqId() },
                    body: JSON.stringify({ refresh_token: this.refreshToken })
                });
                const data = await res.json();
                if (res.ok && data?.status === 'success') {
                    localStorage.setItem('access_token', data.data.access_token);
                    if (data.data.refresh_token) localStorage.setItem('refresh_token', data.data.refresh_token);
                    console.log('[ApiClient] Token refreshed successfully');
                    return true;
                }
                localStorage.removeItem('access_token');
                return false;
            } catch (e) {
                console.error('[ApiClient] Token refresh failed:', e);
                return false;
            } finally {
                this._refreshPromise = null;
            }
        })();
        return this._refreshPromise;
    }

    // Concurrency limiter (semaphore)
    async _withPermit(fn) {
        if (this._permits > 0) {
            this._permits--;
            try { return await fn(); }
            finally { this._release(); }
        }
        await new Promise(res => this._waiters.push(res));
        return this._withPermit(fn);
    }

    _release() {
        if (this._waiters.length) this._waiters.shift()();
        else this._permits = Math.min(this._permits + 1, this.maxConcurrent);
    }

    // Offline queue
    _enqueueOffline(entry) {
        try {
            const q = JSON.parse(localStorage.getItem(this._offlineKey) || '[]');
            q.push({ ...entry, ts: Date.now() });
            if (q.length > 100) q.splice(0, q.length - 100);
            localStorage.setItem(this._offlineKey, JSON.stringify(q));
            console.log('[ApiClient] Request queued for offline replay');
        } catch (e) { console.error('[ApiClient] Offline queue error:', e); }
    }

    async _drainOfflineQueue() {
        try {
            const q = JSON.parse(localStorage.getItem(this._offlineKey) || '[]');
            if (!q.length) return;
            console.log(`[ApiClient] Draining ${q.length} offline requests...`);
            localStorage.removeItem(this._offlineKey);
            for (const item of q) {
                try {
                    await this.request(item.url.replace(`${this.baseUrl}/${this.version}`, ''), {
                        method: item.method,
                        headers: item.headers,
                        body: item.body,
                        idempotencyKey: item.idempotencyKey
                    });
                } catch (e) {
                    this._enqueueOffline(item);
                }
            }
        } catch (e) { console.error('[ApiClient] Drain queue error:', e); }
    }
}

// === PRODUCTION-GRADE VIDEO POLLER ===
// Features: Adaptive polling with jitter, Retry-After support, server-guided next_poll_in
class VideoPoller {
    constructor() {
        this.active = new Map(); // jobId -> { cancelled: false, timeoutId: number }
        this.maxPollDuration = CONFIG.MAX_POLL_DURATION;
    }

    /**
     * Start polling for video status with adaptive intervals + jitter
     * Uses recursive setTimeout for dynamic interval control
     * Respects server 'next_poll_in' and 'Retry-After' hints
     */
    async poll(jobId, onProgress, statusChecker) {
        this.cancel(jobId);
        const state = { cancelled: false, timeoutId: null };
        this.active.set(jobId, state);

        return new Promise((resolve, reject) => {
            const start = Date.now();
            let interval = CONFIG.POLL_INTERVAL;
            let pollCount = 0;

            const tick = async () => {
                if (state.cancelled) return;

                const elapsed = Date.now() - start;
                if (elapsed > this.maxPollDuration) {
                    this.cancel(jobId);
                    return reject(new Error('Video generation timed out after 15 minutes'));
                }

                pollCount++;
                console.log(`[VideoPoller] Poll #${pollCount} for ${jobId} (${Math.floor(elapsed / 1000)}s elapsed, interval=${interval}ms)`);

                try {
                    const status = await statusChecker(jobId);

                    // Update progress
                    if (typeof status.progress === 'number') {
                        onProgress?.(status.progress);
                    }

                    // Completed
                    if (status.status === 'completed' && status.videoUrl) {
                        console.log(`[VideoPoller] Job ${jobId} completed after ${pollCount} polls`);
                        this.cancel(jobId);
                        return resolve(status);
                    }

                    // Failed
                    if (status.status === 'failed') {
                        this.cancel(jobId);
                        return reject(new Error(status.error || 'Video generation failed'));
                    }

                    // Adaptive slowdown after 60s
                    if (elapsed > CONFIG.POLL_SLOWDOWN_AFTER && interval === CONFIG.POLL_INTERVAL) {
                        interval = CONFIG.POLL_INTERVAL_SLOW;
                        console.log(`[VideoPoller] Slowing to ${interval}ms for job ${jobId}`);
                    }

                    // Server hint: next_poll_in (in seconds)
                    if (typeof status.next_poll_in === 'number' && status.next_poll_in > 0) {
                        interval = Math.max(500, Math.min(CONFIG.POLL_INTERVAL_SLOW, status.next_poll_in * 1000));
                        console.log(`[VideoPoller] Server requested next poll in ${status.next_poll_in}s`);
                    }

                } catch (error) {
                    console.error('[VideoPoller] Poll error:', error);

                    // Network errors: keep polling
                    if (error.code === 'NETWORK_ERROR' || error.code === 'TIMEOUT') {
                        console.log('[VideoPoller] Network error, will retry...');
                    }
                    // 429 Rate Limit: respect Retry-After
                    else if (error.status === 429 && typeof error.retryAfter === 'number') {
                        interval = Math.max(1000, error.retryAfter * 1000);
                        console.log(`[VideoPoller] Rate limited, waiting ${error.retryAfter}s`);
                    }
                    // Other errors: stop polling
                    else {
                        this.cancel(jobId);
                        return reject(error);
                    }
                }

                // Schedule next poll with jitter (0-250ms)
                if (!state.cancelled) {
                    const jitter = Math.floor(Math.random() * 250);
                    state.timeoutId = setTimeout(tick, interval + jitter);
                }
            };

            // Start immediately
            tick();
        });
    }

    /**
     * Cancel polling for a specific job
     */
    cancel(jobId) {
        const state = this.active.get(jobId);
        if (state) {
            state.cancelled = true;
            if (state.timeoutId) {
                clearTimeout(state.timeoutId);
            }
            this.active.delete(jobId);
            console.log(`[VideoPoller] Cancelled polling for job ${jobId}`);
        }
    }

    /**
     * Cancel all active polls
     */
    cancelAll() {
        for (const jobId of this.active.keys()) {
            this.cancel(jobId);
        }
        console.log('[VideoPoller] Cancelled all active polls');
    }

    /**
     * Get active poll count
     */
    getActiveCount() {
        return this.active.size;
    }
}

// === STOCK MEDIA SERVICE ===
class StockMedia {
    constructor() {
        this.pexelsApiKey = CONFIG.PEXELS_API_KEY;
        this.pixabayApiKey = CONFIG.PIXABAY_API_KEY;
        this.cache = new Map();
    }

    /**
     * Search for stock videos from multiple sources
     */
    async searchVideos(query, options = {}) {
        const {
            perPage = 15,
            orientation = 'landscape',
            size = 'medium',
            source = 'all' // 'all', 'pexels', 'pixabay'
        } = options;

        // Check cache
        const cacheKey = `${query}_${source}_${perPage}`;
        if (this.cache.has(cacheKey)) {
            console.log('[StockMedia] Returning cached results');
            return this.cache.get(cacheKey);
        }

        const results = [];

        try {
            // Search Pexels
            if ((source === 'all' || source === 'pexels') && this.pexelsApiKey && this.pexelsApiKey !== 'YOUR_PEXELS_API_KEY') {
                console.log('[StockMedia] Searching Pexels...');
                const pexelsResults = await this.searchPexels(query, perPage, orientation);
                results.push(...pexelsResults);
            }

            // Search Pixabay
            if ((source === 'all' || source === 'pixabay') && this.pixabayApiKey && this.pixabayApiKey !== 'YOUR_PIXABAY_API_KEY') {
                console.log('[StockMedia] Searching Pixabay...');
                const pixabayResults = await this.searchPixabay(query, perPage);
                results.push(...pixabayResults);
            }

            // Sort by quality/relevance
            results.sort((a, b) => (b.quality || 0) - (a.quality || 0));

            // Cache results
            this.cache.set(cacheKey, results);

            console.log(`[StockMedia] Found ${results.length} videos`);
            return results;

        } catch (error) {
            console.error('[StockMedia] Search error:', error);
            return [];
        }
    }

    /**
     * Search Pexels API
     */
    async searchPexels(query, perPage, orientation) {
        const url = `https://api.pexels.com/videos/search?query=${encodeURIComponent(query)}&per_page=${perPage}&orientation=${orientation}`;

        try {
            const response = await fetch(url, {
                headers: {
                    'Authorization': this.pexelsApiKey
                }
            });

            if (!response.ok) {
                throw new Error(`Pexels API error: ${response.status}`);
            }

            const data = await response.json();

            return data.videos.map(video => ({
                id: `pexels_${video.id}`,
                source: 'pexels',
                url: video.video_files[0]?.link || '',
                thumbnail: video.image,
                duration: video.duration,
                width: video.width,
                height: video.height,
                quality: video.video_files[0]?.quality || 'hd',
                author: video.user?.name || 'Unknown',
                authorUrl: video.user?.url || '',
                videoUrl: video.url
            }));

        } catch (error) {
            console.error('[StockMedia] Pexels error:', error);
            return [];
        }
    }

    /**
     * Search Pixabay API
     */
    async searchPixabay(query, perPage) {
        const url = `https://pixabay.com/api/videos/?key=${this.pixabayApiKey}&q=${encodeURIComponent(query)}&per_page=${perPage}`;

        try {
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`Pixabay API error: ${response.status}`);
            }

            const data = await response.json();

            return data.hits.map(video => ({
                id: `pixabay_${video.id}`,
                source: 'pixabay',
                url: video.videos?.medium?.url || video.videos?.small?.url || '',
                thumbnail: video.picture_id ? `https://i.vimeocdn.com/video/${video.picture_id}_640.jpg` : '',
                duration: video.duration,
                width: video.videos?.medium?.width || 1280,
                height: video.videos?.medium?.height || 720,
                quality: 'hd',
                author: video.user || 'Unknown',
                authorUrl: `https://pixabay.com/users/${video.user_id}/`,
                videoUrl: `https://pixabay.com/videos/id-${video.id}/`
            }));

        } catch (error) {
            console.error('[StockMedia] Pixabay error:', error);
            return [];
        }
    }

    /**
     * Download stock video
     */
    async downloadVideo(videoData) {
        try {
            console.log('[StockMedia] Downloading video:', videoData.id);

            const response = await fetch(videoData.url);
            if (!response.ok) {
                throw new Error(`Download failed: ${response.status}`);
            }

            const blob = await response.blob();
            return blob;

        } catch (error) {
            console.error('[StockMedia] Download error:', error);
            throw error;
        }
    }

    /**
     * Get video by ID from cache
     */
    getVideoById(videoId) {
        for (const results of this.cache.values()) {
            const video = results.find(v => v.id === videoId);
            if (video) return video;
        }
        return null;
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
        console.log('[StockMedia] Cache cleared');
    }
}

// === STATE MANAGEMENT ===
class AppStateManager {
    constructor() {
        this.state = {
            currentStage: 1,
            isGenerating: false,
            isConnected: false,
            services: {},
            projectData: this.getDefaultProjectData(),
            history: [],
            settings: this.loadSettings()
        };

        this.listeners = new Map();
        this.autoSaveTimer = null;
    }

    getDefaultProjectData() {
        return {
            projectId: null,
            reelType: 'ai-generated',
            duration: 30,
            topic: '',
            brand: '',
            scriptLength: 'medium',
            tone: 'witty',
            emotion: 'energetic',
            script: '',
            voiceType: 'male-professional',
            musicStyle: 'upbeat',
            videoUrl: null,
            audioPath: null,
            jobId: null,
            brandGuidelines: null,
            stockVideos: [],
            selectedStockVideo: null,
            // Image-to-Video fields
            sourceImage: null,
            sourceImageData: null,
            videoQuality: '1080p',
            metadata: {
                created: new Date().toISOString(),
                modified: new Date().toISOString()
            }
        };
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('appSettings');
            return saved ? JSON.parse(saved) : {
                theme: 'light',
                autoSave: true,
                notifications: true
            };
        } catch (e) {
            console.error('Failed to load settings:', e);
            return { theme: 'light', autoSave: true, notifications: true };
        }
    }

    /**
     * Update state and notify listeners
     */
    setState(updates) {
        const oldState = { ...this.state };
        this.state = { ...this.state, ...updates };

        // Notify listeners
        this.notifyListeners(oldState, this.state);

        // Auto-save if enabled
        if (this.state.settings.autoSave) {
            this.scheduleSave();
        }
    }

    /**
     * Update project data
     */
    updateProject(updates) {
        this.setState({
            projectData: {
                ...this.state.projectData,
                ...updates,
                metadata: {
                    ...this.state.projectData.metadata,
                    modified: new Date().toISOString()
                }
            }
        });
    }

    /**
     * Subscribe to state changes
     */
    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }
        this.listeners.get(key).push(callback);

        return () => {
            const callbacks = this.listeners.get(key);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        };
    }

    /**
     * Notify listeners of state changes
     */
    notifyListeners(oldState, newState) {
        this.listeners.forEach((callbacks, key) => {
            callbacks.forEach(callback => {
                callback(newState, oldState);
            });
        });
    }

    /**
     * Schedule auto-save
     */
    scheduleSave() {
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
        }

        this.autoSaveTimer = setTimeout(() => {
            this.save();
        }, 5000); // Save after 5 seconds of inactivity
    }

    /**
     * Get current user ID from localStorage
     */
    getCurrentUserId() {
        try {
            const userStr = localStorage.getItem('user');
            if (userStr) {
                const user = JSON.parse(userStr);
                return user.id || user.email || null;
            }
        } catch (e) {
            console.error('Failed to get user ID:', e);
        }
        return null;
    }

    /**
     * Save state to localStorage (user-specific)
     */
    save() {
        try {
            // Don't save File objects
            const { brandGuidelines, stockVideos, ...projectData } = this.state.projectData;

            const userId = this.getCurrentUserId();
            const saveData = {
                userId: userId,  // Track which user this state belongs to
                currentStage: this.state.currentStage,
                projectData: projectData,
                settings: this.state.settings,
                savedAt: new Date().toISOString()
            };

            localStorage.setItem('appState', JSON.stringify(saveData));

            // Save to history
            this.addToHistory(projectData);

            console.log('State saved successfully for user:', userId);
        } catch (e) {
            console.error('Failed to save state:', e);
        }
    }

    /**
     * Load state from localStorage (user-specific)
     */
    load() {
        try {
            const saved = localStorage.getItem('appState');
            if (saved) {
                const data = JSON.parse(saved);
                const currentUserId = this.getCurrentUserId();

                // Only restore state if it belongs to the current user
                if (data.userId && data.userId !== currentUserId) {
                    console.log('State belongs to different user, clearing...');
                    this.clear();  // Clear old user's state
                    return false;
                }

                this.state = {
                    ...this.state,
                    currentStage: data.currentStage || 1,
                    projectData: {
                        ...this.getDefaultProjectData(),
                        ...data.projectData
                    },
                    settings: data.settings || this.state.settings
                };
                console.log('State loaded successfully for user:', currentUserId);
                return true;
            }
        } catch (e) {
            console.error('Failed to load state:', e);
        }
        return false;
    }

    /**
     * Clear saved state
     */
    clear() {
        try {
            localStorage.removeItem('appState');
            localStorage.removeItem('projectHistory');
            this.state.currentStage = 1;
            this.state.projectData = this.getDefaultProjectData();
            console.log('State cleared');
        } catch (e) {
            console.error('Failed to clear state:', e);
        }
    }

    /**
     * Add project to history
     */
    addToHistory(projectData) {
        try {
            const history = JSON.parse(localStorage.getItem('projectHistory') || '[]');

            // Remove duplicates
            const filtered = history.filter(p => p.projectId !== projectData.projectId);

            // Add to beginning
            filtered.unshift({
                ...projectData,
                savedAt: new Date().toISOString()
            });

            // Keep only last 20 projects
            const trimmed = filtered.slice(0, 20);

            localStorage.setItem('projectHistory', JSON.stringify(trimmed));
            this.state.history = trimmed;
        } catch (e) {
            console.error('Failed to save to history:', e);
        }
    }

    /**
     * Reset to default state
     */
    reset() {
        this.state = {
            ...this.state,
            currentStage: 1,
            projectData: this.getDefaultProjectData()
        };
        this.save();
    }
}

// === UI CONTROLLER ===
class UIController {
    constructor(stateManager, apiClient) {
        this.state = stateManager;
        this.api = apiClient;
        this.videoPoller = new VideoPoller();
        this.stockMedia = new StockMedia();
        this.elements = this.cacheElements();
        this.scrollHandler = null;
    }

    /**
     * Cache DOM elements
     */
    cacheElements() {
        return {
            // Navigation
            mobileToggle: document.getElementById('mobileToggle'),
            navMenu: document.querySelector('.nav-menu'),
            navbar: document.querySelector('.navbar'),

            // Loading
            loadingOverlay: document.getElementById('loadingOverlay'),
            loadingText: document.querySelector('#loadingOverlay p'),

            // Stages
            stages: document.querySelectorAll('.stage-container'),
            steps: document.querySelectorAll('.step'),

            // Form inputs
            topic: document.getElementById('topic'),
            brand: document.getElementById('brand'),
            reelType: document.getElementById('reelType'),
            duration: document.getElementById('duration'),
            videoQuality: document.getElementById('videoQuality'),
            scriptLength: document.getElementById('scriptLength'),
            tone: document.getElementById('tone'),
            brandGuidelines: document.getElementById('brandGuidelines'),

            // Preview
            scriptPreview: document.getElementById('scriptPreview'),
            previewVideo: document.getElementById('previewVideo'),

            // Buttons
            generateBtn: document.getElementById('generateBtn'),

            // Progress
            generationProgress: document.getElementById('generationProgress'),
            progressItems: document.querySelectorAll('.progress-item'),

            // File upload
            fileNameDisplay: document.querySelector('.file-name-display'),
            fileUploadDisplay: document.querySelector('.file-upload-display')
        };
    }

    /**
     * Initialize UI
     */
    init() {
        this.setupEventListeners();
        this.setupNavigation();
        this.setupFileUpload();
        this.restoreState();
        this.setupAutoSave();
        this.initParticles();

        // Subscribe to state changes
        this.state.subscribe('ui', (newState) => {
            this.updateUI(newState);
        });

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.videoPoller.cancelAll();
        });

        console.log('%c🎬 Reel Sense AI Initialized', 'font-size: 20px; color: #826c6c; font-weight: bold');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Stage navigation
        window.goToStage = (stage) => this.goToStage(stage);

        // Script generation
        window.generateScript = () => this.generateScript();
        window.regenerateScript = () => this.regenerateScript();

        // Video generation
        window.startGeneration = () => this.startGeneration();

        // Prompt enhancement
        window.enhanceVisualPrompt = () => this.enhanceVisualPrompt();
        window.useEnhancedPrompt = () => this.useEnhancedPrompt();
        window.revertPrompt = () => this.revertPrompt();

        // Edit functions
        window.regenVoiceover = () => this.regenVoiceover();
        window.changeMusic = () => this.changeMusic();
        window.addCaptions = () => this.addCaptions();

        // Stock media
        window.searchStockMedia = (query) => this.searchStockMedia(query);
        window.selectStockVideo = (videoId) => this.selectStockVideo(videoId);

        // Image-to-Video
        window.toggleImageUpload = () => this.toggleImageUpload();
        window.previewSourceImage = (input) => this.previewSourceImage(input);

        // Utility functions
        window.scrollToGenerator = () => this.scrollToGenerator();
        window.playDemo = () => this.playDemo();
        window.createNew = () => this.createNew();
        window.downloadVideo = (format) => this.downloadVideo(format);
        window.shareVideo = (platform) => this.shareVideo(platform);
        window.viewVideo = () => this.viewVideo();

        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
    }

    /**
     * Setup navigation
     */
    setupNavigation() {
        // Mobile menu toggle
        if (this.elements.mobileToggle) {
            this.elements.mobileToggle.addEventListener('click', () => {
                this.elements.navMenu.classList.toggle('active');
            });
        }

        // Smooth scroll for nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.getAttribute('href').startsWith('#')) {
                    e.preventDefault();
                    const target = document.querySelector(link.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                        this.elements.navMenu?.classList.remove('active');
                    }
                }
            });
        });

        // Navbar scroll effect
        window.addEventListener('scroll', this.throttle(() => {
            if (window.scrollY > 50) {
                this.elements.navbar?.classList.add('scrolled');
            } else {
                this.elements.navbar?.classList.remove('scrolled');
            }
            this.updateActiveNav();
        }, 100));
    }

    /**
     * Update active navigation
     */
    updateActiveNav() {
        const sections = document.querySelectorAll('section[id]');
        const scrollPos = window.scrollY + 100;

        sections.forEach(section => {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const id = section.getAttribute('id');

            if (scrollPos >= top && scrollPos < top + height) {
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    /**
     * Setup file upload
     */
    setupFileUpload() {
        const { brandGuidelines, fileNameDisplay, fileUploadDisplay } = this.elements;

        if (fileUploadDisplay && brandGuidelines) {
            fileUploadDisplay.addEventListener('click', () => {
                brandGuidelines.click();
            });
        }

        if (brandGuidelines && fileNameDisplay) {
            brandGuidelines.addEventListener('change', (e) => {
                const file = e.target.files[0];

                if (file) {
                    // Validate file size
                    if (file.size > CONFIG.MAX_FILE_SIZE) {
                        this.showNotification('File too large. Maximum size is 10MB.', 'warning');
                        brandGuidelines.value = '';
                        fileNameDisplay.textContent = 'No file chosen';
                        return;
                    }

                    // Validate file type
                    if (!CONFIG.ALLOWED_FILE_TYPES.includes(file.type)) {
                        this.showNotification('Invalid file type. Please upload PDF, TXT, or DOC file.', 'warning');
                        brandGuidelines.value = '';
                        fileNameDisplay.textContent = 'No file chosen';
                        return;
                    }

                    // File is valid
                    fileNameDisplay.textContent = file.name;
                    fileNameDisplay.style.color = 'var(--primary)';
                    this.state.updateProject({ brandGuidelines: file });
                    this.showNotification('Brand guidelines uploaded successfully!', 'success');
                } else {
                    fileNameDisplay.textContent = 'No file chosen';
                    fileNameDisplay.style.color = 'var(--gray)';
                    this.state.updateProject({ brandGuidelines: null });
                }
            });
        }
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to proceed
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                const currentStage = this.state.state.currentStage;
                if (currentStage < 5) {
                    this.goToStage(currentStage + 1);
                }
            }

            // Ctrl/Cmd + G to generate
            if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
                e.preventDefault();
                if (this.state.state.currentStage === 2) {
                    this.generateScript();
                } else if (this.state.state.currentStage === 3) {
                    this.startGeneration();
                }
            }

            // Ctrl/Cmd + S to save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.state.save();
                this.showNotification('Project saved', 'success');
            }

            // Escape to cancel
            if (e.key === 'Escape') {
                if (this.state.state.isGenerating) {
                    this.cancelGeneration();
                }
            }
        });
    }

    /**
     * Go to stage
     */
    goToStage(stageNumber) {
        // Validate current stage
        if (stageNumber > this.state.state.currentStage) {
            if (!this.validateStage(this.state.state.currentStage)) {
                this.showNotification('Please complete all required fields', 'warning');
                return;
            }
        }

        // Update stage containers
        this.elements.stages.forEach(container => {
            container.classList.remove('active');
        });
        const targetStage = document.getElementById(`stage${stageNumber}`);
        if (targetStage) {
            targetStage.classList.add('active');
        }

        // Update progress steps
        this.elements.steps.forEach((step, index) => {
            if (index + 1 <= stageNumber) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });

        // Update state
        this.state.setState({ currentStage: stageNumber });

        // If going to Stage 2 and no script exists, auto-generate it
        if (stageNumber === 2 && !this.state.state.projectData.script) {
            this.generateScript();
        }

        // If going to Stage 3, copy script preview and reset UI
        if (stageNumber === 3) {
            const scriptPreview = document.getElementById('scriptPreview');
            const scriptPreviewReadonly = document.getElementById('scriptPreviewReadonly');
            if (scriptPreview && scriptPreviewReadonly) {
                scriptPreviewReadonly.textContent = scriptPreview.value || 'No script generated yet.';
            }

            // Reset Stage 3 UI state
            const stage3Preview = document.getElementById('stage3VideoPreview');
            const stage3NextBtn = document.getElementById('stage3NextBtn');
            const visualPromptSection = document.getElementById('visualPromptSection');

            // If video already generated, show preview
            if (this.state.state.projectData.videoUrl) {
                const stage3Video = document.getElementById('stage3PreviewVideo');
                if (stage3Preview && stage3Video) {
                    stage3Video.src = this.state.state.projectData.videoUrl;
                    stage3Preview.style.display = 'block';
                }
                if (this.elements.generateBtn) {
                    this.elements.generateBtn.style.display = 'none';
                }
                if (stage3NextBtn) {
                    stage3NextBtn.style.display = 'inline-flex';
                }
            } else {
                // No video yet - show generate button, hide preview
                if (stage3Preview) {
                    stage3Preview.style.display = 'none';
                }
                if (this.elements.generateBtn) {
                    this.elements.generateBtn.style.display = 'inline-flex';
                    this.elements.generateBtn.disabled = false;
                    this.elements.generateBtn.innerHTML = '<i class="fas fa-play"></i> Generate Video';
                }
                if (stage3NextBtn) {
                    stage3NextBtn.style.display = 'none';
                }
            }

            // Always show the prompt/quality section
            if (visualPromptSection) {
                visualPromptSection.style.display = 'block';
            }
        }

        // Scroll to generator
        document.getElementById('create')?.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Validate stage
     */
    validateStage(stageNumber) {
        const { projectData } = this.state.state;

        switch (stageNumber) {
            case 1:
                const topic = this.elements.topic?.value.trim();
                if (!topic) {
                    return false;
                }
                this.state.updateProject({
                    topic,
                    brand: this.elements.brand?.value.trim(),
                    reelType: this.elements.reelType?.value,
                    duration: parseInt(this.elements.duration?.value || 30),
                    videoQuality: this.elements.videoQuality?.value || '1080p'
                });
                return true;

            case 2:
                return !!projectData.script;

            case 3:
                return !!projectData.videoUrl;

            default:
                return true;
        }
    }

    /**
     * Generate script
     */
    async generateScript(regenerate = false) {
        if (this.state.state.isGenerating) return;

        const topic = this.elements.topic?.value.trim();
        const brand = this.elements.brand?.value.trim();

        if (!topic) {
            this.showNotification('Please enter a topic first', 'warning');
            this.goToStage(1);
            return;
        }

        this.state.setState({ isGenerating: true });
        const loadingMsg = regenerate ? 'Regenerating script with fresh ideas...' : 'Generating script with AI...';
        this.showLoading(loadingMsg);

        try {
            const response = await this.api.generateScript({
                topic,
                brand,
                length: this.elements.scriptLength?.value || 'medium',
                tone: this.elements.tone?.value || 'witty',
                duration: parseInt(this.elements.duration?.value || 30),
                regenerate: regenerate
            });

            if (response.status === 'success' && response.data) {
                this.state.updateProject({
                    script: response.data.script,
                    projectId: response.data.projectId
                });

                if (this.elements.scriptPreview) {
                    this.elements.scriptPreview.value = response.data.script;
                }

                this.showNotification('Script generated successfully!', 'success');
            } else {
                throw new Error(response.error || 'Failed to generate script');
            }
        } catch (error) {
            console.error('Script generation error:', error);
            this.handleError(error, 'Failed to generate script');
        } finally {
            this.state.setState({ isGenerating: false });
            this.hideLoading();
        }
    }

    /**
     * Regenerate script
     */
    async regenerateScript() {
        this.state.updateProject({ script: '' });
        if (this.elements.scriptPreview) {
            this.elements.scriptPreview.value = '';
        }
        await this.generateScript(true);  // Pass true for regenerate flag
    }

    // =========================================================================
    // PROMPT ENHANCEMENT
    // =========================================================================

    /**
     * Store for original prompt before enhancement
     */
    _originalPrompt = null;
    _enhancedPrompt = null;

    /**
     * Enhance the visual prompt using AI
     */
    async enhanceVisualPrompt() {
        const promptInput = document.getElementById('visualPrompt');
        const enhanceBtn = document.getElementById('enhancePromptBtn');
        const useBtn = document.getElementById('useEnhancedBtn');
        const revertBtn = document.getElementById('revertPromptBtn');
        const previewDiv = document.getElementById('enhancedPromptPreview');
        const previewText = document.getElementById('enhancedPromptText');

        const prompt = promptInput?.value?.trim();

        if (!prompt || prompt.length < 3) {
            this.showNotification('Please enter a visual prompt first (at least 3 characters)', 'warning');
            return;
        }

        // Save original
        this._originalPrompt = prompt;

        // Show loading state
        enhanceBtn.disabled = true;
        enhanceBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enhancing...';

        try {
            const response = await this.api.enhancePrompt(prompt, this.state.state.projectData?.reelType || 'ai-generated');

            if (response.status === 'success' && response.data?.enhanced) {
                this._enhancedPrompt = response.data.enhanced;

                // Show preview
                previewText.textContent = this._enhancedPrompt;
                previewDiv.style.display = 'block';

                // Show use/revert buttons
                useBtn.style.display = 'inline-flex';
                revertBtn.style.display = 'inline-flex';

                this.showNotification('Prompt enhanced! Review and click "Use Enhanced" to apply.', 'success');
            } else {
                throw new Error(response.error || 'Enhancement failed');
            }

        } catch (error) {
            console.error('[EnhancePrompt] Error:', error);
            this.showNotification('Failed to enhance prompt: ' + (error.message || 'Unknown error'), 'error');
        } finally {
            enhanceBtn.disabled = false;
            enhanceBtn.innerHTML = '<i class="fas fa-sparkles"></i> Enhance Prompt';
        }
    }

    /**
     * Apply the enhanced prompt
     */
    useEnhancedPrompt() {
        const promptInput = document.getElementById('visualPrompt');
        const useBtn = document.getElementById('useEnhancedBtn');
        const revertBtn = document.getElementById('revertPromptBtn');
        const previewDiv = document.getElementById('enhancedPromptPreview');

        if (this._enhancedPrompt && promptInput) {
            promptInput.value = this._enhancedPrompt;
            previewDiv.style.display = 'none';
            useBtn.style.display = 'none';
            // Keep revert button visible so user can go back
            revertBtn.style.display = 'inline-flex';

            this.showNotification('Enhanced prompt applied!', 'success');
        }
    }

    /**
     * Revert to original prompt
     */
    revertPrompt() {
        const promptInput = document.getElementById('visualPrompt');
        const useBtn = document.getElementById('useEnhancedBtn');
        const revertBtn = document.getElementById('revertPromptBtn');
        const previewDiv = document.getElementById('enhancedPromptPreview');

        if (this._originalPrompt && promptInput) {
            promptInput.value = this._originalPrompt;
            previewDiv.style.display = 'none';
            useBtn.style.display = 'none';
            revertBtn.style.display = 'none';

            // Clear stored prompts
            this._originalPrompt = null;
            this._enhancedPrompt = null;

            this.showNotification('Reverted to original prompt', 'info');
        }
    }

    // =========================================================================
    // VIDEO GENERATION
    // =========================================================================

    /**
     * Start video generation
     */
    async startGeneration() {
        if (this.state.state.isGenerating) return;

        const { projectData } = this.state.state;

        if (!projectData.script) {
            this.showNotification('Please generate a script first', 'warning');
            this.goToStage(2);
            return;
        }

        // Get visual prompt and quality from UI
        const visualPromptInput = document.getElementById('visualPrompt');
        const qualitySelect = document.getElementById('qualityPreset');
        const voiceSelect = document.getElementById('voiceType');
        const visualPrompt = visualPromptInput?.value?.trim() || null;
        const quality = qualitySelect?.value || 'standard';
        const voiceType = voiceSelect?.value || 'male-professional';

        // Update project data with visual prompt and quality
        if (visualPrompt) {
            this.state.updateProject({ visualPrompt, styleTokens: visualPrompt });
        }
        this.state.updateProject({ quality, voiceType });

        this.state.setState({ isGenerating: true });

        if (this.elements.generateBtn) {
            this.elements.generateBtn.disabled = true;
            this.elements.generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        }

        // Show progress, hide prompt section
        if (this.elements.generationProgress) {
            this.elements.generationProgress.style.display = 'block';
        }
        const promptSection = document.getElementById('visualPromptSection');
        if (promptSection) {
            promptSection.style.display = 'none';
        }

        try {
            // ========== STEP 1: GENERATE VOICEOVER (0-15%) ==========
            this.updateProgressMultiStep(0, 'voiceover', 0);
            console.log('[1/3] Generating voiceover...');

            const voiceoverResponse = await this.api.generateVoiceover({
                script: projectData.script,
                projectId: projectData.projectId,
                voiceType: voiceType,
                targetDuration: projectData.duration
            });

            if (voiceoverResponse.status !== 'success') {
                throw new Error('Voiceover generation failed');
            }

            this.state.updateProject({
                audioPath: voiceoverResponse.data.audioPath,
                audioDuration: voiceoverResponse.data.actualDuration
            });
            this.updateProgressMultiStep(0, 'voiceover', 100);
            console.log('[1/3] Voiceover complete:', voiceoverResponse.data.audioPath);

            // ========== STEP 2: GENERATE VIDEO (15-85%) ==========
            this.updateProgressMultiStep(1, 'video', 0);

            let videoResponse;

            // Check if this is image-to-video mode
            if (projectData.reelType === 'image-to-video') {
                console.log('[2/3] Generating video from image...');

                // Validate image is uploaded
                if (!projectData.sourceImageData) {
                    throw new Error('Please upload a source image for Image-to-Video mode');
                }

                videoResponse = await this.api.generateVideoFromImage({
                    imageUrl: projectData.sourceImageData,  // Base64 data URL
                    prompt: visualPrompt || projectData.script?.substring(0, 200) || 'smooth cinematic motion',
                    projectId: projectData.projectId,
                    duration: projectData.duration,
                    aspectRatio: projectData.aspectRatio || '9:16',
                    resolution: projectData.videoQuality || '1080p'
                });
            } else {
                console.log('[2/3] Generating video...');

                videoResponse = await this.api.generateVideo({
                    script: projectData.script,
                    projectId: projectData.projectId,
                    duration: projectData.duration,
                    reelType: projectData.reelType,
                    aspectRatio: projectData.aspectRatio || '9:16',
                    styleTokens: visualPrompt || projectData.styleTokens || null,
                    quality: quality,
                    resolution: projectData.videoQuality || '1080p'
                });
            }

            if (videoResponse.status !== 'success') {
                throw new Error('Video generation failed');
            }

            this.state.updateProject({ jobId: videoResponse.data.jobId });

            // Poll for video completion
            const videoStatus = await this.videoPoller.poll(
                videoResponse.data.jobId,
                (progress) => this.updateProgressMultiStep(1, 'video', progress),
                async (jobId) => {
                    const response = await this.api.checkVideoStatus(jobId, projectData.projectId);
                    return response.data;
                }
            );

            if (!videoStatus.videoUrl) {
                throw new Error('No video URL received');
            }

            this.state.updateProject({ rawVideoUrl: videoStatus.videoUrl });
            this.updateProgressMultiStep(1, 'video', 100);
            console.log('[2/3] Video complete:', videoStatus.videoUrl);

            // ========== STEP 3: COMPOSE VIDEO + AUDIO (85-100%) ==========
            this.updateProgressMultiStep(2, 'compose', 0);
            console.log('[3/3] Composing final video with audio...');

            const composeResponse = await this.api.composeVideo({
                projectId: projectData.projectId,
                musicStyle: 'none'  // No background music, just voiceover
            });

            if (composeResponse.status !== 'success') {
                // If compose fails, still show the raw video
                console.warn('Composition failed, using raw video');
                this.state.updateProject({ videoUrl: videoStatus.videoUrl });
            } else {
                this.state.updateProject({ videoUrl: composeResponse.data.videoUrl });
            }

            this.updateProgressMultiStep(2, 'compose', 100);
            console.log('[3/3] Composition complete!');

            // ========== COMPLETE ==========
            const finalVideoUrl = this.state.state.projectData.videoUrl;

            // Update Stage 4 preview video
            if (this.elements.previewVideo) {
                this.elements.previewVideo.src = finalVideoUrl;
            }

            // Show Stage 3 video preview
            const stage3Preview = document.getElementById('stage3VideoPreview');
            const stage3Video = document.getElementById('stage3PreviewVideo');
            const stage3NextBtn = document.getElementById('stage3NextBtn');

            if (stage3Preview && stage3Video) {
                stage3Video.src = finalVideoUrl;
                stage3Preview.style.display = 'block';
                stage3Video.load();  // Load the video
            }

            // Hide generate button, show next button
            if (this.elements.generateBtn) {
                this.elements.generateBtn.style.display = 'none';
            }
            if (stage3NextBtn) {
                stage3NextBtn.style.display = 'inline-flex';
            }

            // Hide progress bar
            if (this.elements.generationProgress) {
                this.elements.generationProgress.style.display = 'none';
            }

            this.showNotification('Video generated successfully with voiceover!', 'success');

            // Stay on Stage 3 to let user preview (removed auto-navigation)

        } catch (error) {
            console.error('Video generation error:', error);
            this.handleError(error, 'Failed to generate video');

            // Reset UI on error
            if (this.elements.generateBtn) {
                this.elements.generateBtn.disabled = false;
                this.elements.generateBtn.style.display = 'inline-flex';
                this.elements.generateBtn.innerHTML = '<i class="fas fa-play"></i> Generate Video';
            }
        } finally {
            this.state.setState({ isGenerating: false });

            if (this.elements.generationProgress) {
                setTimeout(() => {
                    this.elements.generationProgress.style.display = 'none';
                }, 500);
            }
        }
    }

    /**
     * Update progress for multi-step generation (voiceover → video → compose)
     * Step 0: Voiceover (0-15%)
     * Step 1: Video (15-85%)
     * Step 2: Compose (85-100%)
     */
    updateProgressMultiStep(step, type, stepProgress) {
        let overallProgress = 0;
        let statusText = '';

        if (step === 0) {
            // Voiceover: 0-15%
            overallProgress = Math.round(stepProgress * 0.15);
            statusText = stepProgress < 100
                ? '<i class="fas fa-microphone fa-spin"></i> Generating voiceover...'
                : '<i class="fas fa-check"></i> Voiceover ready';
        } else if (step === 1) {
            // Video: 15-85%
            overallProgress = 15 + Math.round(stepProgress * 0.70);
            if (stepProgress < 20) {
                statusText = '<i class="fas fa-spinner fa-spin"></i> Starting video generation...';
            } else if (stepProgress < 60) {
                statusText = '<i class="fas fa-film fa-spin"></i> AI generating video frames...';
            } else if (stepProgress < 100) {
                statusText = '<i class="fas fa-video fa-spin"></i> Processing video...';
            } else {
                statusText = '<i class="fas fa-check"></i> Video ready';
            }
        } else if (step === 2) {
            // Compose: 85-100%
            overallProgress = 85 + Math.round(stepProgress * 0.15);
            statusText = stepProgress < 100
                ? '<i class="fas fa-compress fa-spin"></i> Combining video + audio...'
                : '<i class="fas fa-check-circle"></i> Complete!';
        }

        // Update progress bar
        const progressFill = document.getElementById('progressFill');
        if (progressFill) {
            progressFill.style.width = `${overallProgress}%`;
        }

        // Update percentage
        const progressPercentage = document.getElementById('progressPercentage');
        if (progressPercentage) {
            progressPercentage.textContent = `${overallProgress}%`;
        }

        // Update status
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            progressStatus.innerHTML = statusText;
        }

        // Update time estimate
        const progressTime = document.getElementById('progressTime');
        if (progressTime && overallProgress < 100) {
            const remainingPercent = 100 - overallProgress;
            // Estimate: voiceover ~5s, video ~60s, compose ~10s = ~75s total
            const totalSeconds = 75;
            const remainingSeconds = Math.ceil((remainingPercent / 100) * totalSeconds);
            if (remainingSeconds > 60) {
                const mins = Math.ceil(remainingSeconds / 60);
                progressTime.innerHTML = `<i class="fas fa-hourglass-half"></i> Est: ${mins} min remaining`;
            } else {
                progressTime.innerHTML = `<i class="fas fa-hourglass-end"></i> Est: ${remainingSeconds}s remaining`;
            }
        } else if (progressTime) {
            progressTime.innerHTML = '<i class="fas fa-check"></i> Processing complete!';
        }
    }

    /**
     * Update progress indicator with enhanced UI feedback
     * Production-ready for 50-100 concurrent users
     */
    updateProgress(stepIndex, type, percentage) {
        const progressItem = this.elements.progressItems[stepIndex];
        if (!progressItem) return;

        // Update progress fill bar
        const progressFill = document.getElementById('progressFill');
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }

        // Update percentage display
        const progressPercentage = document.getElementById('progressPercentage');
        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(percentage)}%`;
        }

        // Update status message based on progress
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            let statusText = '';
            if (percentage < 10) {
                statusText = '<i class="fas fa-clock"></i> Initializing...';
            } else if (percentage < 30) {
                statusText = '<i class="fas fa-spinner fa-spin"></i> AI analyzing your script...';
            } else if (percentage < 60) {
                statusText = '<i class="fas fa-film"></i> Generating video frames...';
            } else if (percentage < 90) {
                statusText = '<i class="fas fa-microphone"></i> Synthesizing audio...';
            } else if (percentage < 100) {
                statusText = '<i class="fas fa-compress"></i> Finalizing video...';
            } else {
                statusText = '<i class="fas fa-check-circle"></i> Complete!';
            }
            progressStatus.innerHTML = statusText;
        }

        // Update time estimate
        const progressTime = document.getElementById('progressTime');
        if (progressTime && percentage < 100) {
            const remainingPercent = 100 - percentage;
            const estimatedMinutes = Math.ceil((remainingPercent / 100) * 7); // Assumes 7 min total
            const estimatedSeconds = Math.ceil((remainingPercent / 100) * 420); // 7 min = 420 sec

            if (estimatedMinutes >= 1) {
                progressTime.innerHTML = `<i class="fas fa-hourglass-half"></i> Est: ${estimatedMinutes} min remaining`;
            } else if (estimatedSeconds > 0) {
                progressTime.innerHTML = `<i class="fas fa-hourglass-end"></i> Est: ${estimatedSeconds} sec remaining`;
            }
        } else if (progressTime && percentage === 100) {
            progressTime.innerHTML = '<i class="fas fa-check"></i> Processing complete!';
        }

        // Update icon when complete
        if (percentage === 100) {
            const icon = document.getElementById('progressIcon');
            if (icon) {
                icon.style.background = 'var(--success)';
                icon.innerHTML = '<i class="fas fa-check"></i>';
            }
        }
    }

    /**
     * Cancel generation
     */
    cancelGeneration() {
        if (this.state.state.projectData.jobId) {
            this.videoPoller.cancel(this.state.state.projectData.jobId);
        }

        this.state.setState({ isGenerating: false });
        this.showNotification('Generation cancelled', 'info');
    }

    /**
     * Search stock media
     */
    async searchStockMedia(query) {
        if (!query || query.trim() === '') {
            this.showNotification('Please enter a search query', 'warning');
            return;
        }

        this.showLoading('Searching stock media...');

        try {
            const results = await this.stockMedia.searchVideos(query.trim(), {
                perPage: 15,
                orientation: 'landscape'
            });

            this.state.updateProject({ stockVideos: results });
            this.showNotification(`Found ${results.length} stock videos`, 'success');

            // TODO: Display results in UI
            console.log('[StockMedia] Search results:', results);

        } catch (error) {
            console.error('Stock media search error:', error);
            this.handleError(error, 'Failed to search stock media');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Select stock video
     */
    selectStockVideo(videoId) {
        const video = this.stockMedia.getVideoById(videoId);
        if (!video) {
            this.showNotification('Video not found', 'error');
            return;
        }

        this.state.updateProject({ selectedStockVideo: video });
        this.showNotification(`Selected: ${video.id}`, 'success');

        console.log('[StockMedia] Selected video:', video);
    }

    /**
     * Regenerate voiceover
     */
    async regenVoiceover() {
        if (this.state.state.isGenerating) return;

        this.showLoading('Regenerating voice-over...');
        this.state.setState({ isGenerating: true });

        try {
            const response = await this.api.generateVoiceover({
                script: this.state.state.projectData.script,
                voiceType: this.elements.voiceType?.value || 'male-professional',
                projectId: this.state.state.projectData.projectId,
                regenerate: true
            });

            if (response.status === 'success') {
                this.state.updateProject({ audioPath: response.data.audioPath });
                this.elements.previewVideo?.load();
                this.showNotification('Voice-over regenerated!', 'success');
            }
        } catch (error) {
            this.handleError(error, 'Failed to regenerate voice-over');
        } finally {
            this.state.setState({ isGenerating: false });
            this.hideLoading();
        }
    }

    /**
     * Change music
     */
    async changeMusic() {
        // TODO: Implement music selector modal
        this.showNotification('Music selector coming soon!', 'info');
    }

    /**
     * Add captions
     */
    async addCaptions() {
        if (this.state.state.isGenerating) return;

        this.showLoading('Adding captions...');
        this.state.setState({ isGenerating: true });

        try {
            const response = await this.api.addCaptions({
                projectId: this.state.state.projectData.projectId,
                script: this.state.state.projectData.script
            });

            if (response.status === 'success') {
                // Update video preview
                if (this.elements.previewVideo) {
                    this.elements.previewVideo.src = response.data.videoUrl;
                }
                // Save captioned video URL to state for download/view
                this.state.updateProject({
                    videoUrl: response.data.videoUrl,
                    hasCaptions: true
                });
                this.showNotification('Captions added successfully!', 'success');
            }
        } catch (error) {
            this.handleError(error, 'Failed to add captions');
        } finally {
            this.state.setState({ isGenerating: false });
            this.hideLoading();
        }
    }

    // =========================================================================
    // IMAGE-TO-VIDEO FUNCTIONS
    // =========================================================================

    /**
     * Toggle image upload section based on reel type selection
     */
    toggleImageUpload() {
        const reelType = document.getElementById('reelType')?.value;
        const imageUploadSection = document.getElementById('imageUploadSection');

        if (imageUploadSection) {
            if (reelType === 'image-to-video') {
                imageUploadSection.style.display = 'block';
                this.state.updateProject({ reelType: 'image-to-video' });
            } else {
                imageUploadSection.style.display = 'none';
                // Clear any uploaded image when switching away
                const sourceImage = document.getElementById('sourceImage');
                const previewContainer = document.getElementById('imagePreviewContainer');
                if (sourceImage) sourceImage.value = '';
                if (previewContainer) previewContainer.style.display = 'none';
                this.state.updateProject({ sourceImage: null, sourceImageData: null });
            }
        }
    }

    /**
     * Preview uploaded source image for image-to-video
     */
    previewSourceImage(input) {
        const file = input?.files?.[0];
        const previewContainer = document.getElementById('imagePreviewContainer');
        const previewImg = document.getElementById('sourceImagePreview');
        const fileNameDisplay = document.querySelector('#imageUploadSection .file-name-display');

        if (!file) {
            if (previewContainer) previewContainer.style.display = 'none';
            if (fileNameDisplay) fileNameDisplay.textContent = 'Upload an image to animate';
            this.state.updateProject({ sourceImage: null, sourceImageData: null });
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showNotification('Please upload an image file (JPG, PNG, etc.)', 'warning');
            input.value = '';
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showNotification('Image too large. Maximum size is 10MB.', 'warning');
            input.value = '';
            return;
        }

        // Show file name
        if (fileNameDisplay) {
            fileNameDisplay.textContent = file.name;
            fileNameDisplay.style.color = 'var(--primary)';
        }

        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) {
                previewImg.src = e.target.result;
            }
            if (previewContainer) {
                previewContainer.style.display = 'block';
            }
            // Store image data for upload
            this.state.updateProject({
                sourceImage: file,
                sourceImageData: e.target.result  // Base64 data URL
            });
            this.showNotification('Image loaded! Ready for video generation.', 'success');
        };
        reader.onerror = () => {
            this.showNotification('Failed to read image file', 'error');
        };
        reader.readAsDataURL(file);
    }

    /**
     * Download video
     */
    downloadVideo(format = 'mp4') {
        const { projectId } = this.state.state.projectData;

        if (!projectId) {
            this.showNotification('No video to download', 'warning');
            return;
        }

        this.showNotification('Downloading video...', 'info');
        this.api.downloadVideo(projectId, format);
    }

    /**
     * Share video
     */
    async shareVideo(platform) {
        const { videoUrl } = this.state.state.projectData;

        if (!videoUrl) {
            this.showNotification('No video to share', 'warning');
            return;
        }

        if (platform === 'copy') {
            try {
                await navigator.clipboard.writeText(videoUrl);
                this.showNotification('Link copied to clipboard!', 'success');
            } catch (error) {
                this.showNotification('Failed to copy link', 'error');
            }
        } else {
            // TODO: Implement social sharing
            this.showNotification(`${platform} integration coming soon!`, 'info');
        }
    }

    /**
     * View video in browser
     */
    viewVideo() {
        const { videoUrl, projectId } = this.state.state.projectData;

        if (!videoUrl) {
            this.showNotification('No video available', 'warning');
            return;
        }

        // Build full URL for viewing
        let fullUrl = videoUrl;

        // If it's a relative URL, make it absolute
        if (videoUrl.startsWith('/')) {
            fullUrl = window.location.origin + videoUrl;
        }

        // Open video in new tab (no echo notification needed)
        window.open(fullUrl, '_blank');
    }

    /**
     * Create new project
     */
    createNew() {
        if (this.state.state.projectData.projectId) {
            if (!confirm('Start a new project? Current progress will be saved.')) {
                return;
            }

            // Save current project to history
            this.state.addToHistory(this.state.state.projectData);
        }

        // Cancel any active polling
        this.videoPoller.cancelAll();

        // Reset state
        this.state.reset();

        // Reset UI
        this.resetForm();

        this.goToStage(1);
        this.showNotification('New project started!', 'success');
    }

    /**
     * Reset form
     */
    resetForm() {
        if (this.elements.topic) this.elements.topic.value = '';
        if (this.elements.brand) this.elements.brand.value = '';
        if (this.elements.scriptPreview) this.elements.scriptPreview.value = '';
        if (this.elements.previewVideo) this.elements.previewVideo.src = '';
        if (this.elements.brandGuidelines) this.elements.brandGuidelines.value = '';
        if (this.elements.fileNameDisplay) {
            this.elements.fileNameDisplay.textContent = 'No file chosen';
            this.elements.fileNameDisplay.style.color = 'var(--gray)';
        }
    }

    /**
     * Restore state from localStorage
     */
    restoreState() {
        if (this.state.load()) {
            const { projectData, currentStage } = this.state.state;

            // Restore form values
            if (this.elements.topic && projectData.topic) {
                this.elements.topic.value = projectData.topic;
            }
            if (this.elements.brand && projectData.brand) {
                this.elements.brand.value = projectData.brand;
            }
            if (this.elements.scriptPreview && projectData.script) {
                this.elements.scriptPreview.value = projectData.script;
            }
            if (this.elements.previewVideo && projectData.videoUrl) {
                this.elements.previewVideo.src = projectData.videoUrl;
            }

            // Restore stage
            if (currentStage > 1) {
                this.goToStage(currentStage);
            }

            this.showNotification('Previous session restored', 'info');
        }
    }

    /**
     * Setup auto-save
     */
    setupAutoSave() {
        setInterval(() => {
            if (this.state.state.projectData.projectId && this.state.state.settings.autoSave) {
                this.state.save();
            }
        }, CONFIG.AUTO_SAVE_INTERVAL);
    }

    /**
     * Update UI based on state
     */
    updateUI(state) {
        // Update generation button
        if (this.elements.generateBtn) {
            this.elements.generateBtn.disabled = state.isGenerating;
        }

        // Update connection status
        const connectionBanner = document.getElementById('connectionBanner');
        if (connectionBanner) {
            connectionBanner.style.display = state.isConnected ? 'none' : 'block';
        }
    }

    /**
     * Scroll to generator
     */
    scrollToGenerator() {
        document.getElementById('create')?.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Play demo
     */
    playDemo() {
        // TODO: Implement demo video player
        this.showNotification('Demo video coming soon!', 'info');
    }

    /**
     * Show loading overlay
     */
    showLoading(message = 'Processing...') {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.classList.add('active');
            if (this.elements.loadingText) {
                this.elements.loadingText.textContent = message;
            }
        }
    }

    /**
     * Hide loading overlay
     */
    hideLoading() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.classList.remove('active');
        }
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        if (!this.state.state.settings.notifications) return;

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;

        // Add icon
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };

        notification.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <span>${message}</span>
        `;

        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            animation: slideIn 0.3s ease;
            max-width: 400px;
            font-weight: 500;
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, CONFIG.NOTIFICATION_DURATION);
    }

    /**
     * Handle errors
     */
    handleError(error, fallbackMessage) {
        console.error(error);

        let message = fallbackMessage;

        if (error.code === 'NETWORK_ERROR') {
            message = 'Cannot connect to server. Please check your connection.';
        } else if (error.code === 'TIMEOUT') {
            message = 'Request timed out. Please try again.';
        } else if (error.code === 'RATE_LIMIT_EXCEEDED') {
            message = 'Too many requests. Please wait a moment.';
        } else if (error.message) {
            message = error.message;
        }

        this.showNotification(message, 'error');

        // Log error details for debugging
        if (error.requestId) {
            console.error(`Request ID: ${error.requestId}`);
        }
    }

    /**
     * Initialize particle animation
     */
    initParticles() {
        const canvas = document.getElementById('particlesCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        const particles = [];
        const particleCount = 80;

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2 + 1;
                this.speedX = Math.random() * 0.5 - 0.25;
                this.speedY = Math.random() * 0.5 - 0.25;
                this.opacity = Math.random() * 0.5 + 0.2;
            }

            update() {
                this.x += this.speedX;
                this.y += this.speedY;

                if (this.x > canvas.width) this.x = 0;
                if (this.x < 0) this.x = canvas.width;
                if (this.y > canvas.height) this.y = 0;
                if (this.y < 0) this.y = canvas.height;
            }

            draw() {
                ctx.fillStyle = `rgba(99, 102, 241, ${this.opacity})`;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach(particle => {
                particle.update();
                particle.draw();
            });

            // Draw connections
            particles.forEach((p1, i) => {
                particles.slice(i + 1).forEach(p2 => {
                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 120) {
                        ctx.strokeStyle = `rgba(99, 102, 241, ${0.15 * (1 - distance / 120)})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                });
            });

            requestAnimationFrame(animate);
        }

        animate();

        // Handle resize
        window.addEventListener('resize', () => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        });
    }

    /**
     * Throttle function for performance
     */
    throttle(func, wait) {
        let lastTime = 0;
        return function(...args) {
            const now = Date.now();
            if (now - lastTime >= wait) {
                lastTime = now;
                func.apply(this, args);
            }
        };
    }

    /**
     * Debounce function for performance
     */
    debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }
}

// === INITIALIZATION ===
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    const apiClient = new ApiClient();
    const stateManager = new AppStateManager();
    const uiController = new UIController(stateManager, apiClient);

    // Initialize UI
    uiController.init();

    // Check backend connection
    try {
        const health = await apiClient.checkHealth();

        if (health.status === 'success' && health.data) {
            const { status, services } = health.data;

            stateManager.setState({
                isConnected: true,
                services: services || {}
            });

            console.log('✅ Backend connected:', status);

            // Check service availability
            if (services) {
                if (!services.groq) {
                    uiController.showNotification('⚠️ Groq service unavailable - add API key to .env', 'warning');
                }
                // Video service: demo mode is always available
                if (!services.replicate && !services.runpod && !services.demo && !services.video_provider) {
                    uiController.showNotification('⚠️ Video service unavailable - add API key to .env', 'warning');
                }
                if (!services.tts && !services.kokoro && !services.edge_tts) {
                    uiController.showNotification('⚠️ TTS service unavailable', 'warning');
                }
            }
        }
    } catch (error) {
        console.error('❌ Backend connection failed:', error);
        stateManager.setState({ isConnected: false });

        // Show connection error
        uiController.showNotification('⚠️ Backend not connected! Please start the server.', 'error');

        // Show connection banner
        const banner = document.createElement('div');
        banner.id = 'connectionBanner';
        banner.style.cssText = `
            position: fixed;
            top: 80px;
            left: 0;
            right: 0;
            background: #dc2626;
            color: white;
            padding: 16px;
            text-align: center;
            z-index: 9999;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        `;
        banner.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>BACKEND NOT RUNNING - Please run the start script to launch the server</span>
        `;
        document.body.insertBefore(banner, document.body.firstChild);
    }

    // Export for global access
    window.ReelCraftApp = {
        api: apiClient,
        state: stateManager,
        ui: uiController,
        videoPoller: uiController.videoPoller,
        stockMedia: uiController.stockMedia
    };

    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
});
