// Service Worker for Reel Sense PWA
// Production-grade with robust caching for 30-40 concurrent users

const CACHE_VERSION = 'v2';
const CACHE_NAME = `reelsense-${CACHE_VERSION}`;
const API_CACHE_NAME = `reelsense-api-${CACHE_VERSION}`;

// Static assets to cache immediately
const STATIC_ASSETS = [
    '/mobile/',
    '/mobile/index.html',
    '/mobile/manifest.json',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css',
    'https://unpkg.com/html5-qrcode@2.3.8/html5-qrcode.min.js'
];

// API endpoints to cache
const CACHEABLE_API_PATTERNS = [
    /\/api\/v1\/auth\/me$/,
    /\/api\/v1\/projects$/,
    /\/api\/v1\/health$/,
];

// Cache duration settings
const CACHE_DURATIONS = {
    static: 7 * 24 * 60 * 60 * 1000, // 7 days for static assets
    api: 5 * 60 * 1000,              // 5 minutes for API responses
    image: 24 * 60 * 60 * 1000,      // 24 hours for images
};

// Install event - cache static assets
self.addEventListener('install', event => {
    console.log('[SW] Installing service worker...');

    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('[SW] Static assets cached');
                return self.skipWaiting(); // Activate immediately
            })
            .catch(err => {
                console.error('[SW] Cache failed:', err);
            })
    );
});

// Activate event - cleanup old caches
self.addEventListener('activate', event => {
    console.log('[SW] Activating service worker...');

    event.waitUntil(
        caches.keys()
            .then(cacheNames => {
                return Promise.all(
                    cacheNames
                        .filter(name => {
                            return name.startsWith('reelsense-') &&
                                   name !== CACHE_NAME &&
                                   name !== API_CACHE_NAME;
                        })
                        .map(name => {
                            console.log('[SW] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[SW] Old caches cleared');
                return self.clients.claim(); // Take control immediately
            })
    );
});

// Fetch event - handle requests with appropriate strategy
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }

    // API requests - Network first, fall back to cache
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(handleApiRequest(request));
        return;
    }

    // Static assets - Cache first, fall back to network
    if (isStaticAsset(url)) {
        event.respondWith(handleStaticRequest(request));
        return;
    }

    // Default - Network first with cache fallback
    event.respondWith(handleDefaultRequest(request));
});

// Handle API requests - Network first strategy
async function handleApiRequest(request) {
    const url = new URL(request.url);
    const isCacheable = CACHEABLE_API_PATTERNS.some(pattern => pattern.test(url.pathname));

    try {
        const response = await fetch(request);

        // Cache successful cacheable responses
        if (response.ok && isCacheable) {
            const cache = await caches.open(API_CACHE_NAME);
            const clonedResponse = response.clone();

            // Add timestamp header for cache expiration
            const headers = new Headers(clonedResponse.headers);
            headers.set('sw-cached-at', Date.now().toString());

            const cachedResponse = new Response(await clonedResponse.blob(), {
                status: clonedResponse.status,
                statusText: clonedResponse.statusText,
                headers: headers
            });

            cache.put(request, cachedResponse);
        }

        return response;
    } catch (error) {
        console.log('[SW] Network failed for API, checking cache:', url.pathname);

        // Try to serve from cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            const cachedAt = parseInt(cachedResponse.headers.get('sw-cached-at') || '0');
            const age = Date.now() - cachedAt;

            // Check if cache is still valid
            if (age < CACHE_DURATIONS.api) {
                console.log('[SW] Serving API from cache');
                return cachedResponse;
            }
        }

        // Return offline response for API
        return new Response(
            JSON.stringify({
                status: 'error',
                error: 'Offline',
                error_code: 'OFFLINE'
            }),
            {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            }
        );
    }
}

// Handle static requests - Cache first strategy
async function handleStaticRequest(request) {
    const cachedResponse = await caches.match(request);

    if (cachedResponse) {
        // Refresh cache in background
        refreshCache(request);
        return cachedResponse;
    }

    try {
        const response = await fetch(request);

        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
        }

        return response;
    } catch (error) {
        console.log('[SW] Static fetch failed:', request.url);

        // Return offline page for navigation requests
        if (request.mode === 'navigate') {
            return caches.match('/mobile/index.html');
        }

        return new Response('Offline', { status: 503 });
    }
}

// Handle default requests - Network first with cache fallback
async function handleDefaultRequest(request) {
    try {
        const response = await fetch(request);

        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
        }

        return response;
    } catch (error) {
        const cachedResponse = await caches.match(request);

        if (cachedResponse) {
            return cachedResponse;
        }

        // Return offline page for navigation
        if (request.mode === 'navigate') {
            return caches.match('/mobile/index.html');
        }

        return new Response('Offline', { status: 503 });
    }
}

// Check if URL is a static asset
function isStaticAsset(url) {
    const staticExtensions = ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.woff', '.woff2', '.ttf'];
    return staticExtensions.some(ext => url.pathname.endsWith(ext)) ||
           url.hostname !== self.location.hostname;
}

// Refresh cache in background
async function refreshCache(request) {
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response);
        }
    } catch (error) {
        // Silent fail - cache refresh is best effort
    }
}

// Handle background sync for offline actions
self.addEventListener('sync', event => {
    console.log('[SW] Background sync triggered:', event.tag);

    if (event.tag === 'sync-pending-requests') {
        event.waitUntil(syncPendingRequests());
    }
});

// Sync pending requests when back online
async function syncPendingRequests() {
    console.log('[SW] Syncing pending requests...');
}

// Handle push notifications (future feature)
self.addEventListener('push', event => {
    if (event.data) {
        const data = event.data.json();

        event.waitUntil(
            self.registration.showNotification(data.title, {
                body: data.body,
                icon: '/mobile/icons/icon-192.png',
                badge: '/mobile/icons/icon-72.png',
                tag: data.tag || 'reelsense-notification',
                data: data.url || '/'
            })
        );
    }
});

// Handle notification click
self.addEventListener('notificationclick', event => {
    event.notification.close();

    event.waitUntil(
        clients.matchAll({ type: 'window' })
            .then(windowClients => {
                for (const client of windowClients) {
                    if (client.url === event.notification.data && 'focus' in client) {
                        return client.focus();
                    }
                }

                if (clients.openWindow) {
                    return clients.openWindow(event.notification.data);
                }
            })
    );
});

console.log('[SW] Service Worker loaded');
