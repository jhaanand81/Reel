/**
 * REELCRAFT AI - Frontend Configuration
 * Fully Dynamic URL configuration - works on any host/IP
 */

(function() {
    'use strict';

    // Get current origin dynamically (works on localhost, LAN IP, or production domain)
    const origin = window.location.origin; // e.g., "http://10.5.112.125:5000" or "https://contentsense.ai"
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;

    // Environment detection
    const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1';
    const isLanIP = /^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)/.test(hostname);
    const isDevelopment = isLocalhost || isLanIP || hostname.includes('.local');
    const isProduction = !isDevelopment;

    // API Base URL - Always use current origin for maximum compatibility
    let apiBaseUrl;

    if (window.__API_BASE_URL__) {
        // Use injected config (from server-side rendering or build)
        apiBaseUrl = window.__API_BASE_URL__;
    } else {
        // Use current origin - works everywhere (localhost, LAN IP, production)
        apiBaseUrl = `${origin}/api`;
    }

    // Static assets URL - Always use current origin
    let staticBaseUrl;

    if (window.__STATIC_BASE_URL__) {
        staticBaseUrl = window.__STATIC_BASE_URL__;
    } else {
        // Use current origin - works everywhere
        staticBaseUrl = origin;
    }

    // Export configuration
    window.REELSENSE_CONFIG = {
        // Environment
        ENV: isDevelopment ? 'development' : 'production',
        IS_PRODUCTION: isProduction,
        IS_DEVELOPMENT: isDevelopment,

        // Current Origin (for debugging)
        ORIGIN: origin,

        // API Configuration
        API_BASE_URL: apiBaseUrl,
        API_VERSION: 'v1',
        API_TIMEOUT: 60000,

        // Static Assets
        STATIC_BASE_URL: staticBaseUrl,

        // Video URLs helper - handles both relative and absolute URLs
        getVideoUrl: function(path) {
            if (!path) return '';
            // Already absolute URL
            if (path.startsWith('http://') || path.startsWith('https://')) {
                return path;
            }
            // Relative URL - prepend static base
            const cleanPath = path.startsWith('/') ? path : `/${path}`;
            return `${this.STATIC_BASE_URL}${cleanPath}`;
        },

        // Output file URL helper
        getOutputUrl: function(filename) {
            if (!filename) return '';
            const cleanFilename = filename.startsWith('/') ? filename.slice(1) : filename;
            return `${this.STATIC_BASE_URL}/outputs/${cleanFilename}`;
        },

        // API URL helper
        getApiUrl: function(endpoint) {
            const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
            return `${this.API_BASE_URL}/${this.API_VERSION}${cleanEndpoint}`;
        },

        // Polling Configuration
        POLL_INTERVAL: 800,
        POLL_INTERVAL_SLOW: 3000,
        POLL_SLOWDOWN_AFTER: 60000,
        MAX_POLL_DURATION: 900000,

        // Upload Configuration
        MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
        ALLOWED_FILE_TYPES: [
            'application/pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ],

        // Feature Flags
        FEATURES: {
            ENABLE_CAPTIONS: true,
            ENABLE_STOCK_MEDIA: true,
            ENABLE_MUSIC: false, // Coming soon
            ENABLE_SOCIAL_SHARE: false, // Coming soon
        },

        // Debug
        DEBUG: isDevelopment,

        // Version
        VERSION: '2.0.0'
    };

    // Log configuration in development
    if (window.REELSENSE_CONFIG.DEBUG) {
        console.log('%c[REELSENSE CONFIG]', 'color: #826c6c; font-weight: bold', {
            origin: origin,
            apiBaseUrl: apiBaseUrl,
            staticBaseUrl: staticBaseUrl,
            env: window.REELSENSE_CONFIG.ENV
        });
    }

    // Freeze config to prevent modification
    Object.freeze(window.REELSENSE_CONFIG);
    Object.freeze(window.REELSENSE_CONFIG.FEATURES);

})();
