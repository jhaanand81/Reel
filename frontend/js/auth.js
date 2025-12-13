/**
 * REEL SENSE - Authentication JavaScript
 * Handles login, register, and API key management
 */

(function() {
    'use strict';

    // Configuration - Use dynamic origin, never hardcoded URLs
    const origin = window.location.origin;
    const API_BASE = window.REELSENSE_CONFIG?.API_BASE_URL || `${origin}/api`;
    const API_VERSION = 'v1';

    // DOM Elements
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const authTabs = document.querySelectorAll('.auth-tab');
    const apiKeySection = document.getElementById('apiKeySection');
    const inviteCodeGroup = document.getElementById('inviteCodeGroup');

    // Beta mode state
    let isBetaMode = true; // Default to true, will be updated from server

    // ===========================================================================
    // BETA MODE CHECK
    // ===========================================================================

    async function checkBetaStatus() {
        try {
            const response = await fetch(`${API_BASE}/${API_VERSION}/auth/beta-status`);
            const data = await response.json();

            if (data.status === 'success') {
                isBetaMode = data.data.beta_mode;

                // Show/hide invite code field based on beta mode
                if (inviteCodeGroup) {
                    if (isBetaMode) {
                        inviteCodeGroup.classList.remove('hidden');
                        document.getElementById('inviteCode').required = true;
                    } else {
                        inviteCodeGroup.classList.add('hidden');
                        document.getElementById('inviteCode').required = false;
                    }
                }
            }
        } catch (error) {
            console.log('Beta status check failed, defaulting to beta mode');
            // Keep beta mode enabled by default for security
        }
    }

    // Check beta status on load
    checkBetaStatus();

    // ===========================================================================
    // TAB SWITCHING
    // ===========================================================================

    authTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            // Update tab active state
            authTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show corresponding form
            document.querySelectorAll('.auth-form').forEach(form => {
                form.classList.remove('active');
            });

            if (targetTab === 'login') {
                loginForm.classList.add('active');
            } else {
                registerForm.classList.add('active');
            }

            // Clear messages
            clearMessages();
        });
    });

    // ===========================================================================
    // LOGIN
    // ===========================================================================

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearMessages();

        const email = document.getElementById('loginEmail').value.trim();
        const password = document.getElementById('loginPassword').value;
        const submitBtn = loginForm.querySelector('button[type="submit"]');

        // Validate
        if (!email || !password) {
            showError('loginError', 'Please fill in all fields');
            return;
        }

        // Show loading
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/${API_VERSION}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                // Store tokens
                localStorage.setItem('access_token', data.data.access_token);
                localStorage.setItem('refresh_token', data.data.refresh_token);
                localStorage.setItem('api_key', data.data.api_key);
                localStorage.setItem('user', JSON.stringify(data.data.user));
                // Also store as authToken for admin.js compatibility
                localStorage.setItem('authToken', data.data.access_token);

                // Show success and API key
                showSuccess('loginSuccess', 'Login successful!');
                showApiKey(data.data.api_key);

                // Redirect after delay - admin users go to admin dashboard
                const userRole = data.data.user?.role || 'user';
                const redirectUrl = ['admin', 'superadmin'].includes(userRole) ? '/admin.html' : '/';

                setTimeout(() => {
                    window.location.href = redirectUrl;
                }, 2000);
            } else {
                // Handle specific error cases
                if (data.error_code === 'USER_NOT_FOUND') {
                    showError('loginError', data.error + ' Click "Create Account" to register.');
                    // Highlight the register tab
                    document.querySelector('.auth-tab[data-tab="register"]').classList.add('pulse-highlight');
                    setTimeout(() => {
                        document.querySelector('.auth-tab[data-tab="register"]').classList.remove('pulse-highlight');
                    }, 3000);
                } else {
                    showError('loginError', data.error || 'Login failed');
                }
            }
        } catch (error) {
            console.error('Login error:', error);
            showError('loginError', 'Connection error. Please try again.');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });

    // ===========================================================================
    // REGISTER
    // ===========================================================================

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearMessages();

        const inviteCode = document.getElementById('inviteCode')?.value.trim().toUpperCase() || '';
        const name = document.getElementById('registerName').value.trim();
        const email = document.getElementById('registerEmail').value.trim();
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const agreeTerms = document.getElementById('agreeTerms').checked;
        const submitBtn = registerForm.querySelector('button[type="submit"]');

        // Validate invite code in beta mode
        if (isBetaMode && !inviteCode) {
            showError('registerError', 'Please enter your beta invite code');
            document.getElementById('inviteCode')?.focus();
            return;
        }

        // Validate
        if (!name || !email || !password || !confirmPassword) {
            showError('registerError', 'Please fill in all fields');
            return;
        }

        if (password.length < 8) {
            showError('registerError', 'Password must be at least 8 characters');
            return;
        }

        if (password !== confirmPassword) {
            showError('registerError', 'Passwords do not match');
            return;
        }

        if (!agreeTerms) {
            showError('registerError', 'Please agree to the Terms of Service');
            return;
        }

        // Show loading
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/${API_VERSION}/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name, email, password, invite_code: inviteCode })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                // Store tokens
                localStorage.setItem('access_token', data.data.access_token);
                localStorage.setItem('refresh_token', data.data.refresh_token);
                localStorage.setItem('api_key', data.data.api_key);
                localStorage.setItem('user', JSON.stringify(data.data.user));

                // Show success and API key
                showSuccess('registerSuccess', 'Account created successfully!');
                showApiKey(data.data.api_key);

            } else {
                showError('registerError', data.error || 'Registration failed');
            }
        } catch (error) {
            console.error('Register error:', error);
            showError('registerError', 'Connection error. Please try again.');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });

    // ===========================================================================
    // PASSWORD STRENGTH
    // ===========================================================================

    const passwordInput = document.getElementById('registerPassword');
    const strengthIndicator = document.getElementById('passwordStrength');

    if (passwordInput && strengthIndicator) {
        passwordInput.addEventListener('input', () => {
            const password = passwordInput.value;
            const strength = calculatePasswordStrength(password);

            strengthIndicator.className = 'password-strength';
            if (password.length > 0) {
                if (strength < 3) {
                    strengthIndicator.classList.add('weak');
                } else if (strength < 5) {
                    strengthIndicator.classList.add('medium');
                } else {
                    strengthIndicator.classList.add('strong');
                }
            }
        });
    }

    function calculatePasswordStrength(password) {
        let strength = 0;

        if (password.length >= 8) strength++;
        if (password.length >= 12) strength++;
        if (/[a-z]/.test(password)) strength++;
        if (/[A-Z]/.test(password)) strength++;
        if (/[0-9]/.test(password)) strength++;
        if (/[^a-zA-Z0-9]/.test(password)) strength++;

        return strength;
    }

    // ===========================================================================
    // HELPER FUNCTIONS
    // ===========================================================================

    function showError(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = message;
            element.classList.remove('hidden');
        }
    }

    function showSuccess(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = message;
            element.classList.remove('hidden');
        }
    }

    function clearMessages() {
        document.querySelectorAll('.error-message, .success-message').forEach(el => {
            el.classList.add('hidden');
            el.textContent = '';
        });
    }

    function showApiKey(apiKey) {
        // Hide forms
        loginForm.classList.remove('active');
        registerForm.classList.remove('active');
        document.querySelector('.auth-tabs').style.display = 'none';

        // Show API key section
        document.getElementById('apiKeyValue').textContent = apiKey;
        apiKeySection.classList.remove('hidden');
    }

    // ===========================================================================
    // GLOBAL FUNCTIONS (defined early for onclick handlers)
    // ===========================================================================

    // Store reset token for submission
    let currentResetToken = null;

    // Show forgot password form
    window.showForgotPassword = function() {
        console.log('showForgotPassword called');
        clearMessages();
        loginForm.classList.remove('active');
        registerForm.classList.remove('active');
        const authTabs = document.querySelector('.auth-tabs');
        if (authTabs) authTabs.style.display = 'none';
        const forgotSection = document.getElementById('forgotPasswordSection');
        if (forgotSection) {
            forgotSection.classList.remove('hidden');
            forgotSection.classList.add('active');
        }
    };

    // Go back to login
    window.backToLogin = function() {
        clearMessages();
        const forgotSection = document.getElementById('forgotPasswordSection');
        const resetSection = document.getElementById('resetPasswordSection');
        if (forgotSection) {
            forgotSection.classList.add('hidden');
            forgotSection.classList.remove('active');
        }
        if (resetSection) {
            resetSection.classList.add('hidden');
            resetSection.classList.remove('active');
        }
        const authTabs = document.querySelector('.auth-tabs');
        if (authTabs) authTabs.style.display = 'flex';
        loginForm.classList.add('active');
    };

    // Toggle password visibility
    window.togglePassword = function(inputId) {
        const input = document.getElementById(inputId);
        const button = input.parentElement.querySelector('.password-toggle');
        const eyeIcon = button.querySelector('.eye-icon');
        const eyeOffIcon = button.querySelector('.eye-off-icon');

        if (input.type === 'password') {
            input.type = 'text';
            eyeIcon.classList.add('hidden');
            eyeOffIcon.classList.remove('hidden');
        } else {
            input.type = 'password';
            eyeIcon.classList.remove('hidden');
            eyeOffIcon.classList.add('hidden');
        }
    };

    // Copy API key
    window.copyApiKey = function() {
        const apiKey = document.getElementById('apiKeyValue').textContent;
        navigator.clipboard.writeText(apiKey).then(() => {
            // Show feedback
            const btn = document.querySelector('.api-key-display button');
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i>';
            btn.style.background = '#10b981';

            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.style.background = '';
            }, 2000);
        });
    };

    // ===========================================================================
    // SOCIAL LOGIN (OAuth)
    // ===========================================================================

    // Google Sign-In
    window.signInWithGoogle = function() {
        // For now, show a message that OAuth is coming soon
        // In production, this would redirect to Google OAuth
        const message = 'Google Sign-In is coming soon! Please use email/password for now.';

        // Option 1: Show alert
        // alert(message);

        // Option 2: Redirect to Google OAuth (placeholder URL)
        // In production, replace with actual OAuth endpoint from backend
        const googleAuthUrl = `${API_BASE}/${API_VERSION}/auth/google`;

        // Check if backend supports Google OAuth
        fetch(googleAuthUrl, { method: 'GET' })
            .then(response => {
                if (response.ok) {
                    window.location.href = googleAuthUrl;
                } else {
                    showError('loginError', message);
                }
            })
            .catch(() => {
                showError('loginError', message);
            });
    };

    // Microsoft Sign-In
    window.signInWithMicrosoft = function() {
        // For now, show a message that OAuth is coming soon
        const message = 'Microsoft Sign-In is coming soon! Please use email/password for now.';

        const microsoftAuthUrl = `${API_BASE}/${API_VERSION}/auth/microsoft`;

        fetch(microsoftAuthUrl, { method: 'GET' })
            .then(response => {
                if (response.ok) {
                    window.location.href = microsoftAuthUrl;
                } else {
                    showError('loginError', message);
                }
            })
            .catch(() => {
                showError('loginError', message);
            });
    };

    // ===========================================================================
    // CHECK EXISTING SESSION
    // ===========================================================================

    function checkExistingSession() {
        const token = localStorage.getItem('access_token');
        const user = localStorage.getItem('user');

        if (token && user) {
            // Already logged in, redirect to home
            // window.location.href = '/';
        }
    }

    // Check on load
    checkExistingSession();

    // ===========================================================================
    // PASSWORD RESET API FUNCTIONS
    // ===========================================================================

    // Request password reset
    window.requestPasswordReset = async function() {
        clearMessages();
        const email = document.getElementById('forgotEmail').value.trim();

        if (!email) {
            showError('forgotError', 'Please enter your email address');
            return;
        }

        const btn = document.querySelector('#forgotPasswordSection .btn-primary');
        btn.classList.add('loading');
        btn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/${API_VERSION}/auth/forgot-password`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email })
            });

            const data = await response.json();

            if (data.status === 'success') {
                if (data.data && data.data.reset_token) {
                    // For beta: Show the reset form directly
                    showSuccess('forgotSuccess', 'Reset link generated! Redirecting...');
                    currentResetToken = data.data.reset_token;

                    setTimeout(() => {
                        showResetPasswordForm(data.data.reset_token, email);
                    }, 1000);
                } else {
                    showSuccess('forgotSuccess', data.message || 'Check your email for reset instructions.');
                }
            } else {
                showError('forgotError', data.error || 'Failed to request password reset');
            }
        } catch (error) {
            console.error('Forgot password error:', error);
            showError('forgotError', 'Connection error. Please try again.');
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    };

    // Show reset password form
    function showResetPasswordForm(token, email) {
        clearMessages();
        currentResetToken = token;
        document.getElementById('forgotPasswordSection').classList.add('hidden');
        document.getElementById('forgotPasswordSection').classList.remove('active');
        document.getElementById('resetPasswordSection').classList.remove('hidden');
        document.getElementById('resetPasswordSection').classList.add('active');
        document.getElementById('resetEmailDisplay').textContent = `Resetting password for: ${email}`;
    }

    // Submit new password
    window.submitPasswordReset = async function() {
        clearMessages();
        const newPassword = document.getElementById('newPassword').value;
        const confirmPassword = document.getElementById('confirmNewPassword').value;

        if (!newPassword || !confirmPassword) {
            showError('resetError', 'Please fill in both password fields');
            return;
        }

        if (newPassword.length < 8) {
            showError('resetError', 'Password must be at least 8 characters');
            return;
        }

        if (newPassword !== confirmPassword) {
            showError('resetError', 'Passwords do not match');
            return;
        }

        if (!currentResetToken) {
            showError('resetError', 'Reset token is missing. Please request a new reset link.');
            return;
        }

        const btn = document.querySelector('#resetPasswordSection .btn-primary');
        btn.classList.add('loading');
        btn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/${API_VERSION}/auth/reset-password`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    reset_token: currentResetToken,
                    new_password: newPassword
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                showSuccess('resetSuccess', data.message || 'Password reset successful!');
                currentResetToken = null;

                // Redirect to login after success
                setTimeout(() => {
                    backToLogin();
                    showSuccess('loginSuccess', 'Password reset! Please sign in with your new password.');
                }, 2000);
            } else {
                showError('resetError', data.error || 'Failed to reset password');
            }
        } catch (error) {
            console.error('Reset password error:', error);
            showError('resetError', 'Connection error. Please try again.');
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    };

    // Check for reset token in URL on page load
    function checkResetToken() {
        const urlParams = new URLSearchParams(window.location.search);
        const resetToken = urlParams.get('reset_token');

        if (resetToken) {
            // Validate the token first
            fetch(`${API_BASE}/${API_VERSION}/auth/validate-reset-token/${resetToken}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success' && data.data.valid) {
                        // Hide tabs and show reset form
                        document.querySelector('.auth-tabs').style.display = 'none';
                        loginForm.classList.remove('active');
                        showResetPasswordForm(resetToken, data.data.email);
                    } else {
                        showError('loginError', data.error || 'Invalid or expired reset link');
                    }
                })
                .catch(error => {
                    console.error('Token validation error:', error);
                    showError('loginError', 'Failed to validate reset link');
                });
        }
    }

    // Check for reset token on load
    checkResetToken();

})();
