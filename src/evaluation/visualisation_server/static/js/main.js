// Nightingale Visualization Server
// Minimal JavaScript for essential functionality

document.addEventListener('DOMContentLoaded', function() {
    console.log('Nightingale Server Loaded');
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    checkServerStatus();
}

function setupEventListeners() {
    // Status button functionality
    const statusBtn = document.getElementById('status-btn');
    if (statusBtn) {
        statusBtn.addEventListener('click', function() {
            checkServerStatus();
        });
    }

    // Simple navigation enhancement
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
}

async function checkServerStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (data.status === 'healthy') {
            showNotification('Server Status: Healthy', 'success');
        } else {
            showNotification('Server Status: Unknown', 'warning');
        }
    } catch (error) {
        showNotification('Server Status: Error', 'error');
        console.error('Status check failed:', error);
    }
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) {
        existing.remove();
    }

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Style the notification with brain-inspired colors
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 16px',
        backgroundColor: type === 'success' ? 'var(--color-primary)' :
                        type === 'error' ? '#ef4444' :
                        type === 'warning' ? '#f59e0b' : 'var(--color-accent)',
        color: 'white',
        fontFamily: 'var(--font-mono)',
        fontSize: 'var(--font-size-sm)',
        borderRadius: 'var(--border-radius)',
        zIndex: '1000',
        animation: 'slideIn 0.3s ease-out'
    });

    document.body.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease-out forwards';
            setTimeout(() => notification.remove(), 300);
        }
    }, 3000);
}

// Add minimal animations via JavaScript
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

// Export minimal API
window.Nightingale = {
    checkStatus: checkServerStatus,
    notify: showNotification
};
