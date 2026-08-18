/**
 * Scholar AI - Frontend Interactivity
 * Handles theme switching, sidebar toggle, and custom UI interactions
 */

// ============================================================
// THEME MANAGEMENT
// ============================================================

function initTheme() {
    const savedTheme = localStorage.getItem('scholar-ai-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    return savedTheme;
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('scholar-ai-theme', newTheme);
    
    // Dispatch event for Streamlit
    if (window.parent) {
        window.parent.postMessage({ type: 'toggleTheme', theme: newTheme }, '*');
    }
}

// ============================================================
// SIDEBAR MANAGEMENT
// ============================================================

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    const queryBar = document.querySelector('.query-bar-container');
    
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
    queryBar.classList.toggle('expanded');
    
    const isCollapsed = sidebar.classList.contains('collapsed');
    localStorage.setItem('scholar-ai-sidebar-collapsed', isCollapsed);
    
    if (window.parent) {
        window.parent.postMessage({ type: 'toggleSidebar', collapsed: isCollapsed }, '*');
    }
}

function initSidebar() {
    const isCollapsed = localStorage.getItem('scholar-ai-sidebar-collapsed') === 'true';
    if (isCollapsed) {
        const sidebar = document.querySelector('.sidebar');
        const mainContent = document.querySelector('.main-content');
        const queryBar = document.querySelector('.query-bar-container');
        
        sidebar.classList.add('collapsed');
        mainContent.classList.add('expanded');
        queryBar.classList.add('expanded');
    }
}

// ============================================================
// PAGE NAVIGATION
// ============================================================

function setPage(pageName) {
    if (window.parent) {
        window.parent.postMessage({ type: 'setPage', page: pageName }, '*');
    }
}

// ============================================================
// UPLOAD ZONE
// ============================================================

function initUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    if (!uploadZone) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        }, false);
    });
    
    uploadZone.addEventListener('drop', handleDrop, false);
    
    uploadZone.addEventListener('click', () => {
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) fileInput.click();
    });
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        const dataTransfer = new DataTransfer();
        for (let file of files) {
            dataTransfer.items.add(file);
        }
        fileInput.files = dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
    }
}

// ============================================================
// QUERY BAR
// ============================================================

function setQuery(text) {
    const queryInput = document.getElementById('queryInput');
    if (queryInput) {
        queryInput.value = text;
        queryInput.focus();
    }
}

function clearQuery() {
    const queryInput = document.getElementById('queryInput');
    if (queryInput) {
        queryInput.value = '';
        queryInput.focus();
    }
}

function submitQuery() {
    const queryInput = document.getElementById('queryInput');
    if (queryInput && queryInput.value.trim()) {
        // Trigger Streamlit button
        const sendBtn = document.querySelector('[data-testid="baseButton-primary"]');
        if (sendBtn) sendBtn.click();
    }
}

// ============================================================
// VOICE CONTROLS
// ============================================================

let isRecording = false;
let recognition = null;

function initVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.log('Speech recognition not supported');
        return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    
    recognition.onstart = () => {
        isRecording = true;
        updateRecordingUI(true);
    };
    
    recognition.onend = () => {
        isRecording = false;
        updateRecordingUI(false);
    };
    
    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');
        
        const queryInput = document.getElementById('queryInput');
        if (queryInput) {
            queryInput.value = transcript;
        }
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        isRecording = false;
        updateRecordingUI(false);
    };
}

function toggleRecording() {
    if (!recognition) {
        initVoiceRecognition();
        if (!recognition) return;
    }
    
    if (isRecording) {
        recognition.stop();
    } else {
        const language = document.getElementById('languageSelector')?.value || 'auto';
        if (language !== 'auto') {
            recognition.lang = language === 'te' ? 'te-IN' : 'en-US';
        }
        recognition.start();
    }
}

function updateRecordingUI(recording) {
    const indicator = document.getElementById('recordingIndicator');
    const micBtn = document.getElementById('micBtn');
    
    if (indicator) {
        indicator.classList.toggle('hidden', !recording);
    }
    
    if (micBtn) {
        micBtn.classList.toggle('active', recording);
    }
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================

function closeToast(id) {
    const toast = document.getElementById(`toast-${id}`);
    if (toast) {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }
}

// ============================================================
// PDF VIEWER
// ============================================================

function openPdfViewer() {
    const pdfViewer = document.querySelector('.pdf-viewer');
    if (pdfViewer) {
        pdfViewer.classList.add('open');
    }
}

function closePdfViewer() {
    const pdfViewer = document.querySelector('.pdf-viewer');
    if (pdfViewer) {
        pdfViewer.classList.remove('open');
    }
}

function jumpToPage(pageNumber) {
    const pdfPage = document.querySelector(`[data-page="${pageNumber}"]`);
    if (pdfPage) {
        pdfPage.scrollIntoView({ behavior: 'smooth', block: 'center' });
        pdfPage.classList.add('highlighted');
        setTimeout(() => pdfPage.classList.remove('highlighted'), 2000);
    }
}

// ============================================================
// ANIMATIONS
// ============================================================

function animateOnScroll() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    elements.forEach(el => observer.observe(el));
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const queryInput = document.getElementById('queryInput');
            if (queryInput) queryInput.focus();
        }
        
        // Escape to clear query
        if (e.key === 'Escape') {
            clearQuery();
        }
        
        // Ctrl/Cmd + / to toggle sidebar
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            toggleSidebar();
        }
    });
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSidebar();
    initUploadZone();
    initVoiceRecognition();
    animateOnScroll();
    initKeyboardShortcuts();
    
    // Listen for messages from Streamlit
    window.addEventListener('message', (event) => {
        if (event.data.type === 'setPage') {
            setPage(event.data.page);
        } else if (event.data.type === 'toggleSidebar') {
            toggleSidebar();
        } else if (event.data.type === 'toggleTheme') {
            toggleTheme();
        }
    });
});

// ============================================================
// EXPORT FUNCTIONS
// ============================================================

window.ScholarAI = {
    toggleTheme,
    toggleSidebar,
    setPage,
    setQuery,
    clearQuery,
    submitQuery,
    toggleRecording,
    closeToast,
    openPdfViewer,
    closePdfViewer,
    jumpToPage
};
