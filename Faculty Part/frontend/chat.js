const API_URL = 'http://localhost:8001';

const messages       = document.getElementById('messages');
const userInput      = document.getElementById('userInput');
const sendBtn        = document.getElementById('sendBtn');
const stopBtn        = document.getElementById('stopBtn');
const roleLabel      = document.getElementById('roleLabel');
const welcomeScreen  = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const mobileMenuToggle  = document.getElementById('mobileMenuToggle');
const sidebar           = document.getElementById('sidebar');
const mobileBackdrop    = document.getElementById('mobileBackdrop');
const modelIdEl         = document.getElementById('modelId');
const scopeSelector     = document.getElementById('scopeSelector');

// ── Scope management (faculty/admin only) ─────────────────────────────────────
let currentScope = 'both'; // Default scope

// Initialize scope selector based on role
const role = sessionStorage.getItem('classai_role') || '';
if (scopeSelector && (role === 'faculty' || role === 'admin')) {
    // Show scope selector for faculty/admin
    scopeSelector.style.display = 'block';
    
    // Handle scope pill clicks
    const scopePills = scopeSelector.querySelectorAll('.scope-pill');
    scopePills.forEach(pill => {
        pill.addEventListener('click', () => {
            // Remove active class from all pills
            scopePills.forEach(p => p.classList.remove('scope-pill--active'));
            // Add active class to clicked pill
            pill.classList.add('scope-pill--active');
            // Update current scope
            currentScope = pill.getAttribute('data-scope');
            console.log('Scope changed to:', currentScope);
        });
    });
} else {
    // Hide scope selector for students
    if (scopeSelector) {
        scopeSelector.style.display = 'none';
    }
    currentScope = 'student'; // Force student scope
}

// ── Mobile sidebar ────────────────────────────────────────────────────────────
if (mobileMenuToggle && sidebar && mobileBackdrop) {
    function openSidebar() {
        sidebar.classList.add('open');
        sidebar.removeAttribute('aria-hidden');
        mobileBackdrop.classList.add('active');
        mobileMenuToggle.setAttribute('aria-expanded', 'true');
        // A37: move focus into sidebar
        const firstBtn = sidebar.querySelector('button, a');
        if (firstBtn) firstBtn.focus();
    }
    function closeSidebar() {
        sidebar.classList.remove('open');
        sidebar.setAttribute('aria-hidden', 'true');
        mobileBackdrop.classList.remove('active');
        mobileMenuToggle.setAttribute('aria-expanded', 'false');
    }
    mobileMenuToggle.addEventListener('click', () => {
        sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
    });
    mobileBackdrop.addEventListener('click', closeSidebar);
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) closeSidebar();
    });
}

// ── Session ───────────────────────────────────────────────────────────────────
let currentSessionId = null;
let sessionReady = false;

// A8: keep sendBtn disabled until session is ready
async function initSession() {
    sessionReady = false;
    sendBtn.disabled = true;
    try {
        const res = await fetch(`${API_URL}/conversation/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json();
        currentSessionId = data.session_id;
        sessionReady = true;
        // Re-enable only if there's text
        sendBtn.disabled = !userInput.value.trim();
    } catch (err) {
        console.error('Failed to initialize session:', err);
        sessionReady = true; // allow sending even without session
        sendBtn.disabled = !userInput.value.trim();
    }
}

// ── Role display ──────────────────────────────────────────────────────────────
if (roleLabel && role) {
    roleLabel.textContent = role.charAt(0).toUpperCase() + role.slice(1);
}

// ── Token validation ──────────────────────────────────────────────────────────
// Validate token on page load - if invalid, redirect to login
async function validateToken() {
    const token = sessionStorage.getItem('classai_token');
    if (!token) {
        window.location.href = 'signin.html';
        return false;
    }
    
    try {
        // Make a lightweight request to check if token is valid
        const res = await fetch(`${API_URL}/conversation/new`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (res.status === 401) {
            alert('Your session has expired. Please login again.');
            window.location.href = 'signin.html';
            return false;
        }
        
        return true;
    } catch (err) {
        console.error('Token validation failed:', err);
        return true; // Allow to continue on network errors
    }
}

// Validate token before initializing
validateToken().then(valid => {
    if (valid) {
        initSession();
    }
});

// ── Textarea auto-resize (A20: capped at 200px) ───────────────────────────────
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    sendBtn.disabled = !this.value.trim() || !sessionReady;
});

// A30: clear error on type
document.querySelectorAll('.signin-input').forEach(el => {
    el.addEventListener('input', () => {
        const err = document.getElementById('signinError');
        if (err) err.textContent = '';
    });
});

// ── Scroll helper (A25: throttled) ───────────────────────────────────────────
let _scrollRaf = null;
function scrollToBottom() {
    if (_scrollRaf) return;
    _scrollRaf = requestAnimationFrame(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        _scrollRaf = null;
    });
}

// ── Add message ───────────────────────────────────────────────────────────────
function addMessage(content, isUser) {
    if (welcomeScreen) welcomeScreen.style.display = 'none';

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

    const wrapper = document.createElement('div');
    wrapper.className = 'message-wrapper';

    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = isUser ? 'You' : 'ClassAI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    wrapper.appendChild(label);
    wrapper.appendChild(contentDiv);

    if (!isUser) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.setAttribute('aria-label', 'Copy message');
        // A36: aria-live for screen reader feedback
        const copyStatus = document.createElement('span');
        copyStatus.setAttribute('aria-live', 'polite');
        copyStatus.setAttribute('aria-atomic', 'true');
        copyStatus.className = 'sr-only';
        copyBtn.innerHTML = `<i data-lucide="copy" aria-hidden="true"></i><span>Copy</span>`;
        copyBtn.appendChild(copyStatus);
        copyBtn.onclick = () => copyMessage(contentDiv, copyBtn, copyStatus);
        actionsDiv.appendChild(copyBtn);
        wrapper.appendChild(actionsDiv);
    }

    messageDiv.appendChild(wrapper);
    messages.appendChild(messageDiv);

    if (isUser) contentDiv.textContent = content;

    scrollToBottom();
    lucide.createIcons({ nodes: [wrapper] }); // A32: scope to new node only

    return { messageDiv, contentDiv, wrapper };
}

// ── Thinking block ────────────────────────────────────────────────────────────
function showThinking() {
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    const block = document.createElement('div');
    block.className = 'message assistant thinking-message';
    block.id = 'thinkingBlock';
    block.innerHTML = `
        <div class="message-wrapper">
            <span class="message-label">CLASSAI</span>
            <div class="message-content">
                <div class="thinking-scan-container">
                    <div class="thinking-scan-line"></div>
                    <div class="thinking-doc-line">
                        <span class="thinking-frag" style="width:52px"></span>
                        <span class="thinking-frag" style="width:88px"></span>
                        <span class="thinking-frag highlight h1" style="width:64px"></span>
                        <span class="thinking-frag" style="width:40px"></span>
                        <span class="thinking-frag" style="width:72px"></span>
                        <span class="thinking-frag highlight h2" style="width:56px"></span>
                    </div>
                    <div class="thinking-doc-line">
                        <span class="thinking-frag" style="width:44px"></span>
                        <span class="thinking-frag highlight h3" style="width:80px"></span>
                        <span class="thinking-frag" style="width:60px"></span>
                        <span class="thinking-frag" style="width:48px"></span>
                    </div>
                </div>
            </div>
        </div>`;
    messages.appendChild(block);
    scrollToBottom();
    return block;
}

function removeThinking() {
    const b = document.getElementById('thinkingBlock');
    if (b) b.remove();
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function showTypingIndicator(wrapper) {
    const d = document.createElement('div');
    d.className = 'typing-indicator';
    d.id = 'typingIndicator';
    d.setAttribute('aria-label', 'Generating response');
    d.innerHTML = `<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>`;
    wrapper.appendChild(d);
    scrollToBottom();
}
function removeTypingIndicator() {
    const d = document.getElementById('typingIndicator');
    if (d) d.remove();
}

// ── Abort controller (A10) ────────────────────────────────────────────────────
let currentAbortController = null;
let isSending = false; // A9: duplicate-submit guard

stopBtn.addEventListener('click', () => {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    stopBtn.style.display = 'none';
    sendBtn.style.display = '';
    sendBtn.disabled = false;
    userInput.disabled = false;
    removeThinking();
    isSending = false;
});

// ── Send query ────────────────────────────────────────────────────────────────
async function sendQuery(query) {
    if (isSending) return; // A9
    isSending = true;

    sendBtn.style.display = 'none';
    stopBtn.style.display = '';
    userInput.disabled = true;

    currentAbortController = new AbortController();
    const thinkingBlock = showThinking();

    // Get token for Authorization header
    const token = sessionStorage.getItem('classai_token');

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': token ? `Bearer ${token}` : ''
            },
            body: JSON.stringify({ 
                query, 
                session_id: currentSessionId, 
                stream: true,
                scope: currentScope  // Include scope in request
            }),
            signal: currentAbortController.signal
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalResult = null;

        while (true) {
            const { done, value } = await reader.read();
            // A7: flush remaining buffer on stream close
            if (done) {
                if (buffer.trim()) {
                    processLine(buffer.trim());
                }
                break;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) processLine(line);
        }

        function processLine(line) {
            if (!line.startsWith('data:')) return;
            // A6: safe JSON parse
            let data;
            try { data = JSON.parse(line.substring(5).trim()); }
            catch { return; }
            if (data.answer !== undefined) finalResult = data;
        }

        if (!finalResult) throw new Error('No result received');

        removeThinking();

        let structured = null;
        try { structured = JSON.parse(finalResult.answer); }
        catch { structured = null; }

        const { contentDiv, wrapper } = addMessage('', false);

        // Show low confidence warning if needed
        if (structured && structured.confidence === "low") {
            const warningStrip = document.createElement('div');
            warningStrip.className = 'confidence-warning';
            warningStrip.textContent = 'Low confidence — verify with source';
            contentDiv.appendChild(warningStrip);
        }

        if (structured) {
            // Pass sources array to renderer for collection tags
            const sources = finalResult.sources || [];
            const rendered = renderer.render(structured, sources);
            contentDiv.appendChild(rendered);
        } else {
            contentDiv.textContent = finalResult.answer || 'No response received.';
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            removeThinking();
        } else {
            removeThinking();
            
            // Check for authentication errors (401 or invalid token)
            const msg = (err.message || '').toLowerCase();
            if (msg.includes('401') || msg.includes('unauthorized')) {
                // Token is invalid (likely server restart) - redirect to login
                alert('Your session has expired. Please login again.');
                window.location.href = 'signin.html';
                return;
            }
            
            const { contentDiv } = addMessage('', false);
            // A21: structured error display
            const errorDiv = document.createElement('div');
            errorDiv.className = 'response-section';
            const h = document.createElement('h3');
            h.className = 'section-heading';
            h.style.color = '#ef4444';
            const p = document.createElement('p');
            p.className = 'section-paragraph';
            if (msg.includes('429') || msg.includes('rate limit')) {
                h.textContent = 'Rate Limit Reached';
                p.textContent = 'Too many requests. Please wait a moment and try again.';
            } else if (msg.includes('timeout') || msg.includes('timed out')) {
                h.textContent = 'Request Timeout';
                p.textContent = 'The request took too long. Please try a simpler query.';
            } else if (msg.includes('network') || msg.includes('fetch') || msg.includes('failed to fetch')) {
                h.textContent = 'Connection Error';
                p.textContent = 'Unable to reach the server. Check your connection.';
            } else if (msg.includes('503') || msg.includes('not initialized')) {
                h.textContent = 'Service Unavailable';
                p.textContent = 'The system is starting up. Please try again in a moment.';
            } else {
                h.textContent = 'Something Went Wrong';
                p.textContent = 'An unexpected error occurred. Please try again.';
            }
            errorDiv.appendChild(h);
            errorDiv.appendChild(p);
            // A22: retry button
            const retryBtn = document.createElement('button');
            retryBtn.className = 'retry-btn';
            retryBtn.textContent = 'Retry';
            retryBtn.onclick = () => {
                errorDiv.closest('.message')?.remove();
                sendQuery(query);
            };
            errorDiv.appendChild(retryBtn);
            contentDiv.appendChild(errorDiv);
            console.error('Query error:', err);
        }
    } finally {
        isSending = false;
        currentAbortController = null;
        stopBtn.style.display = 'none';
        sendBtn.style.display = '';
        sendBtn.disabled = !userInput.value.trim();
        userInput.disabled = false;
        userInput.focus();
    }
}

// ── Send handlers ─────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', async () => {
    const query = userInput.value.trim();
    if (!query || isSending) return;
    addMessage(query, true);
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    await sendQuery(query);
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendBtn.click();
    }
});

// ── New chat (A23: notify backend) ────────────────────────────────────────────
async function startNewChat() {
    if (currentSessionId) {
        try {
            await fetch(`${API_URL}/conversation/${currentSessionId}`, { method: 'DELETE' });
        } catch { /* best-effort */ }
    }
    messages.innerHTML = '';
    welcomeScreen.style.display = 'flex';
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    userInput.focus();
    await initSession();
}

// ── Sign out (A24: call backend) ──────────────────────────────────────────────
async function signOut() {
    try {
        const token = sessionStorage.getItem('classai_token');
        if (token) {
            await fetch(`${API_URL}/api/auth/signout`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
        }
    } catch { /* best-effort */ }
    sessionStorage.removeItem('classai_token');
    sessionStorage.removeItem('classai_role');
    sessionStorage.removeItem('classai_user');
    window.location.href = '/signin';
}

// ── Copy message (formatted text representation) ─────────────────────────────
async function copyMessage(contentDiv, button, statusEl) {
    // Extract formatted text representation
    let text = '';
    const sections = contentDiv.querySelectorAll('.response-section, .section-paragraph, .section-bullets, .section-steps, .section-table, .section-alert');
    
    if (sections.length > 0) {
        sections.forEach(section => {
            const heading = section.querySelector('h3');
            if (heading) text += heading.textContent + '\n';
            
            const bullets = section.querySelectorAll('ul > li');
            if (bullets.length > 0) {
                bullets.forEach(li => text += '• ' + li.textContent + '\n');
                text += '\n';
                return;
            }
            
            const steps = section.querySelectorAll('ol > li');
            if (steps.length > 0) {
                steps.forEach((li, i) => text += `${i + 1}. ${li.textContent}\n`);
                text += '\n';
                return;
            }
            
            const table = section.querySelector('table');
            if (table) {
                const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent);
                text += headers.join('\t') + '\n';
                table.querySelectorAll('tbody tr').forEach(tr => {
                    const cells = Array.from(tr.querySelectorAll('td')).map(td => td.textContent);
                    text += cells.join('\t') + '\n';
                });
                text += '\n';
                return;
            }
            
            const para = section.querySelector('p');
            if (para) text += para.textContent + '\n\n';
        });
    } else {
        text = contentDiv.innerText || contentDiv.textContent;
    }
    
    try {
        await navigator.clipboard.writeText(text.trim());
    } catch {
        const ta = document.createElement('textarea');
        ta.value = text.trim();
        ta.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
    const span = button.querySelector('span:not(.sr-only)');
    const original = span.textContent;
    span.textContent = 'Copied!';
    button.setAttribute('aria-label', 'Copied!');
    // A36: announce to screen readers
    if (statusEl) statusEl.textContent = 'Message copied to clipboard';
    setTimeout(() => {
        span.textContent = original;
        button.setAttribute('aria-label', 'Copy message');
        if (statusEl) statusEl.textContent = '';
    }, 2000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
// Session initialization is now handled by validateToken() above

// Welcome screen reveal animation
requestAnimationFrame(() => {
    document.querySelectorAll('.welcome .hero-reveal').forEach((el, i) => {
        el.style.transitionDelay = (i * 80) + 'ms';
        el.classList.add('reveal--in');
    });
});
