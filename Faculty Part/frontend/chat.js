const API_URL = 'http://localhost:8000';

const messages = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const roleLabel = document.getElementById('roleLabel');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const sidebar = document.querySelector('.sidebar');
const mobileBackdrop = document.getElementById('mobileBackdrop');

// Mobile menu toggle
if (mobileMenuToggle && sidebar && mobileBackdrop) {
    mobileMenuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        mobileBackdrop.classList.toggle('active');
    });
    
    // Close sidebar when clicking backdrop
    mobileBackdrop.addEventListener('click', () => {
        sidebar.classList.remove('open');
        mobileBackdrop.classList.remove('active');
    });
    
    // Close sidebar when window is resized above mobile breakpoint
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            sidebar.classList.remove('open');
            mobileBackdrop.classList.remove('active');
        }
    });
}

// Session management
let currentSessionId = null;

// Initialize session on load
async function initSession() {
    try {
        const response = await fetch(`${API_URL}/conversation/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        currentSessionId = data.session_id;
        console.log('Session initialized:', currentSessionId);
    } catch (error) {
        console.error('Failed to initialize session:', error);
    }
}

// Display role
const role = sessionStorage.getItem('classai_role') || '';
if (roleLabel && role) {
    roleLabel.textContent = role.charAt(0).toUpperCase() + role.slice(1);
}

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    
    // Enable/disable send button
    sendBtn.disabled = !this.value.trim();
});

// Typewriter function
function typewriterAppend(element, text, speed = 12) {
    return new Promise((resolve) => {
        let i = 0;
        const interval = setInterval(() => {
            if (i < text.length) {
                element.textContent += text[i];
                i++;
                // Auto-scroll
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } else {
                clearInterval(interval);
                resolve();
            }
        }, speed);
    });
}

// Add message
function addMessage(content, isUser, sources = null) {
    // Hide welcome on first message
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    
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
    
    // Add copy button for assistant messages
    if (!isUser) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.innerHTML = `
            <svg data-lucide="copy"></svg>
            <span>Copy</span>
        `;
        copyBtn.onclick = () => copyMessage(contentDiv, copyBtn);
        
        actionsDiv.appendChild(copyBtn);
        wrapper.appendChild(actionsDiv);
    }
    
    messageDiv.appendChild(wrapper);
    messages.appendChild(messageDiv);
    
    if (isUser) {
        contentDiv.textContent = content;
    }
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Re-render lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    return { messageDiv, contentDiv, wrapper };
}

// Show thinking block
function showThinking() {
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    
    const thinkingBlock = document.createElement('div');
    thinkingBlock.className = 'message assistant thinking-message';
    thinkingBlock.id = 'thinkingBlock';
    thinkingBlock.innerHTML = `
        <div class="message-wrapper">
            <span class="message-label">CLASSAI</span>
            <div class="message-content">
                <div class="thinking-scan-container">
                    <div class="thinking-scan-line"></div>
                    <div class="thinking-doc-line">
                        <span class="thinking-frag" style="width: 52px;"></span>
                        <span class="thinking-frag" style="width: 88px;"></span>
                        <span class="thinking-frag highlight h1" style="width: 64px;"></span>
                        <span class="thinking-frag" style="width: 40px;"></span>
                        <span class="thinking-frag" style="width: 72px;"></span>
                        <span class="thinking-frag highlight h2" style="width: 56px;"></span>
                        <span class="thinking-frag" style="width: 36px;"></span>
                    </div>
                    <div class="thinking-doc-line">
                        <span class="thinking-frag" style="width: 44px;"></span>
                        <span class="thinking-frag highlight h3" style="width: 80px;"></span>
                        <span class="thinking-frag" style="width: 60px;"></span>
                        <span class="thinking-frag" style="width: 48px;"></span>
                        <span class="thinking-frag" style="width: 32px;"></span>
                        <span class="thinking-frag" style="width: 68px;"></span>
                    </div>
                    <div class="thinking-doc-line">
                        <span class="thinking-frag" style="width: 64px;"></span>
                        <span class="thinking-frag" style="width: 40px;"></span>
                        <span class="thinking-frag" style="width: 56px;"></span>
                        <span class="thinking-frag highlight h4" style="width: 72px;"></span>
                        <span class="thinking-frag" style="width: 48px;"></span>
                    </div>
                </div>
                <div class="thinking-citations">
                    <span class="thinking-cite c1">[1]</span>
                    <span class="thinking-cite c2">[2]</span>
                    <span class="thinking-cite c3">[3]</span>
                </div>
            </div>
        </div>
    `;
    
    messages.appendChild(thinkingBlock);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return thinkingBlock;
}

// Update thinking status
function updateThinkingStep(stepName, detail, isComplete = false) {
    // No-op: document scan animation doesn't use steps
}

// Show typing indicator
function showTypingIndicator(wrapper) {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    wrapper.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Remove thinking block
function removeThinking() {
    const thinkingBlock = document.getElementById('thinkingBlock');
    if (thinkingBlock) {
        thinkingBlock.remove();
    }
}

// Send query with streaming support
async function sendQuery(query) {
    sendBtn.disabled = true;
    userInput.disabled = true;
    
    const thinkingBlock = showThinking();
    
    try {
        // Use streaming for better UX
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query,
                session_id: currentSessionId,
                stream: true
            })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalResult = null;
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('event:')) {
                    const eventType = line.substring(6).trim();
                    continue;
                }
                
                if (line.startsWith('data:')) {
                    const data = JSON.parse(line.substring(5).trim());
                    
                    if (data.step === 'understanding') {
                        // Already showing understanding step
                    } else if (data.step === 'retrieval') {
                        updateThinkingStep('Retrieving documents', data.message);
                    } else if (data.step === 'generation') {
                        updateThinkingStep('Generating answer', data.message);
                    } else if (data.step === 'cache_hit') {
                        updateThinkingStep('Cache hit', 'Using cached result');
                    } else if (data.answer) {
                        // Final result
                        finalResult = data;
                    }
                }
            }
        }
        
        if (!finalResult) {
            throw new Error('No result received');
        }
        
        // Mark final step as complete
        setTimeout(() => {
            const stepsContainer = document.getElementById('thinkingSteps');
            if (stepsContainer) {
                const activeSteps = stepsContainer.querySelectorAll('.thinking-step.active');
                activeSteps.forEach(step => {
                    step.classList.remove('active');
                    step.classList.add('completed');
                });
            }
        }, 400);
        
        // Wait a bit before removing thinking and showing response
        await new Promise(resolve => setTimeout(resolve, 600));
        
        // Remove thinking
        removeThinking();
        
        // Parse structured response
        let answerHTML = '';
        let sources = finalResult.sources || [];
        let structured = null;
        
        try {
            structured = JSON.parse(finalResult.answer);
            
            // Build HTML from structured response
            if (structured.fallback) {
                answerHTML = `<p>${structured.fallback}</p>`;
            } else {
                const parts = [];
                
                // Title
                if (structured.title) {
                    parts.push(`<h2 class="response-title">${structured.title}</h2>`);
                }
                
                // Subtitle
                if (structured.subtitle) {
                    parts.push(`<p class="response-subtitle">${structured.subtitle}</p>`);
                }
                
                // Sections
                if (structured.sections && structured.sections.length > 0) {
                    structured.sections.forEach(section => {
                        let sectionHTML = '';
                        
                        // Section heading
                        if (section.heading) {
                            sectionHTML += `<h3 class="section-heading">${section.heading}</h3>`;
                        }
                        
                        // Section content based on type
                        if (section.type === 'paragraph' && section.content) {
                            sectionHTML += `<p class="section-paragraph">${section.content}</p>`;
                        } else if (section.type === 'bullets' && section.items) {
                            sectionHTML += '<ul class="section-bullets">';
                            section.items.forEach(item => {
                                sectionHTML += `<li>${item}</li>`;
                            });
                            sectionHTML += '</ul>';
                        } else if (section.type === 'steps' && section.items) {
                            sectionHTML += '<ol class="section-steps">';
                            section.items.forEach(item => {
                                sectionHTML += `<li>${item}</li>`;
                            });
                            sectionHTML += '</ol>';
                        } else if (section.type === 'alert' && section.content) {
                            const severityClass = section.severity || 'info';
                            sectionHTML += `<div class="section-alert alert-${severityClass}">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <line x1="12" y1="8" x2="12" y2="12"></line>
                                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                                </svg>
                                <span>${section.content}</span>
                            </div>`;
                        }
                        
                        if (sectionHTML) {
                            parts.push(`<div class="response-section">${sectionHTML}</div>`);
                        }
                    });
                }
                
                // Footer
                if (structured.footer) {
                    parts.push(`<p class="response-footer">${structured.footer}</p>`);
                }
                
                answerHTML = parts.join('');
            }
        } catch (e) {
            // Use raw answer if not JSON
            answerHTML = `<p>${finalResult.answer}</p>`;
        }
        
        // Create assistant message with typing indicator
        const { contentDiv, wrapper } = addMessage('', false);
        showTypingIndicator(wrapper);
        
        // Wait a moment then remove typing and show formatted content
        await new Promise(resolve => setTimeout(resolve, 800));
        removeTypingIndicator();
        
        // Set HTML content with citation rendering
        contentDiv.innerHTML = this._renderWithCitations(answerHTML, sources);
        
        // Add processing status if available
        if (finalResult.metadata && finalResult.metadata.processing_steps) {
            const statusDiv = document.createElement('div');
            statusDiv.className = 'processing-status collapsed';
            statusDiv.id = `status-${Date.now()}`;
            
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'status-toggle';
            toggleBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
                <span>View processing steps</span>
            `;
            toggleBtn.onclick = () => {
                statusDiv.classList.toggle('collapsed');
                const isCollapsed = statusDiv.classList.contains('collapsed');
                toggleBtn.querySelector('span').textContent = isCollapsed ? 'View processing steps' : 'Hide processing steps';
                toggleBtn.querySelector('svg').style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)';
            };
            
            const stepsContainer = document.createElement('div');
            stepsContainer.className = 'status-steps';
            
            finalResult.metadata.processing_steps.forEach(step => {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'processing-step';
                stepDiv.innerHTML = `
                    <div class="step-icon completed">
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                            <path d="M2 5L4 7L8 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="step-text">
                        <div class="step-name">${step.step}</div>
                        <div class="step-details">${step.details}</div>
                    </div>
                `;
                stepsContainer.appendChild(stepDiv);
            });
            
            statusDiv.appendChild(toggleBtn);
            statusDiv.appendChild(stepsContainer);
            wrapper.appendChild(statusDiv);
        }
        
        // Add sources if present
        if (sources && sources.length > 0) {
            const citationBlock = document.createElement('div');
            citationBlock.className = 'citation-block';
            citationBlock.innerHTML = `<span class="citation-prefix">Source:</span> ${sources[0].title || sources[0].doc_id}`;
            wrapper.appendChild(citationBlock);
            
            // Fade in
            setTimeout(() => {
                citationBlock.classList.add('visible');
            }, 50);
        }
        
    } catch (error) {
        removeThinking();
        const { contentDiv } = addMessage('', false);
        
        // Determine error type and show specific message
        let errorMessage = 'Sorry, I encountered an error. Please try again.';
        let errorDetails = '';
        
        if (error.message) {
            const msg = error.message.toLowerCase();
            
            if (msg.includes('rate limit') || msg.includes('429')) {
                errorMessage = 'Rate Limit Reached';
                errorDetails = 'Too many requests. Please wait a moment and try again.';
            } else if (msg.includes('timeout') || msg.includes('timed out')) {
                errorMessage = 'Request Timeout';
                errorDetails = 'The request took too long. Please try a simpler query or try again.';
            } else if (msg.includes('network') || msg.includes('fetch')) {
                errorMessage = 'Connection Error';
                errorDetails = 'Unable to reach the server. Please check your connection and try again.';
            } else if (msg.includes('no relevant') || msg.includes('not found')) {
                errorMessage = 'No Information Found';
                errorDetails = 'I couldn\'t find relevant information for your query. Try rephrasing or being more specific.';
            } else if (msg.includes('503') || msg.includes('not initialized')) {
                errorMessage = 'Service Unavailable';
                errorDetails = 'The system is starting up or temporarily unavailable. Please try again in a moment.';
            } else {
                errorMessage = 'Something Went Wrong';
                errorDetails = 'An unexpected error occurred. Please try again or contact support if the issue persists.';
            }
        }
        
        contentDiv.innerHTML = `
            <div class="response-section">
                <h3 class="section-heading" style="color: #ef4444;">${errorMessage}</h3>
                <p class="section-paragraph">${errorDetails}</p>
            </div>
        `;
        
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

// Handle send
sendBtn.addEventListener('click', async () => {
    const query = userInput.value.trim();
    if (!query) return;
    
    addMessage(query, true);
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    await sendQuery(query);
});

// Handle Enter key
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) {
            sendBtn.click();
        }
    }
});

// Use prompt
function usePrompt(el) {
    userInput.value = el.textContent.trim();
    userInput.focus();
    userInput.dispatchEvent(new Event('input'));
}

// New chat
function startNewChat() {
    messages.innerHTML = '';
    welcomeScreen.style.display = 'flex';
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    userInput.focus();
    
    // Create new session
    initSession();
}

// Sign out
function signOut() {
    sessionStorage.removeItem('classai_token');
    sessionStorage.removeItem('classai_role');
    sessionStorage.removeItem('classai_user');
    window.location.href = '/signin';
}

// Initialize session on load
initSession();

// Focus input on load
userInput.focus();

// Render citations as superscripts with tooltips
function _renderWithCitations(html, sources) {
    // Strip citation numbers [N] from the text instead of rendering them
    return html.replace(/\[(\d+)\]/g, '');
}

// Attach to window for use in sendQuery
window._renderWithCitations = _renderWithCitations;


// Global abort controller for stopping generation
let currentAbortController = null;

// Copy message to clipboard
async function copyMessage(contentDiv, button) {
    try {
        // Get text content, stripping HTML
        const text = contentDiv.innerText || contentDiv.textContent;
        
        await navigator.clipboard.writeText(text);
        
        // Update button state
        const originalHTML = button.innerHTML;
        button.classList.add('copied');
        button.innerHTML = `
            <svg data-lucide="check"></svg>
            <span>Copied!</span>
        `;
        
        // Re-render icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        
        // Reset after 2 seconds
        setTimeout(() => {
            button.classList.remove('copied');
            button.innerHTML = originalHTML;
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        }, 2000);
        
    } catch (error) {
        console.error('Failed to copy:', error);
        
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = contentDiv.innerText || contentDiv.textContent;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        
        try {
            document.execCommand('copy');
            button.classList.add('copied');
            button.querySelector('span').textContent = 'Copied!';
            
            setTimeout(() => {
                button.classList.remove('copied');
                button.querySelector('span').textContent = 'Copy';
            }, 2000);
        } catch (err) {
            console.error('Fallback copy failed:', err);
        }
        
        document.body.removeChild(textArea);
    }
}
