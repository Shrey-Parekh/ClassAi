const API_URL = 'http://localhost:8001';

const messagesContainer = document.getElementById('messagesContainer');
const messages = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const thinkingIndicator = document.getElementById('thinkingIndicator');
const welcomeScreen = document.getElementById('welcomeScreen');

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

let thinkingInterval = null;
const thinkingPhrases = [
    'Understanding your question...',
    'Searching faculty documents...',
    'Finding relevant information...',
    'Cross-referencing sources...',
    'Generating comprehensive answer...'
];

function startThinking() {
    if (!thinkingIndicator) return;
    let index = 0;
    thinkingIndicator.innerHTML = `<span class="thinking-pulse"></span>${thinkingPhrases[index]}`;
    thinkingIndicator.style.display = 'flex';
    clearInterval(thinkingInterval);
    thinkingInterval = setInterval(() => {
        index = (index + 1) % thinkingPhrases.length;
        thinkingIndicator.innerHTML = `<span class="thinking-pulse"></span>${thinkingPhrases[index]}`;
    }, 1500);
}

function stopThinking() {
    if (!thinkingIndicator) return;
    clearInterval(thinkingInterval);
    thinkingIndicator.style.display = 'none';
    thinkingIndicator.textContent = '';
}

// Check API connection
async function checkConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Connected';
            return true;
        }
    } catch (error) {
        statusIndicator.classList.remove('connected');
        statusText.textContent = 'Disconnected';
        return false;
    }
}

// Add message to chat with ChatGPT-style typing animation
function addMessage(content, isUser, sources = null, animate = false) {
    // Hide welcome screen on first message
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const header = document.createElement('div');
    header.className = 'message-header';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? 'Y' : 'AI';
    
    const name = document.createElement('div');
    name.className = 'message-name';
    name.textContent = isUser ? 'You' : 'Faculty Assistant';
    
    header.appendChild(avatar);
    header.appendChild(name);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);
    
    // Animate typing for assistant messages
    if (!isUser && animate) {
        // Show thinking status in message area
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'message-thinking';
        thinkingDiv.innerHTML = '<span class="thinking-pulse"></span><span class="thinking-text">Thinking...</span>';
        contentDiv.appendChild(thinkingDiv);
        
        typeMessage(contentDiv, content, sources, thinkingDiv);
    } else {
        const formattedContent = formatContent(content);
        contentDiv.innerHTML = formattedContent;
        
        // Add sources if available
        if (sources && sources.length > 0) {
            addSources(contentDiv, sources);
        }
    }
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

// ChatGPT-style typing animation
async function typeMessage(contentDiv, text, sources = null, thinkingDiv = null) {
    // Remove thinking indicator
    if (thinkingDiv) {
        thinkingDiv.remove();
    }
    
    const formattedText = formatContent(text);
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = formattedText;
    
    // Extract text content while preserving structure
    const textContent = tempDiv.textContent || tempDiv.innerText;
    
    let displayedText = '';
    const typingSpeed = 15; // milliseconds per character
    
    for (let i = 0; i < textContent.length; i++) {
        displayedText += textContent[i];
        
        // Format and display
        const formatted = formatContent(displayedText);
        contentDiv.innerHTML = formatted;
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Variable speed for natural feel
        const delay = textContent[i] === '.' || textContent[i] === '!' || textContent[i] === '?' 
            ? typingSpeed * 8 
            : textContent[i] === ',' 
            ? typingSpeed * 4 
            : typingSpeed;
        
        await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    // Add sources after typing completes
    if (sources && sources.length > 0) {
        await new Promise(resolve => setTimeout(resolve, 300));
        addSources(contentDiv, sources);
    }
}

// Add sources section
function addSources(contentDiv, sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'message-sources';
    sourcesDiv.innerHTML = '<strong>Sources:</strong>';
    
    const sourcesList = document.createElement('ul');
    sources.forEach(source => {
        const li = document.createElement('li');
        li.textContent = source.title || source.doc_id;
        if (source.date) {
            li.textContent += ` (${source.date})`;
        }
        sourcesList.appendChild(li);
    });
    
    sourcesDiv.appendChild(sourcesList);
    contentDiv.appendChild(sourcesDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Format content with proper structure and styling
function formatContent(text) {
    if (!text) return '';
    
    // First, escape any HTML to prevent injection
    const escapeHtml = (str) => {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    };
    
    // Don't escape if already contains HTML tags (for recursive calls)
    if (!text.includes('<')) {
        text = escapeHtml(text);
    }
    
    // Preserve line breaks (convert \n to <br>)
    text = text.replace(/\n/g, '<br>');
    
    // Format section headers (ALL CAPS or Title Case followed by colon)
    // Must be at start of line or after <br>
    text = text.replace(/(^|<br>)([A-Z][A-Z\s]{2,50}):(?=<br>|$)/g, '$1<div class="section-header">$2:</div>');
    text = text.replace(/(^|<br>)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):(?=<br>|$)/g, '$1<div class="section-header">$2:</div>');
    
    // Format step headers (Step 1:, Step 2:, etc.)
    text = text.replace(/(Step\s+\d+):/gi, '<span class="step-header">$1:</span>');
    
    // Format numbered lists (1. 2. 3. etc.) - must be at start of line or after <br>
    text = text.replace(/(^|<br>)(\d+)\.\s+([^<]+?)(?=<br>|$)/g, '$1<div class="numbered-item"><span class="number">$2.</span><span class="content">$3</span></div>');
    
    // Format bullet points (-, •, *)
    text = text.replace(/(^|<br>)[-•*]\s+([^<]+?)(?=<br>|$)/g, '$1<li>$2</li>');
    
    // Wrap consecutive <li> items in <ul>
    text = text.replace(/(<li>.*?<\/li>)(?:\s*<br>\s*<li>.*?<\/li>)*/g, (match) => {
        // Remove <br> tags between list items
        const cleaned = match.replace(/<br>/g, '');
        return '<ul>' + cleaned + '</ul>';
    });
    
    // Bold text (**text**)
    text = text.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
    
    // Italic text (*text*)
    text = text.replace(/\*([^*]+?)\*/g, '<em>$1</em>');
    
    // Format source citations at the end
    text = text.replace(/\(Source:\s*([^)]+)\)/g, '<div class="source-citation">(Source: $1)</div>');
    
    // Clean up multiple consecutive <br> tags (max 2)
    text = text.replace(/(<br>\s*){3,}/g, '<br><br>');
    
    // Remove <br> before and after block elements
    text = text.replace(/<br>\s*(<div|<ul)/g, '$1');
    text = text.replace(/(<\/div>|<\/ul>)\s*<br>/g, '$1');
    
    // Add spacing after block elements
    text = text.replace(/<\/div>/g, '</div><br>');
    text = text.replace(/<\/ul>/g, '</ul><br>');
    
    // Final cleanup
    text = text.replace(/(<br>\s*){3,}/g, '<br><br>');
    text = text.replace(/^<br>|<br>$/g, '');
    
    return text;
}

// Add loading indicator
function addLoadingMessage() {
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant loading';
    
    const header = document.createElement('div');
    header.className = 'message-header';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';
    
    const name = document.createElement('div');
    name.className = 'message-name';
    name.textContent = 'Faculty Assistant';
    
    header.appendChild(avatar);
    header.appendChild(name);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    
    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

// Send query to API
async function sendQuery(query) {
    const loadingMessage = addLoadingMessage();
    sendBtn.disabled = true;
    userInput.disabled = true;
    startThinking();
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        
        // Remove loading message
        loadingMessage.remove();
        stopThinking();
        
        // Add assistant response with typing animation
        addMessage(data.answer, false, data.sources, true);
        
    } catch (error) {
        loadingMessage.remove();
        stopThinking();
        addMessage('I apologize, but I encountered an error processing your request. Please make sure the API is running and try again.', false, null, false);
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

// Handle send button click
sendBtn.addEventListener('click', async () => {
    const query = userInput.value.trim();
    if (!query) return;
    
    // Add user message
    addMessage(query, true);
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Send to API
    await sendQuery(query);
});

// Handle Enter key
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
    }
});

// Send suggestion
function sendSuggestion(text) {
    userInput.value = text;
    sendBtn.click();
}

// Start new chat
function startNewChat() {
    messages.innerHTML = '';
    welcomeScreen.style.display = 'block';
    userInput.value = '';
    userInput.focus();
}

// Initialize
(async () => {
    const connected = await checkConnection();
    if (!connected) {
        statusText.textContent = 'API not available';
    }
    userInput.focus();
})();

// Check connection every 30 seconds
setInterval(checkConnection, 30000);
