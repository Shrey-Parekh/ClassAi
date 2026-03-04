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
    'Retrieving relevant faculty documents…',
    'Ranking sections by relevance…',
    'Drafting an answer, constrained to cited sources…'
];

function startThinking() {
    if (!thinkingIndicator) return;
    let index = 0;
    thinkingIndicator.textContent = thinkingPhrases[index];
    clearInterval(thinkingInterval);
    thinkingInterval = setInterval(() => {
        index = (index + 1) % thinkingPhrases.length;
        thinkingIndicator.textContent = thinkingPhrases[index];
    }, 2200);
}

function stopThinking() {
    if (!thinkingIndicator) return;
    clearInterval(thinkingInterval);
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

// Add message to chat
function addMessage(content, isUser, sources = null) {
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
    
    // Format content with markdown-like styling
    const formattedContent = formatContent(content);
    contentDiv.innerHTML = formattedContent;
    
    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = '<strong>Sources</strong>';
        
        const sourcesList = document.createElement('ul');
        sources.forEach(source => {
            const li = document.createElement('li');
            li.textContent = source.title || source.doc_id;
            sourcesList.appendChild(li);
        });
        
        sourcesDiv.appendChild(sourcesList);
        contentDiv.appendChild(sourcesDiv);
    }
    
    messages.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

// Format content with basic markdown
function formatContent(text) {
    // Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    // Lists
    text = text.replace(/^- (.*?)$/gm, '<li>$1</li>');
    if (text.includes('<li>')) {
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    }
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
    contentDiv.innerHTML = '<div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>';
    
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
        
        // Add assistant response
        addMessage(data.answer, false, data.sources);
        
    } catch (error) {
        loadingMessage.remove();
        stopThinking();
        addMessage('I apologize, but I encountered an error processing your request. Please make sure the API is running and try again.', false);
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
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
