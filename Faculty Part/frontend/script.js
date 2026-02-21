const API_URL = 'http://localhost:8000';

const messagesContainer = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const statusElement = document.getElementById('status');
const statusText = document.getElementById('statusText');

// Check API connection
async function checkConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            statusElement.classList.add('connected');
            statusText.textContent = 'Connected';
            return true;
        }
    } catch (error) {
        statusElement.classList.remove('connected');
        statusText.textContent = 'API not available';
        return false;
    }
}

// Add message to chat
function addMessage(content, isUser, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = '<strong>Sources:</strong>';
        
        const sourcesList = document.createElement('ul');
        sources.forEach(source => {
            const li = document.createElement('li');
            li.textContent = `• ${source.title}`;
            if (source.date) {
                li.textContent += ` (${source.date})`;
            }
            sourcesList.appendChild(li);
        });
        
        sourcesDiv.appendChild(sourcesList);
        contentDiv.appendChild(sourcesDiv);
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

// Add loading indicator
function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant loading';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

// Send query to API
async function sendQuery(query) {
    const loadingMessage = addLoadingMessage();
    sendBtn.disabled = true;
    
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
        
        // Add assistant response
        addMessage(data.answer, false, data.sources);
        
    } catch (error) {
        loadingMessage.remove();
        addMessage('Sorry, I encountered an error. Please make sure the API is running.', false);
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
    
    // Send to API
    await sendQuery(query);
});

// Handle Enter key
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
    }
});

// Initialize
(async () => {
    const connected = await checkConnection();
    if (connected) {
        addMessage('Hello! I\'m your Faculty Assistant. Ask me anything about faculty policies, procedures, or guidelines.', false);
    } else {
        addMessage('Unable to connect to the API. Please make sure the backend is running on http://localhost:8000', false);
    }
    userInput.focus();
})();

// Check connection every 30 seconds
setInterval(checkConnection, 30000);
