/**
 * ui-utils.js
 * Utility functions for UI elements and DOM manipulation
 */


function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function addMessageToUI(role, content) {
    const messagesArea = document.querySelector('.chat-messages-area');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message mb-3`;

    const messageContent = document.createElement('div');
    messageContent.className = role === 'user' ? 'message-bubble user-bubble' : 'message-bubble bot-bubble';

    let formattedContent = content;

    // Apply Markdown formatting only to assistant messages
    if (role === 'assistant') {
        // Format code blocks
        formattedContent = formattedContent.replace(/```([\s\S]*?)```/g, function(match, p1) {
            return `<pre><code>${escapeHtml(p1)}</code></pre>`;
        });

        // Format inline code
        formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Format bold text
        formattedContent = formattedContent.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Format italic text
        formattedContent = formattedContent.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Format headers
        formattedContent = formattedContent.replace(/^### (.*$)/gm, '<h3>$1</h3>');
        formattedContent = formattedContent.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        formattedContent = formattedContent.replace(/^# (.*$)/gm, '<h1>$1</h1>');

        // Format lists
        formattedContent = formattedContent.replace(/^\- (.*$)/gm, '<li>$1</li>');

        // Format links
        formattedContent = formattedContent.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    }

    // Line breaks for all messages
    formattedContent = formattedContent.replace(/\n/g, '<br>');

    messageContent.innerHTML = formattedContent;
    messageDiv.appendChild(messageContent);

    messagesArea.appendChild(messageDiv);

    messagesArea.scrollTop = messagesArea.scrollHeight;
}

// Make sure you have this escapeHtml function defined somewhere
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function addLoadingMessageToUI() {
    const messagesArea = document.querySelector('.chat-messages-area');

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message mb-3 loading-message';

    const loadingContent = document.createElement('div');
    loadingContent.className = 'message-bubble bot-bubble';
    loadingContent.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    loadingDiv.appendChild(loadingContent);

    messagesArea.appendChild(loadingDiv);

    messagesArea.scrollTop = messagesArea.scrollHeight;

    return loadingDiv;
}

function addErrorMessageToUI(errorText) {
    const messagesArea = document.querySelector('.chat-messages-area');

    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error-message mb-3';

    const errorContent = document.createElement('div');
    errorContent.className = 'message-bubble error-bubble';
    errorContent.textContent = errorText;

    errorDiv.appendChild(errorContent);

    messagesArea.appendChild(errorDiv);

    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function showToast(message) {
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    const toastEl = document.createElement('div');
    toastEl.className = 'toast';
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');

    toastEl.innerHTML = `
        <div class="toast-header">
            <strong class="me-auto">LLMChat</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;

    toastContainer.appendChild(toastEl);

    const toast = new bootstrap.Toast(toastEl);
    toast.show();

    toastEl.addEventListener('hidden.bs.toast', function () {
        toastEl.remove();
    });
}