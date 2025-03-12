/**
 * LLMChat SPA JavaScript
 * Handles the SPA behavior and UI interactions for the LLMChat application
 */

// API endpoint base URL
const API_BASE_URL = '/api/chat';

// Current conversation state
let currentConversationId = null;
let currentSettings = {
    temperature: 0.5,
    maxTokens: 2000,
    topP: 1.0,
    topK: 5,
    contextLength: 6,
    systemPrompt: "You're a helpful assistant."
};

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    initBootstrapComponents();
    
    // Load existing conversations from server
    loadConversations();
    
    // Initialize all event listeners
    initEventListeners();
    
    // Initialize sliders with their text input counterparts
    initSliders();

    // Add animation effects
    addAnimationEffects();
});

/**
 * Initialize Bootstrap components that require JavaScript
 */
function initBootstrapComponents() {
    // Initialize all tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize all popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Initialize all event listeners for the application
 */
function initEventListeners() {
    // Settings button click handlers
    document.getElementById('settingsBtn').addEventListener('click', openSettingsModal);
    document.getElementById('settingsBtnMobile').addEventListener('click', openSettingsModal);
    
    // New chat button click handler
    document.getElementById('newChatBtn').addEventListener('click', startNewChat);
    
    // Save settings button click handler
    document.querySelector('#settingsModal .btn-primary').addEventListener('click', saveSettings);
    
    // Chat input submission
    const chatInputForm = document.querySelector('.chat-input-container');
    const chatInputField = chatInputForm.querySelector('input');
    const sendButton = chatInputForm.querySelector('button.btn-primary');
    
    // Send message when button is clicked
    sendButton.addEventListener('click', function() {
        sendMessage(chatInputField.value);
        chatInputField.value = '';
    });
    
    // Send message when Enter key is pressed
    chatInputField.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage(this.value);
            this.value = '';
        }
    });
    
    // Save rename button click handler
    document.getElementById('saveRenameBtn').addEventListener('click', saveRename);
    
    // Confirm delete button click handler
    document.getElementById('confirmDeleteBtn').addEventListener('click', confirmDelete);
    
    // Increment/Decrement buttons click handlers
    const incrementButtons = document.querySelectorAll('.increment-btn');
    incrementButtons.forEach(button => {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('input[type="number"]');
            incrementValue(input);
        });
    });
    
    const decrementButtons = document.querySelectorAll('.decrement-btn');
    decrementButtons.forEach(button => {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('input[type="number"]');
            decrementValue(input);
        });
    });
    
    // Reset buttons click handlers
    const resetButtons = document.querySelectorAll('.reset-btn');
    resetButtons.forEach(button => {
        button.addEventListener('click', function() {
            resetToDefault(this);
        });
    });
}

/**
 * Load conversations from the server
 */
async function loadConversations() {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const conversations = await response.json();
        displayConversations(conversations);
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

/**
 * Display conversations in the sidebar
 * @param {Array} conversations - The list of conversations from the server
 */
function displayConversations(conversations) {
    // Clear existing conversations
    const conversationListElement = document.querySelector('.conversation-list');
    conversationListElement.innerHTML = '';
    
    if (conversations.length === 0) {
        // Show empty state
        conversationListElement.innerHTML = `
            <div class="p-3 text-center text-muted">
                <p>No conversations yet</p>
                <p>Start a new chat to begin</p>
            </div>
        `;
        return;
    }
    
    // Group conversations by date
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    const lastWeek = new Date(today);
    lastWeek.setDate(lastWeek.getDate() - 7);
    
    // Create groups
    const groups = {
        today: [],
        yesterday: [],
        lastWeek: [],
        older: []
    };
    
    conversations.forEach(conv => {
        const timestamp = new Date(conv.lastMessageTimestamp);
        
        if (timestamp >= today) {
            groups.today.push(conv);
        } else if (timestamp >= yesterday) {
            groups.yesterday.push(conv);
        } else if (timestamp >= lastWeek) {
            groups.lastWeek.push(conv);
        } else {
            groups.older.push(conv);
        }
    });
    
    // Display grouped conversations
    if (groups.today.length > 0) {
        addConversationGroup(conversationListElement, 'Today', groups.today);
    }
    
    if (groups.yesterday.length > 0) {
        addConversationGroup(conversationListElement, 'Yesterday', groups.yesterday);
    }
    
    if (groups.lastWeek.length > 0) {
        addConversationGroup(conversationListElement, 'Last 7 Days', groups.lastWeek);
    }
    
    if (groups.older.length > 0) {
        addConversationGroup(conversationListElement, 'Older', groups.older);
    }
    
    // Add click event listeners to the newly created conversation items
    addConversationEventListeners();
}

/**
 * Add a group of conversations to the conversation list
 * @param {HTMLElement} container - The container element
 * @param {string} title - The group title
 * @param {Array} conversations - The conversations in this group
 */
function addConversationGroup(container, title, conversations) {
    // Create group container
    const groupDiv = document.createElement('div');
    groupDiv.className = 'conversation-group';
    
    // Add group title
    const titleElement = document.createElement('h6');
    titleElement.className = 'text-muted px-3 pt-3 pb-2';
    titleElement.textContent = title;
    groupDiv.appendChild(titleElement);
    
    // Add conversation items
    conversations.forEach(conv => {
        const convItem = document.createElement('div');
        convItem.className = 'conversation-item p-3 border-bottom';
        convItem.setAttribute('data-id', conv.conversationId);
        
        convItem.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span class="conversation-title">${escapeHtml(conv.title)}</span>
                <div class="conversation-actions">
                    <button class="btn btn-sm text-muted rename-btn" title="Rename">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn btn-sm text-muted delete-btn" title="Delete">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
        
        groupDiv.appendChild(convItem);
    });
    
    container.appendChild(groupDiv);
}

/**
 * Add event listeners to conversation items
 */
function addConversationEventListeners() {
    // Conversation item click handler
    const conversationItems = document.querySelectorAll('.conversation-item');
    conversationItems.forEach(item => {
        item.addEventListener('click', function(e) {
            // Prevent click when clicking on action buttons
            if (e.target.closest('.rename-btn') || e.target.closest('.delete-btn')) {
                return;
            }
            openConversation(this.getAttribute('data-id'));
        });
    });
    
    // Rename buttons click handler
    const renameButtons = document.querySelectorAll('.rename-btn');
    renameButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            openRenameModal(this.closest('.conversation-item').getAttribute('data-id'));
        });
    });
    
    // Delete buttons click handler
    const deleteButtons = document.querySelectorAll('.delete-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            openDeleteModal(this.closest('.conversation-item').getAttribute('data-id'));
        });
    });
}

/**
 * Escape HTML special characters
 * @param {string} unsafe - The unsafe string
 * @returns {string} - The escaped string
 */
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Send a message to the server
 * @param {string} message - The message text
 */
async function sendMessage(message) {
    if (!message.trim()) return;
    
    // Add user message to the UI immediately
    addMessageToUI('user', message);
    
    // Show loading indicator
    const loadingMessage = addLoadingMessageToUI();
    
    try {
        const response = await fetch(`${API_BASE_URL}/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                conversationId: currentConversationId,
                settings: currentSettings
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove loading indicator
        loadingMessage.remove();
        
        // Set the current conversation ID
        currentConversationId = data.conversationId;
        
        // Add bot response to the UI
        addMessageToUI('bot', data.response);
        
        // Reload the conversation list to get the new conversation if it was just created
        if (!currentConversationId) {
            loadConversations();
        }
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remove loading indicator
        loadingMessage.remove();
        
        // Show error message
        addErrorMessageToUI('Failed to get response. Please try again.');
    }
}

/**
 * Add a message to the UI
 * @param {string} role - The message role ('user' or 'bot')
 * @param {string} content - The message content
 */
function addMessageToUI(role, content) {
    const messagesArea = document.querySelector('.chat-messages-area');
    
    // Create message container
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message mb-3`;
    
    // Create message content with styling based on role
    const messageContent = document.createElement('div');
    messageContent.className = role === 'user' ? 'message-bubble user-bubble' : 'message-bubble bot-bubble';
    
    // Use regex to replace markdown code blocks with HTML
    let formattedContent = content.replace(/```([\s\S]*?)```/g, function(match, p1) {
        return `<pre><code>${escapeHtml(p1)}</code></pre>`;
    });
    
    // Replace single line code with inline code elements
    formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert newlines to <br> tags
    formattedContent = formattedContent.replace(/\n/g, '<br>');
    
    messageContent.innerHTML = formattedContent;
    messageDiv.appendChild(messageContent);
    
    // Add message to the chat area
    messagesArea.appendChild(messageDiv);
    
    // Scroll to the bottom
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

/**
 * Add a loading message to the UI
 * @returns {HTMLElement} - The loading message element
 */
function addLoadingMessageToUI() {
    const messagesArea = document.querySelector('.chat-messages-area');
    
    // Create loading message container
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message mb-3 loading-message';
    
    // Create loading animation
    const loadingContent = document.createElement('div');
    loadingContent.className = 'message-bubble bot-bubble';
    loadingContent.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    loadingDiv.appendChild(loadingContent);
    
    // Add loading message to the chat area
    messagesArea.appendChild(loadingDiv);
    
    // Scroll to the bottom
    messagesArea.scrollTop = messagesArea.scrollHeight;
    
    return loadingDiv;
}

/**
 * Add an error message to the UI
 * @param {string} errorText - The error message text
 */
function addErrorMessageToUI(errorText) {
    const messagesArea = document.querySelector('.chat-messages-area');
    
    // Create error message container
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error-message mb-3';
    
    // Create error content
    const errorContent = document.createElement('div');
    errorContent.className = 'message-bubble error-bubble';
    errorContent.textContent = errorText;
    
    errorDiv.appendChild(errorContent);
    
    // Add error message to the chat area
    messagesArea.appendChild(errorDiv);
    
    // Scroll to the bottom
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

/**
 * Load and display messages for a conversation
 * @param {string} conversationId - The ID of the conversation to load
 */
async function loadConversationMessages(conversationId) {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/messages`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const messages = await response.json();
        
        // Clear existing messages
        const messagesArea = document.querySelector('.chat-messages-area');
        messagesArea.innerHTML = '';
        
        // Add each message to the UI
        messages.forEach(msg => {
            addMessageToUI(msg.role, msg.content);
        });
        
        // Scroll to the bottom
        messagesArea.scrollTop = messagesArea.scrollHeight;
    } catch (error) {
        console.error('Error loading conversation messages:', error);
        // Show error message
        const messagesArea = document.querySelector('.chat-messages-area');
        messagesArea.innerHTML = '<div class="text-center text-danger mt-3">Failed to load conversation</div>';
    }
}

function addAnimationEffects() {
    // Add animation class to messages when they appear
    const chatArea = document.querySelector('.chat-messages-area');
    
    // Create an observer for new chat messages
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach(node => {
                    if (node.classList && !node.classList.contains('animated')) {
                        node.classList.add('animated');
                    }
                });
            }
        });
    });
    
    // Start observing chat area for new messages
    observer.observe(chatArea, { childList: true });
    
    // Add ripple effect to all buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('btn')) {
            createRipple(e);
        }
    });
}

/**
 * Create ripple effect on button click
 * @param {Event} e - The click event
 */
function createRipple(e) {
    const button = e.currentTarget;
    
    const circle = document.createElement('span');
    const diameter = Math.max(button.clientWidth, button.clientHeight);
    
    circle.style.width = circle.style.height = `${diameter}px`;
    circle.style.left = `${e.clientX - button.offsetLeft - diameter / 2}px`;
    circle.style.top = `${e.clientY - button.offsetTop - diameter / 2}px`;
    circle.classList.add('ripple');
    
    const ripple = button.getElementsByClassName('ripple')[0];
    
    if (ripple) {
        ripple.remove();
    }
    
    button.appendChild(circle);
}

/**
 * Initialize sliders with their text input counterparts
 */
function initSliders() {
    // Context Length Slider
    initSliderWithInput('contextLengthSlider', '.context-length-value');
    
    // Max Tokens Slider
    initSliderWithInput('maxTokensSlider', '.max-tokens-value');
    
    // Temperature Slider
    initSliderWithInput('temperatureSlider', '.temperature-value');
    
    // TopP Slider
    initSliderWithInput('topPSlider', '.top-p-value');
    
    // TopK Slider
    initSliderWithInput('topKSlider', '.top-k-value');
}

/**
 * Initialize a slider with its text input counterpart
 * @param {string} sliderId - The ID of the slider element
 * @param {string} inputSelector - The selector for the text input element
 */
function initSliderWithInput(sliderId, inputSelector) {
    const slider = document.getElementById(sliderId);
    const input = document.querySelector(inputSelector);
    
    // Update input when slider changes
    slider.addEventListener('input', function() {
        input.value = this.value;
    });
    
    // Update slider when input changes
    input.addEventListener('change', function() {
        let value = parseFloat(this.value);
        
        // Enforce min and max limits
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const step = parseFloat(slider.step) || 1;
        
        if (value < min) value = min;
        if (value > max) value = max;
        
        // Round to the nearest step value if needed
        if (step !== 1) {
            value = Math.round(value / step) * step;
            value = parseFloat(value.toFixed(2)); // Handle floating point precision
        }
        
        this.value = value;
        slider.value = value;
    });
}

/**
 * Open the settings modal
 */
function openSettingsModal() {
    // Load current settings into the modal
    document.getElementById('contextLengthSlider').value = currentSettings.contextLength;
    document.querySelector('.context-length-value').value = currentSettings.contextLength;
    
    document.getElementById('maxTokensSlider').value = currentSettings.maxTokens;
    document.querySelector('.max-tokens-value').value = currentSettings.maxTokens;
    
    document.getElementById('temperatureSlider').value = currentSettings.temperature;
    document.querySelector('.temperature-value').value = currentSettings.temperature;
    
    document.getElementById('topPSlider').value = currentSettings.topP;
    document.querySelector('.top-p-value').value = currentSettings.topP;
    
    document.getElementById('topKSlider').value = currentSettings.topK;
    document.querySelector('.top-k-value').value = currentSettings.topK;
    
    document.querySelector('#settingsModal textarea').value = currentSettings.systemPrompt;
    
    // Show the modal
    const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));
    settingsModal.show();
}

/**
 * Save settings from the modal
 */
function saveSettings() {
    // Get values from the modal
    currentSettings.contextLength = parseInt(document.querySelector('.context-length-value').value);
    currentSettings.maxTokens = parseInt(document.querySelector('.max-tokens-value').value);
    currentSettings.temperature = parseFloat(document.querySelector('.temperature-value').value);
    currentSettings.topP = parseFloat(document.querySelector('.top-p-value').value);
    currentSettings.topK = parseInt(document.querySelector('.top-k-value').value);
    currentSettings.systemPrompt = document.querySelector('#settingsModal textarea').value;
    
    // Hide the modal
    const settingsModalEl = document.getElementById('settingsModal');
    const settingsModal = bootstrap.Modal.getInstance(settingsModalEl);
    settingsModal.hide();
    
    // Show a toast notification
    showToast('Settings saved successfully');
}

/**
 * Show a toast notification
 * @param {string} message - The message to display
 */
function showToast(message) {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
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
    
    // Add to container
    toastContainer.appendChild(toastEl);
    
    // Initialize and show the toast
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
    
    // Remove the toast after it's hidden
    toastEl.addEventListener('hidden.bs.toast', function () {
        toastEl.remove();
    });
}

/**
 * Start a new chat
 */
function startNewChat() {
    // Reset current conversation ID
    currentConversationId = null;
    
    // Hide the explore section
    document.getElementById('exploreSection').classList.add('d-none');
    
    // Show the chat interface
    document.getElementById('chatInterface').classList.remove('d-none');
    
    // Deselect any active conversation
    const activeConversations = document.querySelectorAll('.conversation-item.active');
    activeConversations.forEach(item => {
        item.classList.remove('active');
    });
    
    // Clear the chat messages
    const messagesArea = document.querySelector('.chat-messages-area');
    messagesArea.innerHTML = '';
    
    // Focus on the input field
    document.querySelector('.chat-input-container input').focus();
}

/**
 * Open an existing conversation
 * @param {string} conversationId - The ID of the conversation to open
 */
async function openConversation(conversationId) {
    // Set the current conversation ID
    currentConversationId = conversationId;
    
    // Hide the explore section
    document.getElementById('exploreSection').classList.add('d-none');
    
    // Show the chat interface
    document.getElementById('chatInterface').classList.remove('d-none');
    
    // Deselect any active conversation
    const activeConversations = document.querySelectorAll('.conversation-item.active');
    activeConversations.forEach(item => {
        item.classList.remove('active');
    });
    
    // Mark the selected conversation as active
    const selectedConversation = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    if (selectedConversation) {
        selectedConversation.classList.add('active');
    }
    
    // Load conversation messages
    await loadConversationMessages(conversationId);
    
    // Focus on the input field
    document.querySelector('.chat-input-container input').focus();
}

/**
 * Open the rename modal for a conversation
 * @param {string} conversationId - The ID of the conversation to rename
 */
function openRenameModal(conversationId) {
    // Get the current conversation title
    const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    const currentTitle = conversationItem.querySelector('.conversation-title').textContent;
    
    // Set the current title in the input field
    const renameInput = document.getElementById('renameInput');
    renameInput.value = currentTitle;
    
    // Set the conversation ID in the hidden field
    document.getElementById('conversationId').value = conversationId;
    
    // Show the modal
    const renameModal = new bootstrap.Modal(document.getElementById('renameModal'));
    renameModal.show();
    
    // Focus the input field
    renameInput.focus();
}

/**
 * Save the new name for a conversation
 */
async function saveRename() {
    // Get the new title and conversation ID
    const newTitle = document.getElementById('renameInput').value.trim();
    const conversationId = document.getElementById('conversationId').value;
    
    if (newTitle) {
        try {
            // Update the conversation title in the backend
            const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/rename`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    newTitle: newTitle
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Update the conversation title in the UI
            const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
            conversationItem.querySelector('.conversation-title').textContent = newTitle;
            
            // Hide the modal
            const renameModalEl = document.getElementById('renameModal');
            const renameModal = bootstrap.Modal.getInstance(renameModalEl);
            renameModal.hide();
            
            // Show success toast
            showToast('Conversation renamed successfully');
        } catch (error) {
            console.error('Error renaming conversation:', error);
            // Show error toast
            showToast('Failed to rename conversation');
        }
    }
}

/**
 * Open the delete confirmation modal for a conversation
 * @param {string} conversationId - The ID of the conversation to delete
 */
function openDeleteModal(conversationId) {
    // Set the conversation ID in the hidden field
    document.getElementById('deleteConversationId').value = conversationId;
    
    // Show the modal
    const deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'));
    deleteModal.show();
}

/**
 * Confirm deletion of a conversation
 */
async function confirmDelete() {
    // Get the conversation ID
    const conversationId = document.getElementById('deleteConversationId').value;
    
    try {
        // Delete the conversation in the backend
        const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/delete`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Remove the conversation item from the UI
        const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
        
        if (conversationItem) {
            // Check if this was the active conversation
            const wasActive = conversationItem.classList.contains('active');
            
            // Remove the item
            conversationItem.remove();
            
            // If it was active, show the explore section
            if (wasActive) {
                document.getElementById('chatInterface').classList.add('d-none');
                document.getElementById('exploreSection').classList.remove('d-none');
                
                // Reset current conversation ID
                currentConversationId = null;
            }
        }
        
        // Hide the modal
        const deleteModalEl = document.getElementById('deleteModal');
        const deleteModal = bootstrap.Modal.getInstance(deleteModalEl);
        deleteModal.hide();
        
        // Show success toast
        showToast('Conversation deleted successfully');
    } catch (error) {
        console.error('Error deleting conversation:', error);
        // Show error toast
        showToast('Failed to delete conversation');
    }
}

/**
 * Increment the value of a numeric input
 * @param {HTMLElement} input - The input element to increment
 */
function incrementValue(input) {
    let value = parseFloat(input.value);
    const step = parseFloat(input.step) || 1;
    const max = parseFloat(input.max);
    
    value += step;
    if (value > max) value = max;
    
    // Round to handle floating point precision
    value = parseFloat(value.toFixed(2));
    
    input.value = value;
    
    // Also update the slider if present
    const sliderId = input.closest('.d-flex').querySelector('.form-range').id;
    document.getElementById(sliderId).value = value;
}

/**
 * Decrement the value of a numeric input
 * @param {HTMLElement} input - The input element to decrement
 */
function decrementValue(input) {
    let value = parseFloat(input.value);
    const step = parseFloat(input.step) || 1;
    const min = parseFloat(input.min);
    
    value -= step;
    if (value < min) value = min;
    
    // Round to handle floating point precision
    value = parseFloat(value.toFixed(2));
    
    input.value = value;
    
    // Also update the slider if present
    const sliderId = input.closest('.d-flex').querySelector('.form-range').id;
    document.getElementById(sliderId).value = value;
}

/**
 * Reset a setting to its default value
 * @param {HTMLElement} resetButton - The reset button element that was clicked
 */
function resetToDefault(resetButton) {
    // Determine what to reset based on the parent container
    const container = resetButton.closest('.mb-4, .position-relative');
    
    if (container) {
        const slider = container.querySelector('.form-range');
        const input = container.querySelector('input[type="number"]');
        const textarea = container.querySelector('textarea');
        
        if (slider && input) {
            // Get default value from data attribute or use predefined defaults
            let defaultValue;
            const settingName = slider.id.replace('Slider', '');
            
            switch(settingName.toLowerCase()) {
                case 'contextlength':
                    defaultValue = 6;
                    break;
                case 'maxtokens':
                    defaultValue = 2000;
                    break;
                case 'temperature':
                    defaultValue = 0.5;
                    break;
                case 'topp':
                    defaultValue = 1.0;
                    break;
                case 'topk':
                    defaultValue = 5;
                    break;
                default:
                    defaultValue = parseFloat(slider.getAttribute('data-default') || slider.min);
            }
            
            // Update slider and input values
            slider.value = defaultValue;
            input.value = defaultValue;
        } else if (textarea) {
            // Reset system prompt to default
            textarea.value = "You're a helpful assistant.";
        }
    }
}