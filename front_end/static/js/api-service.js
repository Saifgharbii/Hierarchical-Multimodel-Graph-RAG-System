/**
 * api-service.js
 * Handles all API communication with the backend
 */

const API_BASE_URL = 'http://127.0.0.1:5000/api/chat';

let currentConversationId = null;
let currentSettings = {
    temperature: 0.5,
    maxTokens: 2000,
    topP: 1.0,
    topK: 5,
    contextLength: 6,
    systemPrompt: "You're a helpful assistant."
};

// Fetch all conversations
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

// Send a message to the API
async function sendMessage(message) {
    if (!message.trim()) return;

    addMessageToUI('user', message);

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

        loadingMessage.remove();

        currentConversationId = data.conversationId;

        addMessageToUI('bot', data.response);

        if (!currentConversationId) {
            loadConversations();
        }
    } catch (error) {
        console.error('Error sending message:', error);

        loadingMessage.remove();

        addErrorMessageToUI('Failed to get response. Please try again.');
    }
}

// Load messages for a specific conversation
async function loadConversationMessages(conversationId) {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/messages`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const messages = await response.json();

        const messagesArea = document.querySelector('.chat-messages-area');
        messagesArea.innerHTML = '';

        messages.forEach(msg => {
            addMessageToUI(msg.role, msg.content);
        });

        messagesArea.scrollTop = messagesArea.scrollHeight;
    } catch (error) {
        console.error('Error loading conversation messages:', error);
        const messagesArea = document.querySelector('.chat-messages-area');
        messagesArea.innerHTML = '<div class="text-center text-danger mt-3">Failed to load conversation</div>';
    }
}

// Rename a conversation
async function saveRename() {
    const newTitle = document.getElementById('renameInput').value.trim();
    const conversationId = document.getElementById('conversationId').value;

    if (newTitle) {
        try {
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

            const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
            conversationItem.querySelector('.conversation-title').textContent = newTitle;

            const renameModalEl = document.getElementById('renameModal');
            const renameModal = bootstrap.Modal.getInstance(renameModalEl);
            renameModal.hide();

            showToast('Conversation renamed successfully');
        } catch (error) {
            console.error('Error renaming conversation:', error);
            showToast('Failed to rename conversation');
        }
    }
}

// Delete a conversation
async function confirmDelete() {
    const conversationId = document.getElementById('deleteConversationId').value;

    try {
        const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/delete`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);

        if (conversationItem) {
            const wasActive = conversationItem.classList.contains('active');

            conversationItem.remove();

            if (wasActive) {
                document.getElementById('chatInterface').classList.add('d-none');
                document.getElementById('exploreSection').classList.remove('d-none');

                currentConversationId = null;
            }
        }

        const deleteModalEl = document.getElementById('deleteModal');
        const deleteModal = bootstrap.Modal.getInstance(deleteModalEl);
        deleteModal.hide();

        showToast('Conversation deleted successfully');
    } catch (error) {
        console.error('Error deleting conversation:', error);
        showToast('Failed to delete conversation');
    }
}