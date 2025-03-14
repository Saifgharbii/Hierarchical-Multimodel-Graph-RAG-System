/**
 * conversation-manager.js
 * Manages conversations list and related operations
 */

function displayConversations(conversations) {
    const conversationListElement = document.querySelector('.conversation-list');
    conversationListElement.innerHTML = '';

    if (conversations.length === 0) {
        conversationListElement.innerHTML = `
            <div class="p-3 text-center text-muted">
                <p>No conversations yet</p>
                <p>Start a new chat to begin</p>
            </div>
        `;
        return;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    const lastWeek = new Date(today);
    lastWeek.setDate(lastWeek.getDate() - 7);

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

    addConversationEventListeners();
}

function addConversationGroup(container, title, conversations) {
    const groupDiv = document.createElement('div');
    groupDiv.className = 'conversation-group';

    const titleElement = document.createElement('h6');
    titleElement.className = 'text-muted px-3 pt-3 pb-2';
    titleElement.textContent = title;
    groupDiv.appendChild(titleElement);

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

function addConversationEventListeners() {
    const conversationItems = document.querySelectorAll('.conversation-item');
    conversationItems.forEach(item => {
        item.addEventListener('click', function(e) {
            if (e.target.closest('.rename-btn') || e.target.closest('.delete-btn')) {
                return;
            }
            openConversation(this.getAttribute('data-id'));
        });
    });

    const renameButtons = document.querySelectorAll('.rename-btn');
    renameButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            openRenameModal(this.closest('.conversation-item').getAttribute('data-id'));
        });
    });

    const deleteButtons = document.querySelectorAll('.delete-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            openDeleteModal(this.closest('.conversation-item').getAttribute('data-id'));
        });
    });
}

function startNewChat() {
    currentConversationId = null;

    document.getElementById('exploreSection').classList.add('d-none');

    document.getElementById('chatInterface').classList.remove('d-none');

    const activeConversations = document.querySelectorAll('.conversation-item.active');
    activeConversations.forEach(item => {
        item.classList.remove('active');
    });

    const messagesArea = document.querySelector('.chat-messages-area');
    messagesArea.innerHTML = '';

    document.querySelector('.chat-input-container input').focus();
}

function openConversation(conversationId) {
    currentConversationId = conversationId;

    document.getElementById('exploreSection').classList.add('d-none');

    document.getElementById('chatInterface').classList.remove('d-none');

    const activeConversations = document.querySelectorAll('.conversation-item.active');
    activeConversations.forEach(item => {
        item.classList.remove('active');
    });

    const selectedConversation = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    if (selectedConversation) {
        selectedConversation.classList.add('active');
    }

    loadConversationMessages(conversationId);

    document.querySelector('.chat-input-container input').focus();
}

function openRenameModal(conversationId) {
    const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    const currentTitle = conversationItem.querySelector('.conversation-title').textContent;

    const renameInput = document.getElementById('renameInput');
    renameInput.value = currentTitle;

    document.getElementById('conversationId').value = conversationId;

    const renameModal = new bootstrap.Modal(document.getElementById('renameModal'));
    renameModal.show();

    renameInput.focus();
}

function openDeleteModal(conversationId) {
    document.getElementById('deleteConversationId').value = conversationId;

    const deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'));
    deleteModal.show();
}