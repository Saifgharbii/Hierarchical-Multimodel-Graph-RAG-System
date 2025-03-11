/**
 * LLMChat SPA JavaScript
 * Handles the SPA behavior and UI interactions for the LLMChat application
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    initBootstrapComponents();
    
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

function addAnimationEffects() {
    // Add animation class to messages when they appear
    const chatArea = document.querySelector('.chat-messages-area');
    
    // Create an observer for new chat messages
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach(node => {
                    if (node.classList && !node.classList.contains('message-item')) {
                        node.classList.add('message-item');
                    }
                });
            }
        });
    });
    
    // Start observing chat area for new messages
    observer.observe(chatArea, { childList: true });
    
    // Add staggered animation to conversation items
    const conversationItems = document.querySelectorAll('.conversation-item');
    conversationItems.forEach((item, index) => {
        item.style.animationDelay = `${index * 0.05}s`;
        item.style.opacity = '0';
        item.style.animation = 'fadeInUp 0.5s forwards';
    });
    
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
    const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));
    settingsModal.show();
}

/**
 * Start a new chat
 */
function startNewChat() {
    // Hide the explore section
    document.getElementById('exploreSection').classList.add('d-none');
    
    // Show the chat interface
    document.getElementById('chatInterface').classList.remove('d-none');
    
    // Deselect any active conversation
    const activeConversations = document.querySelectorAll('.conversation-item.active');
    activeConversations.forEach(item => {
        item.classList.remove('active');
    });
    
    // In a real application, you would also clear the chat messages
    // and possibly create a new conversation in the backend
    console.log('Starting new chat');
}

/**
 * Open an existing conversation
 * @param {string} conversationId - The ID of the conversation to open
 */
function openConversation(conversationId) {
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
    
    // In a real application, you would load the conversation history
    console.log('Opening conversation:', conversationId);
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
function saveRename() {
    // Get the new title and conversation ID
    const newTitle = document.getElementById('renameInput').value.trim();
    const conversationId = document.getElementById('conversationId').value;
    
    if (newTitle) {
        // Update the conversation title in the UI
        const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
        conversationItem.querySelector('.conversation-title').textContent = newTitle;
        
        // In a real application, you would also update the backend
        console.log('Renaming conversation:', conversationId, 'to:', newTitle);
        
        // Hide the modal
        const renameModalEl = document.getElementById('renameModal');
        const renameModal = bootstrap.Modal.getInstance(renameModalEl);
        renameModal.hide();
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
function confirmDelete() {
    // Get the conversation ID
    const conversationId = document.getElementById('deleteConversationId').value;
    
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
        }
        
        // In a real application, you would also delete from the backend
        console.log('Deleting conversation:', conversationId);
    }
    
    // Hide the modal
    const deleteModalEl = document.getElementById('deleteModal');
    const deleteModal = bootstrap.Modal.getInstance(deleteModalEl);
    deleteModal.hide();
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
        // If it's a textarea (system prompt)
        const textarea = container.querySelector('textarea');
        if (textarea) {
            textarea.value = "You're helpful assistant that can help me with my questions. Today is {{local_date}}.";
            return;
        }
        
        // If it's a slider with input
        const slider = container.querySelector('.form-range');
        const input = container.querySelector('input[type="number"]');
        
        if (slider && input) {
            // Set default values based on slider ID
            let defaultValue = 0;
            
            switch (slider.id) {
                case 'contextLengthSlider':
                    defaultValue = 6;
                    break;
                case 'maxTokensSlider':
                    defaultValue = 2000;
                    break;
                case 'temperatureSlider':
                    defaultValue = 0.5;
                    break;
                case 'topPSlider':
                    defaultValue = 1.0;
                    break;
                case 'topKSlider':
                    defaultValue = 5;
                    break;
            }
            
            slider.value = defaultValue;
            input.value = defaultValue;
        }
    }
}