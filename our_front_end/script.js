/**
 * LLMChat Interface JavaScript
 * Handles basic interactivity for the LLMChat web interface
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const settingsBtn = document.getElementById('settings-btn');
    const sectionHeadings = document.querySelectorAll('.section-heading');
    const navItems = document.querySelectorAll('.nav-item');
    const contentSections = document.querySelectorAll('.content-section');
    const chatView = document.getElementById('chat-view');
    const settingsView = document.getElementById('settings-view');
    const assistantsView = document.getElementById('assistants-view');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatInput = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.btn-send');
    const examplePrompts = document.querySelectorAll('.btn-prompt');
    
    // Initialize range input links
    initRangeInputs();
    
    // Initialize chat input auto-resize
    initChatInputResize();
    
    // Navigation Functions
    
    /**
     * Shows a specific content section and hides others
     * @param {string} sectionId - The ID of the section to show
     */
    function showSection(sectionId) {
        // Hide all content sections
        contentSections.forEach(section => {
            section.classList.remove('active');
        });
        
        // Show the target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
        }
        
        // Update active state in navigation
        updateNavActiveState(sectionId);
    }
    
    /**
     * Updates the active state of navigation items based on the current section
     * @param {string} sectionId - The ID of the current active section
     */
    function updateNavActiveState(sectionId) {
        // Reset all active states
        sectionHeadings.forEach(heading => {
            heading.classList.remove('active');
        });
        
        navItems.forEach(item => {
            item.classList.remove('active');
        });
        
        // Set active state based on current section
        if (sectionId === 'chat-view') {
            // No specific nav item to highlight for chat view
        } else if (sectionId === 'settings-view') {
            // Highlight settings button
            settingsBtn.classList.add('active');
        } else if (sectionId === 'assistants-view') {
            // Highlight Explore Assistants section
            document.querySelector('.section-heading').classList.add('active');
        }
    }
    
    /**
     * Initializes the linking between range inputs and their corresponding number inputs
     */
    function initRangeInputs() {
        // Link range inputs with their corresponding number inputs
        const ranges = document.querySelectorAll('.form-range');
        
        ranges.forEach(range => {
            const rangeId = range.id;
            const valueId = rangeId.replace('Range', '');
            const valueInput = document.getElementById(valueId);
            
            if (valueInput) {
                // Update number input when range changes
                range.addEventListener('input', function() {
                    valueInput.value = this.value;
                });
                
                // Update range when number input changes
                valueInput.addEventListener('input', function() {
                    range.value = this.value;
                });
            }
        });
    }
    
    /**
     * Initializes auto-resize functionality for the chat input
     */
    function initChatInputResize() {
        if (chatInput) {
            chatInput.addEventListener('input', function() {
                // Reset height to auto to get the correct scrollHeight
                this.style.height = 'auto';
                
                // Set new height based on scrollHeight (with a max height)
                const newHeight = Math.min(this.scrollHeight, 200);
                this.style.height = newHeight + 'px';
            });
        }
    }
    
    /**
     * Simulates starting a new chat
     */
    function startNewChat() {
        // Clear chat input
        if (chatInput) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
        }
        
        // Show chat view
        showSection('chat-view');
    }
    
    /**
     * Handles filling the chat input with example prompts
     * @param {string} promptText - The text of the example prompt
     */
    function fillExamplePrompt(promptText) {
        if (chatInput) {
            chatInput.value = promptText;
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            chatInput.focus();
        }
    }
    
    // Event Listeners
    
    // Settings button click
    if (settingsBtn) {
        settingsBtn.addEventListener('click', function() {
            showSection('settings-view');
        });
    }
    
    // New Chat button click
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function() {
            startNewChat();
        });
    }
    
    // Explore Assistants section click
    sectionHeadings.forEach(heading => {
        heading.addEventListener('click', function() {
            if (this.textContent.trim() === 'Explore Assistants') {
                showSection('assistants-view');
            }
        });
    });
    
    // Example prompts click
    examplePrompts.forEach(prompt => {
        prompt.addEventListener('click', function() {
            fillExamplePrompt(this.textContent);
        });
    });
    
    // Send button click
    if (sendBtn) {
        sendBtn.addEventListener('click', function() {
            // In a real app, this would send the message to the LLM API
            // For this interface prototype, we'll just clear the input
            if (chatInput && chatInput.value.trim()) {
                console.log('Sending message:', chatInput.value);
                chatInput.value = '';
                chatInput.style.height = 'auto';
            }
        });
    }
    
    // Chat input Enter key (with Shift+Enter for new line)
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendBtn.click();
            }
        });
    }
    
    // Initialize the UI with the chat view active
    showSection('chat-view');
});