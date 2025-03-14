// main.js - Entry point that initializes the application
document.addEventListener('DOMContentLoaded', function() {
    initBootstrapComponents();
    loadConversations();
    initEventListeners();
    initSliders();
    addAnimationEffects();
});

function initBootstrapComponents() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

function initEventListeners() {
    document.getElementById('settingsBtn').addEventListener('click', openSettingsModal);
    document.getElementById('settingsBtnMobile').addEventListener('click', openSettingsModal);
    document.getElementById('newChatBtn').addEventListener('click', startNewChat);
    document.querySelector('#settingsModal .btn-primary').addEventListener('click', saveSettings);

    const chatInputForm = document.querySelector('.chat-input-container');
    const chatInputField = chatInputForm.querySelector('input');
    const sendButton = chatInputForm.querySelector('button.btn-primary');

    sendButton.addEventListener('click', function() {
        sendMessage(chatInputField.value);
        chatInputField.value = '';
    });

    chatInputField.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage(this.value);
            this.value = '';
        }
    });

    document.getElementById('saveRenameBtn').addEventListener('click', saveRename);
    document.getElementById('confirmDeleteBtn').addEventListener('click', confirmDelete);

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

    const resetButtons = document.querySelectorAll('.reset-btn');
    resetButtons.forEach(button => {
        button.addEventListener('click', function() {
            resetToDefault(this);
        });
    });
}

function addAnimationEffects() {
    const chatArea = document.querySelector('.chat-messages-area');

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

    observer.observe(chatArea, { childList: true });

    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('btn')) {
            createRipple(e);
        }
    });
}

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