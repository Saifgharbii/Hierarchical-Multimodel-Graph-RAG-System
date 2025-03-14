// settings-manager.js - Handles configuration settings for the chat

function initSliders() {
    initSliderWithInput('contextLengthSlider', '.context-length-value');
    initSliderWithInput('maxTokensSlider', '.max-tokens-value');
    initSliderWithInput('temperatureSlider', '.temperature-value');
    initSliderWithInput('topPSlider', '.top-p-value');
    initSliderWithInput('topKSlider', '.top-k-value');
}

function initSliderWithInput(sliderId, inputSelector) {
    const slider = document.getElementById(sliderId);
    const input = document.querySelector(inputSelector);

    slider.addEventListener('input', function() {
        input.value = this.value;
    });

    input.addEventListener('change', function() {
        let value = parseFloat(this.value);

        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const step = parseFloat(slider.step) || 1;

        if (value < min) value = min;
        if (value > max) value = max;

        if (step !== 1) {
            value = Math.round(value / step) * step;
            value = parseFloat(value.toFixed(2));
        }

        this.value = value;
        slider.value = value;
    });
}

function openSettingsModal() {
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

    const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));
    settingsModal.show();
}

function saveSettings() {
    currentSettings.contextLength = parseInt(document.querySelector('.context-length-value').value);
    currentSettings.maxTokens = parseInt(document.querySelector('.max-tokens-value').value);
    currentSettings.temperature = parseFloat(document.querySelector('.temperature-value').value);
    currentSettings.topP = parseFloat(document.querySelector('.top-p-value').value);
    currentSettings.topK = parseInt(document.querySelector('.top-k-value').value);
    currentSettings.systemPrompt = document.querySelector('#settingsModal textarea').value;

    const settingsModalEl = document.getElementById('settingsModal');
    const settingsModal = bootstrap.Modal.getInstance(settingsModalEl);
    settingsModal.hide();

    showToast('Settings saved successfully');
}

function incrementValue(input) {
    let value = parseFloat(input.value);
    const step = parseFloat(input.step) || 1;
    const max = parseFloat(input.max);

    value += step;
    if (value > max) value = max;

    value = parseFloat(value.toFixed(2));

    input.value = value;

    const sliderId = input.closest('.d-flex').querySelector('.form-range').id;
    document.getElementById(sliderId).value = value;
}

function decrementValue(input) {
    let value = parseFloat(input.value);
    const step = parseFloat(input.step) || 1;
    const min = parseFloat(input.min);

    value -= step;
    if (value < min) value = min;

    value = parseFloat(value.toFixed(2));

    input.value = value;

    const sliderId = input.closest('.d-flex').querySelector('.form-range').id;
    document.getElementById(sliderId).value = value;
}

function resetToDefault(resetButton) {
    const container = resetButton.closest('.mb-4, .position-relative');

    if (container) {
        const slider = container.querySelector('.form-range');
        const input = container.querySelector('input[type="number"]');
        const textarea = container.querySelector('textarea');

        if (slider && input) {
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

            slider.value = defaultValue;
            input.value = defaultValue;
        } else if (textarea) {
            textarea.value = "You're a helpful assistant.";
        }
    }
}