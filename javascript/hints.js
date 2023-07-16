(function () {
    // mouseover tooltips for various UI elements
    const titles = {
        '🔄': 'Refresh saves',
        '💾': 'save',
        '\uD83D\uDCD2\u200B': 'load saves',
        '\uD83D\uDCD2\u200D': 'load prompts from Civitai extension',
        '🔢': 'add index to prompt and checkpoints',
        
    };

    onUiUpdate(function () {
        gradioApp().querySelectorAll('.batch-checkpoint-prompt').forEach(function (button) {
            const tooltip = titles[button.textContent];
            if (tooltip) {
                button.title = tooltip;
            }
        })
    });
})();