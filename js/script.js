document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const textInput = document.getElementById('text-input');
    const originalText = document.getElementById('original-text');
    const correctedText = document.getElementById('corrected-text');
    const errorCount = document.querySelector('.error-count');
    const correctionsList = document.getElementById('corrections-list');
    const correctButton = document.getElementById('correct-button');
    const clearButton = document.querySelector('.btn-secondary');
    const resultsSection = document.getElementById('results-section');
    const charCount = document.querySelector('.char-count');
    const fileInput = document.querySelector('input[type="file"]');
    const copyButton = document.getElementById('copy-button');

    // Debounce function to limit API calls
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Function to check if we should trigger correction
    function shouldTriggerCorrection(text, lastChar) {
        // Trigger on space, period, comma, exclamation, question mark, or newline
        const triggerChars = [' ', '.', ',', '!', '?', '\n'];
        return triggerChars.includes(lastChar) || text.length === 0;
    }

    let lastProcessedText = '';

    // Real-time correction function
    const correctInRealTime = debounce(async (text) => {
        if (text.trim() === lastProcessedText.trim()) return;
        lastProcessedText = text.trim();

        if (text.length > 10000) {
            charCount.classList.add('text-danger');
            return;
        }

        try {
            const response = await fetch('/correct', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            // Don't show alert for real-time corrections
            resultsSection.style.display = 'none';
        }
    }, 500); // Wait 500ms after last keystroke

    // Update character count and trigger correction
    textInput.addEventListener('input', function(e) {
        const text = this.value;
        const length = text.length;
        charCount.textContent = `${length} / 10,000 characters`;
        
        if (length > 10000) {
            charCount.classList.add('text-danger');
        } else {
            charCount.classList.remove('text-danger');
            
            // Show results section as soon as user starts typing
            if (length > 0) {
                resultsSection.style.display = 'block';
            } else {
                resultsSection.style.display = 'none';
            }

            // Get the last character typed
            const lastChar = text[length - 1] || '';
            
            // If we should trigger correction, do it
            if (shouldTriggerCorrection(text, lastChar)) {
                correctInRealTime(text);
            }
        }
    });

    // File upload handling
    fileInput.addEventListener('change', async function(e) {
        if (this.files.length === 0) return;

        const file = this.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            correctButton.disabled = true;
            correctButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            displayResults(data);
            // Don't clear the file input, just disable it until Clear is clicked
            fileInput.disabled = true;

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the file: ' + error.message);
        } finally {
            correctButton.disabled = false;
            correctButton.innerHTML = '<i class="fas fa-check"></i> Correct Grammar';
        }
    });

    // Clear button functionality
    clearButton.addEventListener('click', function() {
        textInput.value = '';
        resultsSection.style.display = 'none';
        charCount.textContent = '0 / 10,000 characters';
        fileInput.value = ''; // Clear the file input
        fileInput.disabled = false; // Re-enable the file input
        lastProcessedText = '';
    });

    // Manual correction button (now optional since we have real-time correction)
    correctButton.addEventListener('click', async function() {
        const text = textInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to correct.');
            return;
        }

        if (text.length > 10000) {
            alert('Text is too long. Maximum 10,000 characters allowed.');
            return;
        }

        try {
            correctButton.disabled = true;
            correctButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Correcting...';

            const response = await fetch('/correct', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while correcting the text: ' + error.message);
        } finally {
            correctButton.disabled = false;
            correctButton.innerHTML = '<i class="fas fa-check"></i> Correct Grammar';
        }
    });

    // Copy button functionality
    copyButton.addEventListener('click', async function() {
        try {
            const textToCopy = correctedText.textContent;
            await navigator.clipboard.writeText(textToCopy);
            
            // Visual feedback
            const originalText = copyButton.innerHTML;
            copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyButton.classList.add('copied');
            
            // Reset button after 2 seconds
            setTimeout(() => {
                copyButton.innerHTML = originalText;
                copyButton.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
            alert('Failed to copy text to clipboard');
        }
    });

    // Display results function
    function displayResults(data) {
        // Show results section
        resultsSection.style.display = 'block';

        // Update text areas
        originalText.textContent = data.original;
        correctedText.textContent = data.corrected;
        
        // Update error count with color
        const errorCountText = `${data.error_count} errors found`;
        errorCount.textContent = errorCountText;
        
        // Apply appropriate class based on error count
        errorCount.classList.remove('has-errors', 'no-errors');
        if (data.error_count === 0) {
            errorCount.classList.add('no-errors');
        } else {
            errorCount.classList.add('has-errors');
        }

        // Clear previous corrections
        correctionsList.innerHTML = '';

        // Add new corrections
        if (data.errors && data.errors.length > 0) {
            data.errors.forEach((correction, index) => {
                const correctionDiv = document.createElement('div');
                correctionDiv.className = 'correction-item';
                
                const typeClass = correction.category.toLowerCase().includes('typo') ? 'type-typo' : 'type-grammar';
                
                correctionDiv.innerHTML = `
                    <div class="correction-type ${typeClass}">
                        ${index + 1}. ${correction.category}
                    </div>
                    <div class="correction-context">
                        <strong>Context:</strong> "${correction.original}"
                    </div>
                    <div class="correction-context">
                        <strong>Correction:</strong> "${correction.correction}"
                    </div>
                    <div class="correction-context">
                        <strong>Explanation:</strong> ${correction.message}
                    </div>
                    ${correction.suggestions ? `
                        <div class="correction-suggestions">
                            ${correction.suggestions.map(s => `<span class="suggestion-chip">${s}</span>`).join('')}
                        </div>
                    ` : ''}
                `;
                
                correctionsList.appendChild(correctionDiv);
            });
        }

        // Don't scroll for real-time corrections
        if (!lastProcessedText) {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
});