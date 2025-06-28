class GrammarCorrectionApp {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.currentFile = null;
        this.correctedData = null;
    }

    initializeElements() {
        // Input elements
        this.inputText = document.getElementById('inputText');
        this.fileInput = document.getElementById('fileInput');
        this.charCount = document.getElementById('charCount');
        
        // Button elements
        this.correctBtn = document.getElementById('correctBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.copyBtn = document.getElementById('copyBtn');
        
        // Section elements
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        
        // Result elements
        this.originalText = document.getElementById('originalText');
        this.correctedText = document.getElementById('correctedText');
        this.errorsSection = document.getElementById('errorsSection');
        this.errorsList = document.getElementById('errorsList');
        this.errorCount = document.getElementById('errorCount');
        this.errorMessage = document.getElementById('errorMessage');
        this.realtimeStatus = document.getElementById('realtimeStatus');
        
        // Toast elements
        this.successToast = new bootstrap.Toast(document.getElementById('successToast'));
        this.successMessage = document.getElementById('successMessage');
    }

    attachEventListeners() {
        // Text input events
        this.inputText.addEventListener('input', () => {
            this.updateCharCount();
        });

        // File input events
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        // Button events
        this.correctBtn.addEventListener('click', () => {
            this.correctGrammar();
        });

        this.clearBtn.addEventListener('click', () => {
            this.clearAll();
        });

        this.copyBtn.addEventListener('click', () => {
            this.copyToClipboard();
        });

        // Real-time correction
        let timeout;
        let lastWord = '';
        
        const checkForNewWord = (text) => {
            const words = text.trim().split(/\s+/);
            const currentWord = words[words.length - 1];
            if (currentWord !== lastWord) {
                lastWord = currentWord;
                return true;
            }
            return false;
        };

        this.inputText.addEventListener('input', () => {
            clearTimeout(timeout);
            
            // Show the results section immediately if hidden
            if (this.resultsSection.style.display === 'none') {
                this.resultsSection.style.display = 'block';
            }
            
            const text = this.inputText.value.trim();
            
            // Update immediately for new words or if there's a period, question mark, or exclamation mark
            if (checkForNewWord(text) || /[.!?]$/.test(text)) {
                if (text.length > 0) {
                    this.correctGrammar(true);
                }
            } else {
                // For other changes, wait a short time before updating
                timeout = setTimeout(() => {
                    if (text.length > 0) {
                        this.correctGrammar(true);
                    }
                }, 300);
            }
        });
    }

    updateCharCount() {
        const count = this.inputText.value.length;
        this.charCount.textContent = count;
        
        if (count > 8000) {
            this.charCount.classList.add('text-warning');
        } else if (count > 9500) {
            this.charCount.classList.add('text-danger');
        } else {
            this.charCount.classList.remove('text-warning', 'text-danger');
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) {
            this.currentFile = null;
            return;
        }

        const allowedExtensions = ['.docx', '.pdf'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        
        if (!allowedExtensions.includes(fileExtension)) {
            this.showError('Please select a .docx or .pdf file');
            this.fileInput.value = '';
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            this.fileInput.value = '';
            return;
        }

        this.currentFile = file;
        this.inputText.value = ''; // Clear text input when file is selected
        this.hideResults();
    }

    async correctGrammar(silent = false) {
        try {
            if (!silent) {
                this.showLoading();
            }

            let textToCorrect = '';
            let isFileUpload = false;

            if (this.currentFile) {
                // Handle file upload
                const formData = new FormData();
                formData.append('file', this.currentFile);
                
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Upload failed');
                }

                this.correctedData = result;
                isFileUpload = true;
                
            } else {
                // Handle text input
                textToCorrect = this.inputText.value.trim();
                
                if (!textToCorrect) {
                    // Don't throw error, just clear the results
                    this.clearResults();
                    return;
                }

                const response = await fetch('/correct', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: textToCorrect })
                });

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Correction failed');
                }

                this.correctedData = result;
            }

            this.displayResults(isFileUpload);

        } catch (error) {
            if (!silent) {
                this.showError(error.message);
            }
            console.error('Correction error:', error);
        } finally {
            if (!silent) {
                this.hideLoading();
            }
        }
    }

    displayResults(isFileUpload) {
        if (!this.correctedData) return;

        // Display original and corrected text
        this.originalText.textContent = this.correctedData.original;
        this.correctedText.textContent = this.correctedData.corrected;

        // Update error count
        const errorCount = this.correctedData.error_count || 0;
        this.errorCount.textContent = `${errorCount} error${errorCount !== 1 ? 's' : ''} found`;
        this.errorCount.className = errorCount > 0 ? 'badge bg-warning' : 'badge bg-success';

        // Display errors
        this.displayErrors(this.correctedData.errors || []);

        // Show results
        this.resultsSection.style.display = 'block';
        this.errorSection.style.display = 'none';
    }

    displayErrors(errors) {
        if (!errors || errors.length === 0) {
            this.errorsSection.style.display = 'none';
            return;
        }

        this.errorsSection.style.display = 'block';
        this.errorsList.innerHTML = '';

        errors.forEach((error, index) => {
            const errorItem = document.createElement('div');
            errorItem.className = 'alert alert-warning py-2 mb-2';
            
            let suggestionsText = '';
            let suggestionsButtons = '';
            if (error.suggestions && error.suggestions.length > 0) {
                suggestionsButtons = error.suggestions.map(suggestion => 
                    `<button class="btn btn-sm btn-outline-primary me-1 mb-1" onclick="this.closest('.alert').querySelector('.suggestion-text').textContent='${suggestion}'">${suggestion}</button>`
                ).join('');
            }

            const categoryBadge = error.category ? `<span class="badge bg-secondary me-2">${error.category}</span>` : '';

            errorItem.innerHTML = `
                <div class="d-flex align-items-start">
                    <span class="badge bg-warning me-2">${index + 1}</span>
                    <div class="flex-grow-1">
                        ${categoryBadge}
                        <strong style="color: #000000;">${error.message}</strong>
                        <br><small class="text-muted" style="color: #000000;">Context: "${error.context}"</small>
                        ${error.extended_context ? `<br><small class="text-muted" style="color: #000000;">Extended: "${error.extended_context}"</small>` : ''}
                        ${error.suggestions && error.suggestions.length > 0 ? `<br><small class="text-muted" style="color: #000000;">Suggestions: </small>${suggestionsButtons}<br><small class="text-info suggestion-text"></small>` : ''}
                    </div>
                </div>
            `;
            
            this.errorsList.appendChild(errorItem);
        });
    }

    async copyToClipboard() {
        if (!this.correctedData) return;

        try {
            await navigator.clipboard.writeText(this.correctedData.corrected);
            this.showSuccess('Corrected text copied to clipboard!');
        } catch (error) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = this.correctedData.corrected;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showSuccess('Corrected text copied to clipboard!');
        }
    }

    clearAll() {
        this.inputText.value = '';
        this.fileInput.value = '';
        this.currentFile = null;
        this.correctedData = null;
        this.updateCharCount();
        this.hideResults();
    }

    clearResults() {
        this.originalText.textContent = '';
        this.correctedText.textContent = '';
        this.errorCount.textContent = '0 errors found';
        this.errorCount.className = 'badge bg-success';
        this.errorsSection.style.display = 'none';
    }

    showLoading() {
        this.loadingSection.style.display = 'block';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
        this.correctBtn.disabled = true;
    }

    hideLoading() {
        this.loadingSection.style.display = 'none';
        this.correctBtn.disabled = false;
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorSection.style.display = 'block';
        this.resultsSection.style.display = 'none';
        this.hideLoading();
    }

    showSuccess(message) {
        this.successMessage.textContent = message;
        this.successToast.show();
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GrammarCorrectionApp();
});
