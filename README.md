# Grammar Corrector Application - Local Setup Instructions

## Overview
This Grammar Corrector is a web-based application that uses a hybrid approach combining rule-based checking with statistical validation using the FCE (First Certificate in English) corpus. It provides real-time grammar correction through a modern web interface.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (recommended 8GB)
- **Storage**: At least 1GB free space
- **Internet Connection**: Required for initial setup to download dependencies

### Required Software
- Python 3.8+ with pip
- Web browser (Chrome, Firefox, Safari, or Edge)
- Terminal/Command Prompt access

## Installation Steps

### 1. Download the Project
# Download and extract the ZIP file, then navigate to the folder

### 2. Verify Required Files
Ensure you have these essential files in your project directory:
```
GrammarCorrector/
├── app.py
├── models/
│   ├── __init__.py
│   └── grammar_corrector.py
├── fce/json/
│   └── fce.train.json          # Critical - 3.5MB training data
├── index.html
├── css/
│   └── style.css
├── js/
│   └── script.js
└── requirements.txt
```

### 3. Install Python Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Alternative: Using conda
conda install --file requirements.txt
```

### 4. Verify Installation
```bash
# Check if Python packages are installed correctly
python -c "import flask, nltk, language_tool_python; print('All packages installed successfully')"
```

## Running the Application

### 1. Start the Application
```bash
# Navigate to the project directory
cd /path/to/GrammarCorrector

# Run the Flask application
python app.py
```

### 2. Initial Setup (First Run)
On the first run, the application will:
- Download NLTK data packages (punkt, averaged_perceptron_tagger, stopwords)
- Download LanguageTool Java components
- Load and process the FCE training dataset (3.5MB)
- Build language models and error patterns

**Note**: This initial setup may take 2-5 minutes depending on your internet connection and system performance.

### 3. Application Startup Output
You should see output similar to:
```
Training model using FCE dataset...
Loading file: fce/json/fce.train.json
Loaded 2116 training examples
Built language model with 242932 trigrams
Extracted patterns for 1584 POS sequences
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5001
* Running on http://192.168.1.x:5001
```

## Using the Application

### 1. Access the Web Interface
Open your web browser and navigate to:
- **Local Access**: http://localhost:5001
- **Local Network**: http://127.0.0.1:5001

### 2. Grammar Correction Methods

#### Method 1: Direct Text Input
1. Type or paste text into the text area
2. Click "Correct Grammar" button
3. View results showing:
   - Original text with errors highlighted
   - Corrected text
   - List of grammar issues found

#### Method 2: File Upload
1. Click "Choose File" button
2. Select a `.docx` or `.pdf` file (max 16MB)
3. Click "Correct Grammar" button
4. View extracted text and corrections

### 3. Features
- **Real-time Grammar Checking**: Hybrid rule-based and statistical approach
- **File Support**: Upload Word documents (.docx) or PDF files (.pdf)
- **Error Analysis**: Detailed breakdown of grammar issues
- **Copy Functionality**: Easy copying of corrected text
- **Character Limit**: 10,000 characters for direct input, 50,000 for files

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use
```
Error: Address already in use
```
**Solution**: 
- Kill existing process: `pkill -f "python app.py"`
- Or change port in `app.py`: `app.run(host='0.0.0.0', port=5002, debug=True)`

#### 2. Missing FCE Training Data
```
Error: fce.train.json not found
```
**Solution**: 
- Ensure `fce/json/fce.train.json` exists in your project directory
- This file is critical (3.5MB) and contains training data

#### 3. NLTK Download Issues
```
LookupError: Resource not found
```
**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

#### 4. LanguageTool Java Issues
```
Error: Could not find or load main class
```
**Solution**:
- Ensure Java is installed: `java -version`
- Install Java if missing: https://java.com/download/
- Restart the application

#### 5. Memory Issues
```
MemoryError or slow performance
```
**Solution**:
- Ensure at least 4GB RAM available
- Close other applications
- Consider using a smaller text input

#### 6. File Upload Issues
```
File processing error
```
**Solution**:
- Ensure file is .docx or .pdf format
- Check file size (max 16MB)
- Verify file is not corrupted

### Performance Optimization

#### For Better Performance:
1. **Increase Memory**: Close unnecessary applications
2. **Use SSD Storage**: Faster file access for large datasets
3. **Stable Network**: Required for initial downloads
4. **Modern Browser**: Use updated Chrome, Firefox, or Safari

#### For Production Use:
1. **Use Production WSGI Server**: Replace Flask dev server with Gunicorn/uWSGI
2. **Add Caching**: Implement Redis for frequently corrected texts
3. **Load Balancing**: Use multiple instances for high traffic

## Development Mode

### Enable Debug Mode
The application runs in debug mode by default for development:
- **Auto-reload**: Changes to Python files trigger restart
- **Debug Info**: Detailed error messages in browser
- **Debug PIN**: Shown in console for interactive debugging

### Disable Debug Mode
For production-like testing:
```python
# In app.py, change:
app.run(host='0.0.0.0', port=5001, debug=False)
```

## Stopping the Application

### Normal Shutdown
- Press `Ctrl+C` in the terminal where the app is running

### Force Stop
```bash
# Find the process
ps aux | grep python

# Kill the process
kill -9 <process_id>

# Or kill all Python processes (use with caution)
pkill -f "python app.py"
```

## System Information

### Default Configuration
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 5001
- **Debug Mode**: Enabled
- **Max File Size**: 16MB
- **Max Text Length**: 10,000 characters (direct input)
- **Max File Text**: 50,000 characters

### File Structure
```
GrammarCorrector/
├── app.py                   # Main Flask application
├── models/                  # Grammar correction models
├── fce/json/                # FCE training dataset
├── css/                     # Web interface styles
├── js/                      # Frontend JavaScript
├── index.html               # Main web page
├── requirements.txt         # Python dependencies
└── INSTRUCTIONS.md          # This file
```

## Support

### Getting Help
1. **Check Console Output**: Look for error messages in terminal
2. **Browser Console**: Press F12 to check for JavaScript errors
3. **Log Files**: Check application logs for detailed error information
4. **System Resources**: Monitor CPU and memory usage

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Error messages from console
- Steps to reproduce the problem
- File types and sizes being processed

---

**Note**: This application is designed for educational and research purposes. For production use, additional security measures and optimizations should be implemented. 