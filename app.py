import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from docx import Document
from PyPDF2 import PdfReader
from models.grammar_corrector import GrammarCorrector

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create the app
app = Flask(__name__, static_url_path='', static_folder='.')
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'docx', 'pdf'}

# Initialize the grammar corrector
corrector = None
try:
    corrector = GrammarCorrector()
except Exception as e:
    logging.error(f"Failed to initialize grammar corrector: {e}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def correct_grammar(text):
    """Correct grammar in the given text using hybrid approach"""
    if not corrector:
        return {"error": "Grammar correction service is not available"}
    
    try:
        # Clean and prepare text
        text = text.strip()
        if not text:
            return {"error": "Empty text provided"}
        
        # Use hybrid corrector
        result = corrector.correct_text(text)
        return result
        
    except Exception as e:
        logging.error(f"Grammar correction error: {e}")
        return {"error": f"Grammar correction failed: {str(e)}"}

def extract_text_from_file(file):
    """Extract text from different file types"""
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.docx'):
            doc = Document(file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif filename.endswith('.pdf'):
            pdf = PdfReader(file)
            text = []
            for page in pdf.pages:
                text.append(page.extract_text())
            return '\n'.join(text)
        else:
            raise ValueError("Unsupported file format. Please upload a .docx or .pdf file.")
    except Exception as e:
        logging.error(f"Error extracting text from file: {e}")
        raise ValueError(f"Error reading file: {str(e)}")

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

@app.route('/css/<path:path>')
def send_css(path):
    """Serve CSS files"""
    return send_from_directory('css', path)

@app.route('/js/<path:path>')
def send_js(path):
    """Serve JavaScript files"""
    return send_from_directory('js', path)

@app.route('/correct', methods=['POST'])
def correct():
    """Handle text correction requests"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 10000:  # Limit text length
            return jsonify({'error': 'Text too long. Maximum 10,000 characters allowed.'}), 400
        
        result = corrector.correct_text(text)
        return jsonify(result)

    except Exception as e:
        logging.error(f"Text correction error: {e}")
        return jsonify({
            'error': str(e),
            'corrections': [],
            'corrected_text': text,
            'error_count': 0
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and correction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filename = file.filename
        if not filename or filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(filename):
            return jsonify({'error': 'Only .docx or .pdf files are allowed'}), 400
        
        # Extract text from file
        try:
            content = extract_text_from_file(file)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logging.error(f"File processing error: {e}")
            return jsonify({'error': 'Error processing file'}), 500
        
        if len(content) > 50000:  # Limit file size
            return jsonify({'error': 'File too large. Maximum 50,000 characters allowed.'}), 400
        
        if not content.strip():
            return jsonify({'error': 'File is empty'}), 400
        
        # Correct grammar
        result = correct_grammar(content)
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'original': result['original'],
            'corrected': result['corrected'],
            'errors': result['errors'],
            'error_count': result['error_count']
        })
    
    except Exception as e:
        logging.error(f"File upload error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return send_from_directory('.', 'index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logging.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
