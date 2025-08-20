import streamlit as st
import requests
import base64
import json
from io import BytesIO
from PIL import Image
import PyPDF2
import tempfile
import os
import sys
import subprocess
from typing import Optional, Dict, Any, List, Tuple
import traceback
import hashlib
import pickle
import sqlite3
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import markdown
import re

# Try importing pdf2image with error handling
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    st.warning("⚠️ pdf2image not available. PDF conversion will use fallback method.")

# Page configuration
st.set_page_config(
    page_title="🌪️ Typhoon OCR with RAG",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .model-card {
        background: linear-gradient(45deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .knowledge-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
EMBEDDING_API_URL = "http://209.15.123.47:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_API_URL = "http://209.15.123.47:11434/api/generate"

# Initialize session state for RAG
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_db_path" not in st.session_state:
    st.session_state.rag_db_path = "typhoon_rag_knowledge.db"

AVAILABLE_MODELS = {
    "scb10x/typhoon-ocr-7b:latest": {
        "name": "Typhoon OCR 7B",
        "description": "เชี่ยวชาญ OCR ไทย-อังกฤษ",
        "icon": "🌪️",
        "best_for": "OCR, Document parsing, Thai-English text"
    },
    "qwen2.5:14b": {
        "name": "Qwen2.5 14B", 
        "description": "โมเดลทั่วไปขนาดใหญ่",
        "icon": "🧠",
        "best_for": "General purpose, Complex reasoning"
    },
    "scb10x/llama3.1-typhoon2-8b-instruct:latest": {
        "name": "Typhoon2 8B",
        "description": "โมเดลภาษาไทยล่าสุด", 
        "icon": "🇹🇭",
        "best_for": "Thai language, Instructions following"
    }
}

# ==================== RAG SYSTEM FUNCTIONS ====================

class RAGKnowledgeBase:
    def __init__(self, db_path: str = "typhoon_rag_knowledge.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for knowledge storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                chunk_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama API"""
        try:
            response = requests.post(
                EMBEDDING_API_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get('embedding', [])
        except Exception as e:
            st.error(f"❌ Error getting embedding: {str(e)}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_ends = ['. ', '! ', '? ', '\n\n', '。', '！', '？']
                best_break = end
                
                for i in range(min(100, len(text) - end)):
                    for ending in sentence_ends:
                        if text[end + i:end + i + len(ending)] == ending:
                            best_break = end + i + len(ending)
                            break
                    if best_break != end:
                        break
                
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def add_document(self, filename: str, content: str, metadata: Dict = None) -> bool:
        """Add document to knowledge base"""
        try:
            # Create file hash for deduplication
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if document already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone():
                conn.close()
                st.warning(f"⚠️ Document {filename} already exists in knowledge base")
                return False
            
            # Process markdown content
            if filename.endswith('.md'):
                # Convert markdown to plain text for better chunking
                html = markdown.markdown(content)
                # Remove HTML tags
                plain_text = re.sub('<[^<]+?>', '', html)
                content = plain_text
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
                progress_bar.progress((i + 1) / len(chunks))
                
                # Get embedding
                embedding = self.get_embedding(chunk)
                if embedding:
                    # Store in database
                    cursor.execute('''
                        INSERT INTO documents 
                        (filename, content, embedding, chunk_id, file_hash, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        filename, 
                        chunk, 
                        pickle.dumps(embedding), 
                        i,
                        file_hash,
                        json.dumps(metadata or {})
                    ))
            
            conn.commit()
            conn.close()
            
            status_text.text(f"✅ Added {len(chunks)} chunks from {filename}")
            progress_bar.progress(1.0)
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding document: {str(e)}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar content"""
        try:
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT filename, content, embedding FROM documents")
            
            results = []
            for filename, content, embedding_blob in cursor.fetchall():
                stored_embedding = pickle.loads(embedding_blob)
                similarity = cosine_similarity(
                    [query_embedding], 
                    [stored_embedding]
                )[0][0]
                results.append((filename, content, similarity))
            
            conn.close()
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"❌ Error searching: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chat_sessions")
            total_chats = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_chat_sessions": total_chats
            }
        except Exception as e:
            return {"error": str(e)}

def generate_rag_response(query: str, context_docs: List[Tuple[str, str, float]], model: str = "qwen2.5:14b") -> Optional[str]:
    """Generate response using RAG context"""
    try:
        # Prepare context
        context_text = "\n\n".join([
            f"จากเอกสาร '{doc[0]}':\n{doc[1]}" 
            for doc in context_docs[:3]  # Use top 3 most relevant
        ])
        
        # Create prompt
        prompt = f"""คุณเป็น AI Assistant ที่ช่วยตอบคำถามจากเอกสารที่ให้มา ใช้ข้อมูลจากบริบทต่อไปนี้ในการตอบคำถาม:

บริบทจากเอกสาร:
{context_text}

คำถาม: {query}

กรุณาตอบคำถามโดยอ้างอิงข้อมูลจากเอกสารที่ให้มา หากไม่มีข้อมูลที่เกี่ยวข้องในเอกสาร ให้บอกว่าไม่พบข้อมูลที่เกี่ยวข้อง

คำตอบ:"""
        
        # Call API
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.3,
                "top_p": 0.8,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
        
    except Exception as e:
        st.error(f"❌ Error generating RAG response: {str(e)}")
        return None

# ==================== ORIGINAL OCR FUNCTIONS ====================

def check_system_dependencies():
    """Check if system dependencies are available"""
    dependencies = {
        'poppler': False,
        'pdf2image': PDF2IMAGE_AVAILABLE,
        'pymupdf_available': False
    }
    
    # Check for poppler-utils
    try:
        result = subprocess.run(['pdftoppm', '--help'], 
                              capture_output=True, text=True, timeout=5)
        dependencies['poppler'] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        dependencies['poppler'] = False
    
    # Check for PyMuPDF as alternative
    try:
        import fitz  # PyMuPDF
        dependencies['pymupdf_available'] = True
    except ImportError:
        dependencies['pymupdf_available'] = False
    
    return dependencies

def convert_pdf_to_images_pypdf2(pdf_file, quality="medium") -> List[Image.Image]:
    """Fallback: Convert PDF to images using PyPDF2 (text extraction only)"""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        images = []
        
        # Create a simple text image for each page
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    # Create a simple white image with text information
                    img = Image.new('RGB', (800, 1000), color='white')
                    images.append(img)
                else:
                    # Empty page
                    img = Image.new('RGB', (800, 1000), color='white')
                    images.append(img)
            except Exception as e:
                st.warning(f"⚠️ Error processing page {page_num + 1}: {str(e)}")
                # Create placeholder image
                img = Image.new('RGB', (800, 1000), color='white')
                images.append(img)
        
        return images
    except Exception as e:
        st.error(f"❌ Error using PyPDF2 fallback: {str(e)}")
        return []

def convert_pdf_with_pymupdf(pdf_file, quality="medium") -> List[Image.Image]:
    """Convert PDF using PyMuPDF (fitz)"""
    try:
        import fitz
        
        # Set DPI based on quality
        dpi = {"high": 300, "medium": 200, "low": 150}.get(quality, 200)
        
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        # Open PDF from bytes
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc[page_num]
                
                # Create matrix for scaling
                zoom = dpi / 72.0  # 72 DPI is default
                matrix = fitz.Matrix(zoom, zoom)
                
                # Render page to image
                pix = page.get_pixmap(matrix=matrix)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(BytesIO(img_data))
                images.append(img)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing page {page_num + 1}: {str(e)}")
                # Create placeholder image
                img = Image.new('RGB', (800, 1000), color='white')
                images.append(img)
        
        pdf_doc.close()
        return images
        
    except ImportError:
        st.error("❌ PyMuPDF (fitz) not available")
        return []
    except Exception as e:
        st.error(f"❌ Error using PyMuPDF: {str(e)}")
        return []

def convert_pdf_to_images(pdf_file, quality="medium") -> List[Image.Image]:
    """Convert PDF to images with multiple fallback methods"""
    
    # Method 1: Try pdf2image with poppler
    if PDF2IMAGE_AVAILABLE:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                pdf_file.seek(0)
                tmp_file.write(pdf_file.read())
                tmp_file.flush()
                
                # Set DPI based on quality
                dpi = {"high": 300, "medium": 200, "low": 150}.get(quality, 200)
                
                # Try to convert
                images = pdf2image.convert_from_path(tmp_file.name, dpi=dpi)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                if images:
                    st.success(f"✅ PDF converted successfully using pdf2image ({len(images)} pages)")
                    return images
                    
        except Exception as e:
            st.warning(f"⚠️ pdf2image failed: {str(e)}")
            # Clean up temp file if it exists
            try:
                if 'tmp_file' in locals():
                    os.unlink(tmp_file.name)
            except:
                pass
    
    # Method 2: Try PyMuPDF
    st.info("🔄 Trying alternative PDF converter (PyMuPDF)...")
    images = convert_pdf_with_pymupdf(pdf_file, quality)
    if images:
        st.success(f"✅ PDF converted using PyMuPDF ({len(images)} pages)")
        return images
    
    # Method 3: Fallback to PyPDF2 (text extraction)
    st.warning("⚠️ Using fallback method (PyPDF2) - limited functionality")
    images = convert_pdf_to_images_pypdf2(pdf_file, quality)
    if images:
        st.info(f"ℹ️ PDF processed using PyPDF2 fallback ({len(images)} pages)")
        return images
    
    # If all methods fail
    st.error("❌ All PDF conversion methods failed!")
    return []

def extract_text_from_pdf_page(pdf_file, page_num: int) -> str:
    """Extract text from specific PDF page using PyPDF2"""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if page_num < len(pdf_reader.pages):
            page = pdf_reader.pages[page_num]
            return page.extract_text()
        return ""
    except Exception as e:
        st.warning(f"⚠️ Text extraction failed for page {page_num + 1}: {str(e)}")
        return ""

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    try:
        buffer = BytesIO()
        # Optimize image size
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 2048 pixels)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        image.save(buffer, format='PNG', optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        st.error(f"❌ Error converting image to base64: {str(e)}")
        return ""

def get_ocr_prompt(prompt_type: str, raw_text: str = "") -> str:
    """Get OCR prompt based on type"""
    if prompt_type == "structure":
        return f"""Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. 
Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.
Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.
If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.
Your final output must be in JSON format with a single key `natural_text` containing the response.
RAW_TEXT_START
{raw_text}
RAW_TEXT_END"""
    else:
        return f"""Below is an image of a document page along with its dimensions. 
Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.
If the document contains images, use a placeholder like dummy.png for each image.
Your final output must be in JSON format with a single key `natural_text` containing the response.
RAW_TEXT_START
{raw_text}
RAW_TEXT_END"""

def call_ollama_api(model: str, prompt: str, image_base64: str, params: dict) -> Optional[str]:
    """Call Ollama API for OCR processing with error handling"""
    try:
        # Validate inputs
        if not image_base64:
            raise ValueError("Empty image data")
        
        # For OCR models, we need to use chat completion format
        if "typhoon-ocr" in model:
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "temperature": params.get('temperature', 0.1),
                "top_p": params.get('top_p', 0.6),
                "num_predict": params.get('max_tokens', 12000),
                "stream": False
            }
        else:
            # For other models, use standard format
            payload = {
                "model": model,
                "prompt": f"Analyze this image and extract all text content. Return the result in markdown format.\n\nImage: data:image/png;base64,{image_base64}\n\n{prompt}",
                "temperature": params.get('temperature', 0.1),
                "top_p": params.get('top_p', 0.6),
                "num_predict": params.get('max_tokens', 12000),
                "stream": False
            }
        
        # Add timeout and retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OLLAMA_API_URL, 
                    json=payload, 
                    timeout=300,  # 5 minutes timeout
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result.get('response', '')
                
                if content.strip():
                    return content
                else:
                    raise ValueError("Empty response from API")
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Timeout on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    raise
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Connection error on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    raise
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API Request Error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - HTTP {e.response.status_code}"
        st.error(f"❌ {error_msg}")
        return None
    except ValueError as e:
        st.error(f"❌ Validation Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")
        st.error(f"Debug: {traceback.format_exc()}")
        return None

def process_single_document(file, model: str, params: dict) -> Dict[str, Any]:
    """Process a single document with comprehensive error handling"""
    results = {
        'filename': file.name,
        'file_type': file.type,
        'file_size': file.size,
        'pages': [],
        'total_pages': 0,
        'success': False,
        'error': None,
        'warnings': []
    }
    
    try:
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size > max_size:
            results['error'] = f"File too large ({file.size/1024/1024:.1f}MB). Maximum size is 10MB."
            return results
        
        # Handle different file types
        if file.type == 'application/pdf':
            st.info(f"📄 Processing PDF: {file.name}")
            images = convert_pdf_to_images(file, params.get('image_quality', 'medium'))
            results['total_pages'] = len(images) if images else 0
            
            if not images:
                results['error'] = "Could not convert PDF to images"
                return results
                
        elif file.type.startswith('image/'):
            st.info(f"🖼️ Processing Image: {file.name}")
            try:
                image = Image.open(file)
                images = [image]
                results['total_pages'] = 1
            except Exception as e:
                results['error'] = f"Could not open image file: {str(e)}"
                return results
        else:
            results['error'] = f"Unsupported file type: {file.type}"
            return results
        
        # Process each page/image
        for i, image in enumerate(images):
            page_result = {
                'page_number': i + 1,
                'success': False,
                'content': '',
                'raw_response': '',
                'extracted_text': '',
                'processing_time': 0,
                'error': None
            }
            
            try:
                import time
                start_time = time.time()
                
                # For PDF pages, try to extract text first
                if file.type == 'application/pdf':
                    try:
                        page_result['extracted_text'] = extract_text_from_pdf_page(file, i)
                    except:
                        pass
                
                # Convert image to base64
                image_base64 = image_to_base64(image)
                if not image_base64:
                    page_result['error'] = "Failed to convert image to base64"
                    continue
                
                # Get appropriate prompt
                prompt = get_ocr_prompt(
                    params.get('prompt_type', 'default'),
                    page_result['extracted_text']
                )
                
                # Show processing status
                with st.spinner(f"🔄 Processing page {i + 1}..."):
                    # Call API
                    response = call_ollama_api(model, prompt, image_base64, params)
                
                if response:
                    page_result['raw_response'] = response
                    page_result['processing_time'] = time.time() - start_time
                    
                    # Try to parse JSON response
                    try:
                        json_response = json.loads(response)
                        page_result['content'] = json_response.get('natural_text', response)
                    except json.JSONDecodeError:
                        page_result['content'] = response
                    
                    # Validate content
                    if page_result['content'].strip():
                        page_result['success'] = True
                        st.success(f"✅ Page {i + 1} processed successfully ({page_result['processing_time']:.1f}s)")
                    else:
                        page_result['error'] = "Empty content returned"
                        st.warning(f"⚠️ Page {i + 1} returned empty content")
                else:
                    page_result['error'] = "No response from API"
                    st.error(f"❌ Page {i + 1} failed - no response")
                
            except Exception as e:
                page_result['error'] = str(e)
                page_result['processing_time'] = time.time() - start_time if 'start_time' in locals() else 0
                st.error(f"❌ Page {i + 1} failed: {str(e)}")
            
            results['pages'].append(page_result)
        
        # Check if any pages were successful
        successful_pages = [p for p in results['pages'] if p['success']]
        results['success'] = len(successful_pages) > 0
        
        if results['success']:
            results['summary'] = {
                'successful_pages': len(successful_pages),
                'failed_pages': len(results['pages']) - len(successful_pages),
                'total_processing_time': sum(p['processing_time'] for p in results['pages']),
                'average_time_per_page': sum(p['processing_time'] for p in results['pages']) / len(results['pages']) if results['pages'] else 0
            }
        
    except Exception as e:
        results['error'] = f"Document processing failed: {str(e)}"
        st.error(f"❌ Fatal error processing {file.name}: {str(e)}")
    
    return results

def process_documents(uploaded_files, model: str, params: dict):
    """Process uploaded documents with enhanced error handling and progress tracking"""
    
    # Handle single file vs multiple files
    files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    
    st.header("🔄 Processing Results")
    
    # Show system status first
    deps = check_system_dependencies()
    
    with st.expander("🔧 System Status", expanded=False):
        st.write("**Dependencies Status:**")
        status_items = [
            ("Poppler (PDF2Image)", "✅" if deps['poppler'] else "❌"),
            ("pdf2image Library", "✅" if deps['pdf2image'] else "❌"),
            ("PyMuPDF Alternative", "✅" if deps['pymupdf_available'] else "❌"),
            ("PyPDF2 Fallback", "✅")  # Always available
        ]
        
        for item, status in status_items:
            st.write(f"{status} {item}")
        
        if not deps['poppler'] and not deps['pymupdf_available']:
            st.warning("⚠️ Limited PDF processing capabilities. Consider installing poppler or PyMuPDF.")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    total_files = len(files_to_process)
    
    for i, file in enumerate(files_to_process):
        status_text.text(f"📄 Processing {file.name} ({i+1}/{total_files})...")
        progress_bar.progress(i / total_files)
        
        # Reset file pointer
        file.seek(0)
        
        # Process document
        result = process_single_document(file, model, params)
        all_results.append(result)
        
        # Display results for this file
        status_emoji = "✅" if result['success'] else "❌"
        summary_text = ""
        
        if result['success'] and 'summary' in result:
            s = result['summary']
            summary_text = f" - {s['successful_pages']}/{result['total_pages']} pages, {s['total_processing_time']:.1f}s total"
        
        with st.expander(f"{status_emoji} {file.name}{summary_text}", expanded=result['success']):
            if result['success']:
                # Show summary
                if 'summary' in result:
                    s = result['summary']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("✅ Success", f"{s['successful_pages']}/{result['total_pages']}")
                    with col2:
                        st.metric("⏱️ Total Time", f"{s['total_processing_time']:.1f}s")
                    with col3:
                        st.metric("📊 Avg/Page", f"{s['average_time_per_page']:.1f}s")
                    with col4:
                        st.metric("📄 File Size", f"{result['file_size']/1024:.1f}KB")
                
                # RAG Integration - Auto-add to knowledge base option
                if st.button(f"🧠 Add to Knowledge Base", key=f"add_kb_{i}"):
                    # Combine all successful page content
                    combined_content = "\n\n".join([
                        page['content'] for page in result['pages'] if page['success']
                    ])
                    
                    # Add to RAG knowledge base
                    kb = RAGKnowledgeBase(st.session_state.rag_db_path)
                    success = kb.add_document(
                        filename=file.name,
                        content=combined_content,
                        metadata={
                            "processing_time": s.get('total_processing_time', 0),
                            "model_used": model,
                            "successful_pages": s.get('successful_pages', 0)
                        }
                    )
                    if success:
                        st.success(f"✅ Added {file.name} to knowledge base!")
                        st.rerun()
                
                # Show results for each page
                for page in result['pages']:
                    if page['success']:
                        st.subheader(f"📄 Page {page['page_number']}")
                        
                        # Tabs for different views
                        tab1, tab2, tab3, tab4 = st.tabs(["📖 Preview", "📝 Markdown", "🔧 Raw", "ℹ️ Info"])
                        
                        with tab1:
                            if params.get('output_format') == 'html':
                                st.markdown(page['content'], unsafe_allow_html=True)
                            else:
                                st.markdown(page['content'])
                        
                        with tab2:
                            st.code(page['content'], language='markdown')
                        
                        with tab3:
                            st.code(page['raw_response'], language='json')
                        
                        with tab4:
                            st.write(f"**Processing Time:** {page['processing_time']:.1f}s")
                            st.write(f"**Content Length:** {len(page['content'])} characters")
                            if page['extracted_text']:
                                st.write(f"**Extracted Text Length:** {len(page['extracted_text'])} characters")
                            
                        # Download button
                        file_extension = {
                            'markdown': 'md',
                            'html': 'html', 
                            'json': 'json'
                        }.get(params.get('output_format', 'markdown'), 'md')
                        
                        st.download_button(
                            f"💾 Download Page {page['page_number']}",
                            page['content'],
                            f"{file.name}_page_{page['page_number']}.{file_extension}",
                            f"text/{file_extension}"
                        )
                    else:
                        st.error(f"❌ Page {page['page_number']} failed: {page.get('error', 'Unknown error')}")
                        if page.get('processing_time', 0) > 0:
                            st.write(f"Processing time: {page['processing_time']:.1f}s")
            else:
                st.error(f"❌ Failed to process {file.name}")
                if result.get('error'):
                    st.write(f"**Error:** {result['error']}")
                
                # Show system recommendations
                if 'pdf' in file.type.lower() and not deps['poppler'] and not deps['pymupdf_available']:
                    st.info("""
                    **💡 Recommendation for PDF processing:**
                    
                    For better PDF support, install one of these:
                    
                    **Option 1: Install Poppler (Recommended)**
                    ```bash
                    # Ubuntu/Debian
                    apt-get install poppler-utils
                    
                    # CentOS/RHEL
                    yum install poppler-utils
                    
                    # macOS
                    brew install poppler
                    ```
                    
                    **Option 2: Install PyMuPDF**
                    ```bash
                    pip install PyMuPDF
                    ```
                    """)
    
    # Final progress
    progress_bar.progress(1.0)
    status_text.text("🎉 Processing complete!")
    
    # Summary statistics
    successful_files = sum(1 for r in all_results if r['success'])
    total_pages = sum(r['total_pages'] for r in all_results)
    successful_pages = sum(len([p for p in r['pages'] if p['success']]) for r in all_results)
    total_processing_time = sum(
        r.get('summary', {}).get('total_processing_time', 0) for r in all_results if r['success']
    )
    
    # Display final summary
    st.markdown("---")
    st.subheader("📊 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Files", f"{successful_files}/{len(files_to_process)}")
    with col2:
        st.metric("📄 Pages", f"{successful_pages}/{total_pages}")
    with col3:
        st.metric("⏱️ Total Time", f"{total_processing_time:.1f}s")
    with col4:
        st.metric("🤖 Model", AVAILABLE_MODELS[model]['name'])
    
    # Show processing efficiency
    if successful_pages > 0:
        avg_time_per_page = total_processing_time / successful_pages
        pages_per_minute = 60 / avg_time_per_page if avg_time_per_page > 0 else 0
        
        st.info(f"""
        **⚡ Performance Metrics:**
        - Average processing time: **{avg_time_per_page:.1f} seconds per page**
        - Processing rate: **{pages_per_minute:.1f} pages per minute**
        - Success rate: **{(successful_pages/total_pages)*100:.1f}%**
        """)
    
    # Bulk download for successful results
    if successful_pages > 0:
        # Combine all successful content
        combined_content = f"# Typhoon OCR Results\n\n"
        combined_content += f"**Processing Summary:**\n"
        combined_content += f"- Files processed: {successful_files}/{len(files_to_process)}\n"
        combined_content += f"- Pages processed: {successful_pages}/{total_pages}\n"
        combined_content += f"- Model used: {AVAILABLE_MODELS[model]['name']}\n"
        combined_content += f"- Total processing time: {total_processing_time:.1f}s\n\n"
        combined_content += f"---\n\n"
        
        for result in all_results:
            if result['success']:
                combined_content += f"# 📄 {result['filename']}\n\n"
                if 'summary' in result:
                    s = result['summary']
                    combined_content += f"**File Summary:**\n"
                    combined_content += f"- Pages: {s['successful_pages']}/{result['total_pages']}\n"
                    combined_content += f"- Processing time: {s['total_processing_time']:.1f}s\n\n"
                
                for page in result['pages']:
                    if page['success']:
                        combined_content += f"## Page {page['page_number']}\n\n"
                        combined_content += f"{page['content']}\n\n"
                        combined_content += f"---\n\n"
        
        # File extension based on output format
        file_extension = {
            'markdown': 'md',
            'html': 'html', 
            'json': 'json'
        }.get(params.get('output_format', 'markdown'), 'md')
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "📦 Download All Results",
                combined_content,
                f"typhoon_ocr_results.{file_extension}",
                f"text/{file_extension}",
                help=f"Download combined results in {params.get('output_format', 'markdown')} format"
            )
        
        with col2:
            if st.button("🧠 Add All to Knowledge Base"):
                kb = RAGKnowledgeBase(st.session_state.rag_db_path)
                added_count = 0
                for result in all_results:
                    if result['success']:
                        combined_content = "\n\n".join([
                            page['content'] for page in result['pages'] if page['success']
                        ])
                        if kb.add_document(
                            filename=result['filename'],
                            content=combined_content,
                            metadata={"batch_processed": True}
                        ):
                            added_count += 1
                
                if added_count > 0:
                    st.success(f"✅ Added {added_count} documents to knowledge base!")
                    st.rerun()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌪️ Typhoon OCR with RAG</h1>
        <p>AI-Powered Thai-English Document Parser + Knowledge Q&A</p>
        <p>Powered by SCB 10X with AI NT North Team</p>
    </div>
    """, unsafe_allow_html=True)

    # System status check
    deps = check_system_dependencies()
    
    # Show system warnings if needed
    if not deps['poppler'] and not deps['pymupdf_available']:
        st.markdown("""
        <div class="warning-message">
            <strong>⚠️ Limited PDF Processing Capability</strong><br>
            For optimal PDF processing, please install poppler-utils or PyMuPDF.<br>
            Currently using PyPDF2 fallback which has limited functionality.
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System status in sidebar
        with st.expander("🔧 System Status"):
            st.write("**PDF Processing:**")
            if deps['poppler'] and deps['pdf2image']:
                st.success("✅ Full PDF support (pdf2image + poppler)")
            elif deps['pymupdf_available']:
                st.info("ℹ️ Good PDF support (PyMuPDF)")
            else:
                st.warning("⚠️ Basic PDF support (PyPDF2 only)")
            
        # RAG Knowledge Base Status
        kb = RAGKnowledgeBase(st.session_state.rag_db_path)
        stats = kb.get_stats()
        
        st.header("🧠 Knowledge Base")
        if not stats.get("error"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📚 Documents", stats["total_documents"])
            with col2:
                st.metric("📄 Chunks", stats["total_chunks"])
            st.metric("💬 Chat Sessions", stats["total_chat_sessions"])
        else:
            st.error("❌ Knowledge base error")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose AI Model:",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: f"{AVAILABLE_MODELS[x]['icon']} {AVAILABLE_MODELS[x]['name']}"
        )
        
        # Display model info
        model_info = AVAILABLE_MODELS[selected_model]
        st.info(f"**{model_info['name']}**\n\n{model_info['description']}\n\n**Best for:** {model_info['best_for']}")
        
        # Processing parameters
        st.subheader("🎯 Processing Parameters")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, help="Lower = more accurate, Higher = more creative")
        top_p = st.slider("Top P", 0.0, 1.0, 0.6, 0.1, help="Controls randomness of word selection")
        max_tokens = st.slider("Max Tokens", 1000, 16384, 12000, 500, help="Maximum length of generated text")
        
        # OCR specific settings
        st.subheader("📄 OCR Settings")
        prompt_type = st.selectbox(
            "Prompt Type:",
            ["default", "structure"],
            format_func=lambda x: "Default (Simple)" if x == "default" else "Structure (Complex Documents)",
            help="Structure mode better for tables, charts, and complex layouts"
        )
        
        output_format = st.selectbox(
            "Output Format:",
            ["markdown", "html", "json"],
            help="Choose output format - HTML best for tables, JSON for data processing"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            repetition_penalty = st.slider(
                "Repetition Penalty", 
                1.0, 2.0, 1.2, 0.1,
                help="Higher values reduce repetitive text"
            )
            image_quality = st.selectbox(
                "Image Quality", 
                ["high", "medium", "low"],
                index=1,
                help="Higher quality = better accuracy but slower processing"
            )
            batch_processing = st.checkbox(
                "Enable Batch Processing",
                help="Process multiple files at once"
            )
            
            # File size limit info
            st.info("📏 **File Limits:**\n- Max file size: 10MB\n- Supported: PDF, PNG, JPG, JPEG")

    # ===== Main content area with Tabs =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload & OCR", 
        "🧠 Knowledge Base", 
        "💬 AI Chat", 
        "✨ Features", 
        "📖 คู่มือการใช้งาน"
    ])

    # ---- Tab 1: Upload & Process (Original OCR) ----
    with tab1:
        st.header("📁 Upload Document for OCR")

        # File upload
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=batch_processing,
            help="Support: PDF, PNG, JPG, JPEG (Max: 10MB per file)"
        )
        
        # Preview uploaded files
        if uploaded_files:
            if isinstance(uploaded_files, list):
                total_size = sum(file.size for file in uploaded_files)
                st.success(f"✅ {len(uploaded_files)} files uploaded ({total_size/1024/1024:.1f}MB total)")
                
                # Show file details
                for i, file in enumerate(uploaded_files[:5]):  # Show first 5
                    size_mb = file.size / 1024 / 1024
                    st.write(f"{i+1}. **{file.name}** ({size_mb:.1f}MB, {file.type})")
                    
                if len(uploaded_files) > 5:
                    st.write(f"... and {len(uploaded_files) - 5} more files")
                    
            else:
                size_mb = uploaded_files.size / 1024 / 1024
                st.success(f"✅ File uploaded: **{uploaded_files.name}** ({size_mb:.1f}MB)")
                
                # Show preview for single file
                if uploaded_files.type.startswith('image'):
                    try:
                        image = Image.open(uploaded_files)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"❌ Cannot preview image: {str(e)}")
                elif uploaded_files.type == 'application/pdf':
                    st.info("📄 PDF file uploaded - preview will be shown during processing")
        
        # Process button with validation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Process Document(s)", type="primary", disabled=not uploaded_files):
                if uploaded_files:
                    # Validate total processing load
                    files_to_check = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                    total_size = sum(file.size for file in files_to_check)
                    
                    # Warn for large processing jobs
                    if total_size > 50 * 1024 * 1024:  # 50MB
                        st.warning("⚠️ Large file size detected. Processing may take several minutes.")
                    
                    if len(files_to_check) > 5:
                        st.warning("⚠️ Processing multiple files. This may take some time.")
                    
                    # Start processing
                    try:
                        process_documents(uploaded_files, selected_model, {
                            'temperature': temperature,
                            'top_p': top_p,
                            'max_tokens': max_tokens,
                            'repetition_penalty': repetition_penalty,
                            'prompt_type': prompt_type,
                            'output_format': output_format,
                            'image_quality': image_quality
                        })
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.error("Please check the error logs above and try again.")
                else:
                    st.error("❌ Please upload at least one file!")
        
        with col2:
            if st.button("🔄 Clear Files"):
                st.rerun()

    # ---- Tab 2: Knowledge Base Management ----
    with tab2:
        st.header("🧠 Knowledge Base Management")
        
        kb = RAGKnowledgeBase(st.session_state.rag_db_path)
        stats = kb.get_stats()
        
        # Display stats
        if not stats.get("error"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Total Documents", stats["total_documents"])
            with col2:
                st.metric("📄 Total Chunks", stats["total_chunks"])
            with col3:
                st.metric("💬 Chat Sessions", stats["total_chat_sessions"])
        
        st.markdown("---")
        
        # Add documents manually
        st.subheader("📤 Add Documents to Knowledge Base")
        
        # Method 1: Upload Markdown files
        st.write("**Method 1: Upload Markdown Files**")
        markdown_files = st.file_uploader(
            "Upload Markdown files",
            type=['md', 'txt'],
            accept_multiple_files=True,
            help="Upload .md or .txt files to add to knowledge base"
        )
        
        if markdown_files and st.button("➕ Add Markdown Files"):
            added_count = 0
            for file in markdown_files:
                try:
                    content = file.read().decode('utf-8')
                    if kb.add_document(
                        filename=file.name,
                        content=content,
                        metadata={"source": "manual_upload", "file_type": "markdown"}
                    ):
                        added_count += 1
                except Exception as e:
                    st.error(f"❌ Error adding {file.name}: {str(e)}")
            
            if added_count > 0:
                st.success(f"✅ Added {added_count} markdown files to knowledge base!")
                st.rerun()
        
        # Method 2: Direct text input
        st.write("**Method 2: Direct Text Input**")
        with st.form("add_text_form"):
            doc_title = st.text_input("Document Title", placeholder="Enter document title")
            doc_content = st.text_area(
                "Document Content",
                height=200,
                placeholder="Paste or type your document content here..."
            )
            
            submitted = st.form_submit_button("➕ Add to Knowledge Base")
            if submitted and doc_title and doc_content:
                if kb.add_document(
                    filename=f"{doc_title}.md",
                    content=doc_content,
                    metadata={"source": "manual_input"}
                ):
                    st.success(f"✅ Added '{doc_title}' to knowledge base!")
                    st.rerun()
        
        st.markdown("---")
        
        # Knowledge base browser
        st.subheader("📖 Browse Knowledge Base")
        
        if stats.get("total_documents", 0) > 0:
            # Get document list
            conn = sqlite3.connect(st.session_state.rag_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT filename, COUNT(*) as chunk_count, MAX(created_at) as latest FROM documents GROUP BY filename ORDER BY latest DESC")
            docs = cursor.fetchall()
            conn.close()
            
            # Display documents
            for filename, chunk_count, created_at in docs:
                with st.expander(f"📄 {filename} ({chunk_count} chunks)"):
                    st.write(f"**Created:** {created_at}")
                    st.write(f"**Chunks:** {chunk_count}")
                    
                    # Show first chunk as preview
                    conn = sqlite3.connect(st.session_state.rag_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT content FROM documents WHERE filename = ? LIMIT 1", (filename,))
                    preview = cursor.fetchone()
                    if preview:
                        st.write("**Preview:**")
                        st.text(preview[0][:200] + "..." if len(preview[0]) > 200 else preview[0])
                    conn.close()
                    
                    # Delete button
                    if st.button(f"🗑️ Delete {filename}", key=f"del_{filename}"):
                        conn = sqlite3.connect(st.session_state.rag_db_path)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
                        conn.commit()
                        conn.close()
                        st.success(f"✅ Deleted {filename}")
                        st.rerun()
        else:
            st.info("📝 No documents in knowledge base yet. Add some documents to get started!")

    # ---- Tab 3: AI Chat with RAG ----
    with tab3:
        st.header("💬 AI Chat with Knowledge Base")
        
        # Check if knowledge base has content
        if stats.get("total_documents", 0) == 0:
            st.warning("⚠️ No documents in knowledge base. Please add documents in the Knowledge Base tab first.")
            return
        
        # Chat interface
        st.subheader("🗣️ Ask Questions About Your Documents")
        
        # Display chat history
        for i, (question, answer, context) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🤔 You:</strong><br>
                {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Show context sources
            if context:
                with st.expander(f"📚 Sources (Chat {i+1})", expanded=False):
                    for j, (filename, content, similarity) in enumerate(context):
                        st.markdown(f"""
                        <div class="knowledge-card">
                            <strong>📄 {filename}</strong> (Similarity: {similarity:.3f})<br>
                            <em>{content[:200]}...</em>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question:",
                placeholder="What would you like to know about your documents?",
                key="chat_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                ask_button = st.form_submit_button("🚀 Ask")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and user_question:
            with st.spinner("🔍 Searching knowledge base and generating response..."):
                # Search for relevant context
                context_docs = kb.search_similar(user_question, top_k=5)
                
                if context_docs:
                    # Generate RAG response
                    response = generate_rag_response(
                        user_question, 
                        context_docs, 
                        selected_model
                    )
                    
                    if response:
                        # Add to chat history
                        st.session_state.chat_history.append((
                            user_question,
                            response,
                            context_docs[:3]  # Store top 3 for context
                        ))
                        
                        # Save to database
                        conn = sqlite3.connect(st.session_state.rag_db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO chat_sessions (session_id, question, answer, context)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            "default",  # Simple session management
                            user_question,
                            response,
                            json.dumps([(doc[0], doc[1][:200], doc[2]) for doc in context_docs[:3]])
                        ))
                        conn.commit()
                        conn.close()
                        
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate response")
                else:
                    st.warning("⚠️ No relevant information found in knowledge base")

    # ---- Tab 4: Features ----
    with tab4:
        st.header("✨ Enhanced Features with RAG")
        
        # Enhanced feature cards
        features = [
            {
                "icon": "🌪️",
                "title": "Advanced OCR Processing",
                "description": "ประมวลผลเอกสารด้วย AI ที่ทันสมัย",
                "items": [
                    "Thai-English OCR with high accuracy",
                    "Complex document structure recognition", 
                    "Batch processing for multiple files",
                    "Multiple output formats (MD, HTML, JSON)"
                ]
            },
            {
                "icon": "🧠", 
                "title": "RAG Knowledge Base",
                "description": "สร้างฐานความรู้และค้นหาข้อมูลอัจฉริยะ",
                "items": [
                    "Auto-import OCR results to knowledge base",
                    "Manual markdown document upload",
                    "Intelligent text chunking and embedding",
                    "Semantic search with similarity scoring"
                ]
            },
            {
                "icon": "💬",
                "title": "AI-Powered Q&A",
                "description": "ถามตอบกับเอกสารด้วย AI",
                "items": [
                    "Context-aware responses from your documents",
                    "Multi-language support (Thai-English)",
                    "Source attribution and transparency",
                    "Chat history and session management"
                ]
            },
            {
                "icon": "📊",
                "title": "Document Analytics", 
                "description": "วิเคราะห์และจัดการเอกสารอย่างครอบคลุม",
                "items": [
                    "Processing performance metrics",
                    "Knowledge base statistics",
                    "Document similarity analysis",
                    "Content organization and retrieval"
                ]
            }
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
                <ul>
                    {' '.join([f'<li>✓ {item}</li>' for item in feature['items']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Workflow diagram
        st.subheader("🔄 Enhanced Workflow")
        st.markdown("""
        ```mermaid
        graph TD
            A[📄 Upload Document] --> B[🌪️ OCR Processing]
            B --> C[📝 Extract Text/Markdown]
            C --> D{Add to Knowledge Base?}
            D -->|Yes| E[🧠 Store in RAG DB]
            D -->|No| F[💾 Download Results]
            E --> G[💬 AI Chat Ready]
            G --> H[🔍 Ask Questions]
            H --> I[🤖 Context-Aware Answers]
            
            J[📤 Manual Upload] --> E
            K[✍️ Direct Input] --> E
        ```
        """)
        
        # Performance comparison
        st.subheader("⚡ Model Performance Comparison")
        
        performance_data = {
            "Model": ["Typhoon OCR 7B", "Qwen2.5 14B", "Typhoon2 8B"],
            "Thai OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐"],
            "English OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐"],
            "Complex Tables": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
            "Thai Q&A": ["⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            "Processing Speed": ["⭐⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐"],
            "Best For": ["OCR Tasks", "RAG Q&A", "Thai Content"]
        }
        
        st.table(performance_data)
        
        # RAG Benefits
        st.subheader("🎯 RAG System Benefits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📚 Knowledge Management:**
            - Persistent document storage
            - Intelligent text chunking
            - Semantic search capabilities
            - Multi-document correlation
            
            **🔍 Smart Retrieval:**
            - Context-aware responses
            - Source attribution
            - Similarity scoring
            - Multi-language support
            """)
        
        with col2:
            st.markdown("""
            **⚡ Efficiency Gains:**
            - Instant document lookup
            - Automated knowledge base building
            - Batch processing integration
            - Scalable architecture
            
            **🎯 Use Cases:**
            - Corporate document Q&A
            - Research paper analysis
            - Legal document review
            - Technical documentation
            """)

    # ---- Tab 5: User Guide ----
    with tab5:
        st.header("📖 คู่มือการใช้งาน Typhoon OCR with RAG")
        
        st.markdown("""
## 🌟 ภาพรวมระบบที่ปรับปรุงแล้ว

Typhoon OCR with RAG เป็นระบบที่รวม **OCR ขั้นสูง** กับ **RAG (Retrieval-Augmented Generation)** เพื่อให้คุณสามารถ:

1. **📄 ประมวลผลเอกสาร** - แปลง PDF/รูปภาพเป็นข้อความ
2. **🧠 สร้างฐานความรู้** - เก็บข้อมูลอย่างเป็นระบบ
3. **💬 ถามตอบอัจฉริยะ** - ค้นหาและตอบคำถามจากเอกสาร

---

## 🔄 ขั้นตอนการใช้งานแบบครบวงจร

### ขั้นตอนที่ 1: OCR Processing
1. **เปิดแท็บ "📁 Upload & OCR"**
2. **อัพโหลดไฟล์** (PDF, PNG, JPG, JPEG)
3. **ตั้งค่าพารามิเตอร์** ตามต้องการ
4. **กดปุ่ม "🚀 Process Document(s)"**
5. **รอผลลัพธ์** และตรวจสอบความแม่นยำ

### ขั้นตอนที่ 2: เพิ่มเข้า Knowledge Base
หลังจาก OCR เสร็จแล้ว มี **3 วิธี** ในการเพิ่มข้อมูลเข้า Knowledge Base:

#### 🔹 วิธีที่ 1: Auto-add จากผล OCR
- หลังจาก OCR เสร็จ จะมีปุ่ม **"🧠 Add to Knowledge Base"**
- กดปุ่มนี้เพื่อเพิ่มข้อมูลโดยอัตโนมัติ

#### 🔹 วิธีที่ 2: Upload Markdown Files
- ไปที่แท็บ **"🧠 Knowledge Base"**
- ในส่วน **"Method 1: Upload Markdown Files"**
- อัพโหลดไฟล์ .md หรือ .txt
- กดปุ่ม **"➕ Add Markdown Files"**

#### 🔹 วิธีที่ 3: Direct Text Input
- ในส่วน **"Method 2: Direct Text Input"**
- ใส่ชื่อเอกสารและเนื้อหา
- กดปุ่ม **"➕ Add to Knowledge Base"**

### ขั้นตอนที่ 3: ถามตอบด้วย AI
1. **ไปที่แท็บ "💬 AI Chat"**
2. **พิมพ์คำถาม** ในช่อง "Ask a question"
3. **กดปุ่ม "🚀 Ask"**
4. **รอคำตอบ** พร้อมแหล่งอ้างอิง

---

## 🎯 เทคนิคการตั้งค่าพารามิเตอร์

### 📊 สำหรับ OCR (การแปลงเอกสาร)
- **Temperature: 0.1** → ความแม่นยำสูงสุด
- **Top P: 0.6** → สมดุลที่ดีที่สุด
- **Model: Typhoon OCR 7B** → เชี่ยวชาญด้าน OCR

### 💬 สำหรับ Q&A (การถามตอบ)
- **Temperature: 0.3** → ความสมดุลระหว่างแม่นยำและสร้างสรรค์
- **Model: Qwen2.5 14B** → เหมาะสำหรับการวิเคราะห์และตอบคำถาม
- **Model: Typhoon2 8B** → ดีที่สุดสำหรับภาษาไทย

---

## 🧠 RAG Knowledge Base คืออะไร?

**RAG (Retrieval-Augmented Generation)** เป็นเทคโนโลยีที่ช่วยให้ AI สามารถ:

### 🔍 การทำงานของ RAG:
1. **จัดเก็บข้อมูล** → แบ่งเอกสารเป็นชิ้นเล็ก (chunks)
2. **สร้าง Embeddings** → แปลงข้อความเป็นตัวเลขที่ AI เข้าใจ
3. **ค้นหาอัจฉริยะ** → หาข้อมูลที่เกี่ยวข้องด้วย Semantic Search
4. **ตอบคำถาม** → ใช้ข้อมูลที่ค้นหาได้มาสร้างคำตอบ

### 💡 ข้อดีของ RAG:
- ✅ **ตอบได้แม่นยำ** จากเอกสารจริง
- ✅ **แสดงแหล่งอ้างอิง** โปร่งใส
- ✅ **ไม่ Hallucination** ไม่แต่งเรื่อง
- ✅ **ปรับปรุงได้ตลอด** เพิ่มเอกสารใหม่ได้เรื่อย ๆ

---

## 📱 วิธีใช้แต่ละแท็บ

### 📁 Upload & OCR
- **จุดประสงค์:** แปลงเอกสารเป็นข้อความ
- **ไฟล์ที่รองรับ:** PDF, PNG, JPG, JPEG (สูงสุด 10MB)
- **เทคนิค:** ใช้ "Structure mode" สำหรับเอกสารที่มีตาราง

### 🧠 Knowledge Base  
- **จุดประสงค์:** จัดการฐานข้อมูลความรู้
- **ฟีเจอร์หลัก:**
  - ดูสถิติเอกสารทั้งหมด
  - เพิ่มเอกสารใหม่
  - ดูตัวอย่างเนื้อหา
  - ลบเอกสารที่ไม่ต้องการ

### 💬 AI Chat
- **จุดประสงค์:** ถามตอบเกี่ยวกับเอกสาร
- **ฟีเจอร์พิเศษ:**
  - แสดงแหล่งอ้างอิง
  - คะแนน Similarity
  - บันทึก Chat History
  - ล้างการสนทนาได้

---

## ⚡ เคล็ดลับการใช้งาน

### 🎯 เพิ่มประสิทธิภาพ OCR:
1. **สแกนด้วยความละเอียดสูง** (300 DPI ขึ้นไป)
2. **ตรวจสอบแสงให้เหมาะสม** หลีกเลี่ยงเงา
3. **เอกสารตรง** ไม่เอียง
4. **ใช้ "Structure mode"** สำหรับตารางซับซ้อน

### 🧠 เพิ่มประสิทธิภาพ Knowledge Base:
1. **แบ่งเอกสารใหญ่** เป็นหัวข้อย่อย ๆ
2. **ใช้ชื่อไฟล์ที่มีความหมาย** เช่น "รายงานการเงิน_Q1_2024.md"
3. **เพิ่มข้อมูลที่เกี่ยวข้องกัน** เพื่อให้ AI เข้าใจบริบท
4. **ทำความสะอาดข้อมูล** เป็นระยะ ลบเอกสารที่ไม่จำเป็น

### 💬 เทคนิคการถามคำถาม:
1. **ใช้คำถามที่ชัดเจน** เช่น "มียอดขายในเดือนมกราคมเท่าไหร่?"
2. **อ้างอิงชื่อเอกสาร** เช่น "จากรายงานการเงิน มีกำไรเท่าไหร่?"
3. **ใช้ภาษาไทยหรืออังกฤษ** ตามความเหมาะสม
4. **ขอให้เปรียบเทียบ** เช่น "เปรียบเทียบผลงานไตรมาส 1 กับ 2"

### 🔧 แก้ปัญหาเบื้องต้น:
- **OCR ไม่แม่นยำ** → เพิ่ม Image Quality, ใช้ Structure mode
- **ไม่พบข้อมูล** → ตรวจสอบว่าเพิ่มเอกสารเข้า Knowledge Base แล้ว
- **ตอบไม่ตรงคำถาม** → ทำให้คำถามชัดเจนขึ้น, เปลี่ยนโมเดล

---

## 🚀 การใช้งานขั้นสูง

### 📊 Batch Processing:
- เปิด "Enable Batch Processing" ในแท็บแรก
- อัพโหลดหลายไฟล์พร้อมกัน
- ใช้ปุ่ม "🧠 Add All to Knowledge Base" หลังประมวลผลเสร็จ

### 🔄 Workflow แนะนำ:
1. **วันแรก:** อัพโหลดเอกสารสำคัญทั้งหมด
2. **ทุกสัปดาห์:** เพิ่มเอกสารใหม่
3. **ใช้ประจำ:** ถามตอบเพื่อค้นหาข้อมูล
4. **ทุกเดือน:** ทำความสะอาด Knowledge Base

### 🎛️ การปรับแต่งโมเดล:
- **งาน OCR** → Typhoon OCR 7B
- **ถามตอบภาษาไทย** → Typhoon2 8B  
- **วิเคราะห์ซับซ้อน** → Qwen2.5 14B

---

## ❓ FAQ

**Q: ฐานข้อมูลจะหายไหมถ้าปิดเบราว์เซอร์?**
A: ไม่หาย ระบบใช้ SQLite เก็บข้อมูลแบบถาวร

**Q: สามารถใช้งานออฟไลน์ได้ไหม?**
A: ไม่ได้ ต้องเชื่อมต่ออินเทอร์เน็ตเพื่อใช้งาน AI

**Q: รองรับภาษาไทยแค่ไหน?**
A: รองรับเต็มรูปแบบทั้ง OCR และ Q&A

**Q: จำกัดขนาดไฟล์เท่าไหร่?**
A: ไฟล์เดี่ยว 10MB, ไม่จำกัดจำนวนไฟล์

**Q: ลบข้อมูลใน Knowledge Base ได้ไหม?**
A: ได้ ไปที่แท็บ Knowledge Base แล้วใช้ปุ่มลบในแต่ละเอกสาร
        """)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.error(f"Debug info: {traceback.format_exc()}")
