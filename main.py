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
    
    .rejection-message {
        background: #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #fdcb6e;
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
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200

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
    },
    "nomic-embed-text:latest": {
        "name": "Nomic Embed Text",
        "description": "โมเดล Embedding สำหรับ RAG",
        "icon": "🔍",
        "best_for": "Text embedding, Similarity search"
    }
}

# ==================== ENHANCED RAG SYSTEM FUNCTIONS ====================

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
                metadata TEXT,
                heading TEXT,
                content_type TEXT DEFAULT 'paragraph'
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
    
    def chunk_markdown(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Tuple[str, str, str]]:
        """
        Enhanced markdown chunking that preserves headings and structure
        Returns: List of (content, heading, content_type) tuples
        """
        chunks = []
        
        # Split by major markdown headings
        sections = re.split(r'\n(?=#{1,6}\s)', text)
        
        current_heading = "Introduction"
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this section starts with a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)', section)
            if heading_match:
                level = len(heading_match.group(1))
                current_heading = heading_match.group(2).strip()
                
                # Remove the heading from content for processing
                content_without_heading = re.sub(r'^#{1,6}\s+.+\n?', '', section, flags=re.MULTILINE)
            else:
                content_without_heading = section
            
            # If content is small enough, add as single chunk
            if len(content_without_heading) <= chunk_size:
                if content_without_heading.strip():
                    chunks.append((
                        f"# {current_heading}\n\n{content_without_heading}",
                        current_heading,
                        "section"
                    ))
            else:
                # Split large sections into smaller chunks
                sub_chunks = self.chunk_text(content_without_heading, chunk_size, overlap)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk_title = f"{current_heading} (Part {i+1})" if len(sub_chunks) > 1 else current_heading
                    chunks.append((
                        f"# {chunk_title}\n\n{sub_chunk}",
                        current_heading,
                        "subsection"
                    ))
        
        return chunks
    
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
        """Enhanced document addition with markdown awareness"""
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
            
            # Enhanced chunking for markdown files
            if filename.endswith('.md'):
                chunks = self.chunk_markdown(content, st.session_state.chunk_size, st.session_state.chunk_overlap)
            else:
                # Convert markdown to plain text for other files
                if filename.endswith('.md'):
                    html = markdown.markdown(content)
                    plain_text = re.sub('<[^<]+?>', '', html)
                    content = plain_text
                
                # Traditional text chunking
                text_chunks = self.chunk_text(content, st.session_state.chunk_size, st.session_state.chunk_overlap)
                chunks = [(chunk, "Document", "paragraph") for chunk in text_chunks]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (chunk_content, heading, content_type) in enumerate(chunks):
                status_text.text(f"Processing chunk {i+1}/{len(chunks)}: {heading[:50]}...")
                progress_bar.progress((i + 1) / len(chunks))
                
                # Get embedding
                embedding = self.get_embedding(chunk_content)
                if embedding:
                    # Store in database with enhanced metadata
                    cursor.execute('''
                        INSERT INTO documents 
                        (filename, content, embedding, chunk_id, file_hash, metadata, heading, content_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        filename, 
                        chunk_content, 
                        pickle.dumps(embedding), 
                        i,
                        file_hash,
                        json.dumps(metadata or {}),
                        heading,
                        content_type
                    ))
            
            conn.commit()
            conn.close()
            
            status_text.text(f"✅ Added {len(chunks)} chunks from {filename}")
            progress_bar.progress(1.0)
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding document: {str(e)}")
            return False
    
    def mmr_search(self, query: str, top_k: int = 5, lambda_param: float = 0.7) -> List[Tuple[str, str, float, str, str]]:
        """
        Enhanced search with MMR (Maximal Marginal Relevance) to reduce redundancy
        Returns: List of (filename, content, similarity, heading, content_type) tuples
        """
        try:
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT filename, content, embedding, heading, content_type FROM documents")
            
            # Calculate similarities
            candidates = []
            for filename, content, embedding_blob, heading, content_type in cursor.fetchall():
                stored_embedding = pickle.loads(embedding_blob)
                similarity = cosine_similarity(
                    [query_embedding], 
                    [stored_embedding]
                )[0][0]
                candidates.append((filename, content, similarity, heading, content_type))
            
            conn.close()
            
            # Sort by similarity
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Apply MMR
            selected = []
            remaining = candidates.copy()
            
            while len(selected) < top_k and remaining:
                if not selected:
                    # Select the most similar item first
                    best_item = remaining.pop(0)
                    selected.append(best_item)
                else:
                    # Apply MMR formula
                    best_score = -float('inf')
                    best_idx = 0
                    
                    for i, candidate in enumerate(remaining):
                        # Relevance score
                        relevance = candidate[2]
                        
                        # Calculate max similarity to already selected items
                        max_sim_to_selected = 0
                        candidate_embedding = self.get_embedding(candidate[1])
                        
                        if candidate_embedding:
                            for selected_item in selected:
                                selected_embedding = self.get_embedding(selected_item[1])
                                if selected_embedding:
                                    sim = cosine_similarity(
                                        [candidate_embedding], 
                                        [selected_embedding]
                                    )[0][0]
                                    max_sim_to_selected = max(max_sim_to_selected, sim)
                        
                        # MMR score
                        mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                        
                        if mmr_score > best_score:
                            best_score = mmr_score
                            best_idx = i
                    
                    selected.append(remaining.pop(best_idx))
            
            return selected
            
        except Exception as e:
            st.error(f"❌ Error in MMR search: {str(e)}")
            return []
    
    def search_similar(self, query: str, top_k: int = 5, use_mmr: bool = True) -> List[Tuple[str, str, float, str, str]]:
        """
        Enhanced search with option for MMR or traditional cosine similarity
        """
        if use_mmr:
            return self.mmr_search(query, top_k)
        else:
            # Traditional cosine similarity search
            try:
                query_embedding = self.get_embedding(query)
                if not query_embedding:
                    return []
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT filename, content, embedding, heading, content_type FROM documents")
                
                results = []
                for filename, content, embedding_blob, heading, content_type in cursor.fetchall():
                    stored_embedding = pickle.loads(embedding_blob)
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [stored_embedding]
                    )[0][0]
                    results.append((filename, content, similarity, heading, content_type))
                
                conn.close()
                
                # Sort by similarity and return top_k
                results.sort(key=lambda x: x[2], reverse=True)
                return results[:top_k]
                
            except Exception as e:
                st.error(f"❌ Error searching: {str(e)}")
                return []
    
    def reindex_documents(self) -> bool:
        """Re-create embeddings for all documents"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all documents
            cursor.execute("SELECT id, content FROM documents WHERE embedding IS NULL OR LENGTH(embedding) = 0")
            docs_to_reindex = cursor.fetchall()
            
            if not docs_to_reindex:
                st.info("ℹ️ All documents already have embeddings")
                return True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (doc_id, content) in enumerate(docs_to_reindex):
                status_text.text(f"Re-indexing document {i+1}/{len(docs_to_reindex)}...")
                progress_bar.progress((i + 1) / len(docs_to_reindex))
                
                embedding = self.get_embedding(content)
                if embedding:
                    cursor.execute(
                        "UPDATE documents SET embedding = ? WHERE id = ?",
                        (pickle.dumps(embedding), doc_id)
                    )
            
            conn.commit()
            conn.close()
            
            status_text.text("✅ Re-indexing completed")
            progress_bar.progress(1.0)
            return True
            
        except Exception as e:
            st.error(f"❌ Error re-indexing: {str(e)}")
            return False
    
    def reset_knowledge_base(self) -> bool:
        """Reset the entire knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM chat_sessions")
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error resetting knowledge base: {str(e)}")
            return False
    
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
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NULL OR LENGTH(embedding) = 0")
            missing_embeddings = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_chat_sessions": total_chats,
                "missing_embeddings": missing_embeddings
            }
        except Exception as e:
            return {"error": str(e)}

def generate_rag_response(query: str, context_docs: List[Tuple[str, str, float, str, str]], model: str = "qwen2.5:14b", min_similarity: float = 0.3) -> Optional[str]:
    """Enhanced RAG response generation with strict rejection criteria"""
    try:
        # Filter documents by minimum similarity threshold
        relevant_docs = [doc for doc in context_docs if doc[2] >= min_similarity]
        
        if not relevant_docs:
            return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในเอกสารที่มีอยู่ กรุณาลองใช้คำถามที่ชัดเจนมากขึ้น หรือตรวจสอบว่าได้เพิ่มเอกสารที่เกี่ยวข้องเข้าไปในฐานความรู้แล้ว"
        
        # Prepare enhanced context with source attribution
        context_text = ""
        for i, (filename, content, similarity, heading, content_type) in enumerate(relevant_docs[:3]):
            context_text += f"**แหล่งที่ {i+1}:** {filename} (หัวข้อ: {heading}) [ความเชื่อมั่น: {similarity:.2%}]\n"
            context_text += f"{content}\n\n"
        
        # Enhanced prompt with strict instructions
        prompt = f"""คุณเป็น AI Assistant ที่ตอบคำถามจากเอกสารที่ให้มาเท่านั้น กรุณาปฏิบัติตามกฎเหล่านี้อย่างเคร่งครัด:

1. ตอบเฉพาะข้อมูลที่มีในเอกสารที่ให้มา
2. ห้ามคาดเดาหรือเพิ่มเติมข้อมูลที่ไม่มีในเอกสาร
3. ระบุแหล่งที่มาของข้อมูลอย่างชัดเจน
4. หากไม่มีข้อมูลที่เกี่ยวข้อง ให้บอกว่า "ไม่พบข้อมูลที่เกี่ยวข้อง"

บริบทจากเอกสาร:
{context_text}

คำถาม: {query}

คำตอบ (พร้อมอ้างอิงแหล่งที่มา):"""
        
        # Call API
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.2,  # Lower temperature for more accurate responses
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

# ==================== ORIGINAL OCR FUNCTIONS (unchanged) ====================

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
        <h1>🌪️ Typhoon OCR with Enhanced RAG</h1>
        <p>AI-Powered Thai-English Document Parser + Intelligent Knowledge Q&A</p>
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
            
            # Show missing embeddings warning
            if stats.get("missing_embeddings", 0) > 0:
                st.warning(f"⚠️ {stats['missing_embeddings']} chunks need re-indexing")
        else:
            st.error("❌ Knowledge base error")
        
        # RAG Settings
        st.subheader("🔧 RAG Configuration")
        
        # Chunk settings
        new_chunk_size = st.slider(
            "Chunk Size", 
            500, 2000, 
            st.session_state.chunk_size, 
            100,
            help="Size of text chunks for embedding"
        )
        new_chunk_overlap = st.slider(
            "Chunk Overlap", 
            50, 500, 
            st.session_state.chunk_overlap, 
            25,
            help="Overlap between chunks"
        )
        
        # Update session state if changed
        if new_chunk_size != st.session_state.chunk_size or new_chunk_overlap != st.session_state.chunk_overlap:
            st.session_state.chunk_size = new_chunk_size
            st.session_state.chunk_overlap = new_chunk_overlap
            st.warning("⚠️ Chunk settings changed. Re-index documents for best results.")
        
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

    # ---- Tab 2: Enhanced Knowledge Base Management ----
    with tab2:
        st.header("🧠 Enhanced Knowledge Base Management")
        
        kb = RAGKnowledgeBase(st.session_state.rag_db_path)
        stats = kb.get_stats()
        
        # Display enhanced stats
        if not stats.get("error"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📚 Total Documents", stats["total_documents"])
            with col2:
                st.metric("📄 Total Chunks", stats["total_chunks"])
            with col3:
                st.metric("💬 Chat Sessions", stats["total_chat_sessions"])
            with col4:
                missing = stats.get("missing_embeddings", 0)
                st.metric("⚠️ Missing Embeddings", missing, delta="Need re-indexing" if missing > 0 else "All indexed")
        
        # Management buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Re-index Documents", help="Re-create embeddings for documents"):
                with st.spinner("Re-indexing documents..."):
                    success = kb.reindex_documents()
                    if success:
                        st.success("✅ Re-indexing completed!")
                        st.rerun()
        
        with col2:
            # Two-step confirmation for reset
            if 'reset_confirm' not in st.session_state:
                st.session_state.reset_confirm = False
            
            if not st.session_state.reset_confirm:
                if st.button("🗑️ Reset Knowledge Base", help="⚠️ This will delete all documents!"):
                    st.session_state.reset_confirm = True
                    st.rerun()
            else:
                st.warning("⚠️ This will permanently delete all documents and chat history!")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Confirm Reset", type="primary"):
                        if kb.reset_knowledge_base():
                            st.success("✅ Knowledge base reset successfully!")
                            st.session_state.reset_confirm = False
                            st.session_state.chat_history = []
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset knowledge base")
                with col_cancel:
                    if st.button("❌ Cancel"):
                        st.session_state.reset_confirm = False
                        st.rerun()
        
        with col3:
            # Export knowledge base
            if st.button("💾 Export Knowledge Base", help="Export all documents as markdown"):
                try:
                    conn = sqlite3.connect(st.session_state.rag_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT filename, content, heading FROM documents ORDER BY filename, chunk_id")
                    
                    export_content = "# Knowledge Base Export\n\n"
                    current_file = None
                    
                    for filename, content, heading in cursor.fetchall():
                        if filename != current_file:
                            export_content += f"\n---\n\n# 📄 {filename}\n\n"
                            current_file = filename
                        
                        export_content += f"{content}\n\n"
                    
                    conn.close()
                    
                    st.download_button(
                        "📦 Download Export",
                        export_content,
                        "knowledge_base_export.md",
                        "text/markdown"
                    )
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        st.markdown("---")
        
        # Add documents manually
        st.subheader("📤 Add Documents to Knowledge Base")
        
        # Method 1: Upload Markdown files
        st.write("**Method 1: Upload Markdown Files**")
        markdown_files = st.file_uploader(
            "Upload Markdown files",
            type=['md', 'txt'],
            accept_multiple_files=True,
            help="Upload .md or .txt files to add to knowledge base",
            key="kb_upload"
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
                placeholder="Paste or type your document content here...\n\n# Use markdown formatting\n## For better organization"
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
        
        # Enhanced knowledge base browser
        st.subheader("📖 Browse Knowledge Base")
        
        if stats.get("total_documents", 0) > 0:
            # Search within knowledge base
            search_query = st.text_input("🔍 Search documents:", placeholder="Search within your knowledge base...")
            
            # Get document list
            conn = sqlite3.connect(st.session_state.rag_db_path)
            cursor = conn.cursor()
            
            if search_query:
                # Simple text search for now
                cursor.execute("""
                    SELECT DISTINCT filename, COUNT(*) as chunk_count, MAX(created_at) as latest 
                    FROM documents 
                    WHERE content LIKE ? OR heading LIKE ? 
                    GROUP BY filename 
                    ORDER BY latest DESC
                """, (f"%{search_query}%", f"%{search_query}%"))
            else:
                cursor.execute("""
                    SELECT DISTINCT filename, COUNT(*) as chunk_count, MAX(created_at) as latest 
                    FROM documents 
                    GROUP BY filename 
                    ORDER BY latest DESC
                """)
            
            docs = cursor.fetchall()
            conn.close()
            
            # Display documents with enhanced info
            for filename, chunk_count, created_at in docs:
                with st.expander(f"📄 {filename} ({chunk_count} chunks) - {created_at}"):
                    # Get document chunks with headings
                    conn = sqlite3.connect(st.session_state.rag_db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, heading, content_type 
                        FROM documents 
                        WHERE filename = ? 
                        ORDER BY chunk_id
                    """, (filename,))
                    chunks = cursor.fetchall()
                    conn.close()
                    
                    st.write(f"**File:** {filename}")
                    st.write(f"**Chunks:** {chunk_count}")
                    st.write(f"**Created:** {created_at}")
                    
                    # Show chunk organization
                    headings = list(set([chunk[1] for chunk in chunks if chunk[1]]))
                    if headings:
                        st.write(f"**Sections:** {', '.join(headings[:5])}")
                        if len(headings) > 5:
                            st.write(f"... and {len(headings) - 5} more sections")
                    
                    # Show first chunk as preview
                    if chunks:
                        st.write("**Preview:**")
                        preview_content = chunks[0][0]
                        st.text(preview_content[:300] + "..." if len(preview_content) > 300 else preview_content)
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔍 View Full Content", key=f"view_{filename}"):
                            full_content = "\n\n---\n\n".join([chunk[0] for chunk in chunks])
                            st.text_area(f"Full content of {filename}:", full_content, height=400)
                    
                    with col2:
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

    # ---- Tab 3: Enhanced AI Chat with RAG ----
    with tab3:
        st.header("💬 Enhanced AI Chat with Knowledge Base")
        
        # Check if knowledge base has content
        if stats.get("total_documents", 0) == 0:
            st.warning("⚠️ No documents in knowledge base. Please add documents in the Knowledge Base tab first.")
            return
        
        # Chat model selection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            chat_model = st.selectbox(
                "🤖 Chat Model:",
                ["qwen2.5:14b", "scb10x/llama3.1-typhoon2-8b-instruct:latest", "nomic-embed-text:latest"],
                format_func=lambda x: f"{AVAILABLE_MODELS[x]['icon']} {AVAILABLE_MODELS[x]['name']}",
                help="Select model for generating responses"
            )
        
        with col2:
            use_mmr = st.checkbox("🎯 Use MMR Search", value=True, help="Reduces redundancy in search results")
        
        with col3:
            min_similarity = st.slider("🎚️ Min Similarity", 0.0, 1.0, 0.3, 0.05, help="Minimum similarity threshold for relevant documents")
        
        st.subheader("🗣️ Ask Questions About Your Documents")
        
        # Display enhanced chat history
        for i, (question, answer, context) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🤔 You:</strong><br>
                {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Check if this is a rejection response
            is_rejection = "ไม่พบข้อมูลที่เกี่ยวข้อง" in answer or "ขออภัย" in answer
            
            # Assistant message with appropriate styling
            message_class = "rejection-message" if is_rejection else "assistant-message"
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <strong>🤖 Assistant:</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Show context sources with enhanced info
            if context and not is_rejection:
                with st.expander(f"📚 Sources (Chat {i+1})", expanded=False):
                    for j, source in enumerate(context):
                        if len(source) >= 5:  # New format with heading and content_type
                            filename, content, similarity, heading, content_type = source[:5]
                            st.markdown(f"""
                            <div class="knowledge-card">
                                <strong>📄 {filename}</strong> → <em>{heading}</em> ({content_type})<br>
                                <strong>Similarity:</strong> {similarity:.1%}<br>
                                <strong>Content:</strong> {content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
                        else:  # Fallback for old format
                            filename, content, similarity = source[:3]
                            st.markdown(f"""
                            <div class="knowledge-card">
                                <strong>📄 {filename}</strong> (Similarity: {similarity:.1%})<br>
                                <em>{content[:200]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Enhanced chat input with suggestions
        st.markdown("---")
        
        # Sample questions
        if not st.session_state.chat_history:
            st.write("**💡 Sample Questions:**")
            sample_questions = [
                "สรุปเนื้อหาหลักของเอกสารทั้งหมด",
                "มีข้อมูลเกี่ยวกับ [หัวข้อที่สนใจ] หรือไม่",
                "เปรียบเทียบข้อมูลระหว่างเอกสารต่าง ๆ",
                "แสดงรายการสิ่งที่สำคัญจากเอกสาร"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                with cols[i % 2]:
                    if st.button(f"💬 {question}", key=f"sample_{i}"):
                        st.session_state.suggested_question = question
                        st.rerun()
        
        # Chat input form
        with st.form("enhanced_chat_form", clear_on_submit=True):
            default_question = st.session_state.get('suggested_question', '')
            if default_question:
                del st.session_state.suggested_question
            
            user_question = st.text_input(
                "Ask a question:",
                value=default_question,
                placeholder="What would you like to know about your documents?",
                key="enhanced_chat_input"
            )
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                ask_button = st.form_submit_button("🚀 Ask")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
            with col3:
                debug_button = st.form_submit_button("🔍 Debug Search")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and user_question:
            with st.spinner("🔍 Searching knowledge base and generating response..."):
                # Enhanced search with MMR
                context_docs = kb.search_similar(user_question, top_k=5, use_mmr=use_mmr)
                
                if context_docs:
                    # Generate enhanced RAG response
                    response = generate_rag_response(
                        user_question, 
                        context_docs, 
                        chat_model,
                        min_similarity
                    )
                    
                    if response:
                        # Add to chat history with enhanced context
                        st.session_state.chat_history.append((
                            user_question,
                            response,
                            context_docs[:3]  # Store top 3 for context
                        ))
                        
                        # Save to database with enhanced metadata
                        conn = sqlite3.connect(st.session_state.rag_db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO chat_sessions (session_id, question, answer, context)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            "enhanced_session",
                            user_question,
                            response,
                            json.dumps([{
                                "filename": doc[0], 
                                "content": doc[1][:200], 
                                "similarity": doc[2],
                                "heading": doc[3] if len(doc) > 3 else "Unknown",
                                "content_type": doc[4] if len(doc) > 4 else "paragraph"
                            } for doc in context_docs[:3]])
                        ))
                        conn.commit()
                        conn.close()
                        
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate response")
                else:
                    st.warning("⚠️ No relevant information found in knowledge base")
        
        if debug_button and user_question:
            with st.expander("🔍 Debug Search Results", expanded=True):
                st.write(f"**Query:** {user_question}")
                st.write(f"**Search Method:** {'MMR' if use_mmr else 'Cosine Similarity'}")
                st.write(f"**Minimum Similarity:** {min_similarity}")
                
                context_docs = kb.search_similar(user_question, top_k=10, use_mmr=use_mmr)
                
                if context_docs:
                    for i, doc in enumerate(context_docs):
                        filename, content, similarity = doc[:3]
                        heading = doc[3] if len(doc) > 3 else "Unknown"
                        content_type = doc[4] if len(doc) > 4 else "paragraph"
                        
                        color = "green" if similarity >= min_similarity else "red"
                        st.markdown(f"""
                        **Result {i+1}:** `{filename}` → `{heading}` ({content_type})  
                        **Similarity:** <span style="color: {color}">{similarity:.1%}</span>  
                        **Content:** {content[:150]}...
                        """, unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.warning("No results found")

    # ---- Tab 4: Enhanced Features ----
    with tab4:
        st.header("✨ Enhanced RAG Features")
        
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
                    "Multiple output formats (MD, HTML, JSON)",
                    "Automatic quality optimization"
                ]
            },
            {
                "icon": "🧠", 
                "title": "Intelligent RAG System",
                "description": "ระบบค้นหาและตอบคำถามที่ฉลาด",
                "items": [
                    "Smart markdown chunking with heading preservation",
                    "MMR search to reduce redundancy",
                    "Configurable similarity thresholds",
                    "Enhanced context attribution",
                    "Strict rejection of irrelevant queries"
                ]
            },
            {
                "icon": "💬",
                "title": "Enhanced AI Chat",
                "description": "ถามตอบอัจฉริยะที่ปรับปรุงแล้ว",
                "items": [
                    "Multiple model support with specializations",
                    "Source attribution with section headers",
                    "Debug mode for search analysis",
                    "Context-aware response generation",
                    "Conversation history with metadata"
                ]
            },
            {
                "icon": "🛠️",
                "title": "Advanced Management", 
                "description": "จัดการข้อมูลและระบบอย่างครอบคลุม",
                "items": [
                    "Re-indexing capabilities for updated settings",
                    "Two-step confirmation for destructive actions",
                    "Knowledge base export functionality",
                    "Enhanced document browsing with search",
                    "Performance monitoring and statistics"
                ]
            }
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p><strong>{feature['description']}</strong></p>
                <ul>
                    {' '.join([f'<li>✅ {item}</li>' for item in feature['items']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced workflow
        st.subheader("🔄 Enhanced Workflow")
        st.markdown("""
        ```mermaid
        graph TD
            A[📄 Upload Document] --> B[🌪️ OCR Processing]
            B --> C[📝 Extract Text/Markdown]
            C --> D{Add to Knowledge Base?}
            D -->|Yes| E[🧠 Intelligent Chunking]
            E --> F[🔍 Create Embeddings]
            F --> G[💾 Store with Metadata]
            D -->|No| H[💾 Download Results]
            G --> I[💬 Enhanced AI Chat]
            I --> J[🎯 MMR Search]
            J --> K[🤖 Context-Aware Response]
            K --> L[📚 Source Attribution]
            
            M[📤 Manual Upload] --> E
            N[✍️ Direct Input] --> E
            O[🔄 Re-index] --> F
            P[🗑️ Reset KB] --> Q[⚠️ Confirm]
        ```
        """)
        
        # Enhanced model comparison
        st.subheader("⚡ Model Performance & Specialization")
        
        performance_data = {
            "Model": ["Typhoon OCR 7B", "Qwen2.5 14B", "Typhoon2 8B", "Nomic Embed"],
            "Thai OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "N/A"],
            "English OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "N/A"],
            "Complex Tables": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "N/A"],
            "Thai Q&A": ["⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "N/A"],
            "Processing Speed": ["⭐⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            "Embedding Quality": ["N/A", "N/A", "N/A", "⭐⭐⭐⭐⭐"],
            "Best For": ["OCR Tasks", "Complex Reasoning", "Thai Content", "Embeddings"]
        }
        
        st.table(performance_data)
        
        # RAG System Benefits
        st.subheader("🎯 Enhanced RAG Benefits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Intelligent Processing:**
            - Markdown-aware chunking preserves document structure
            - Section headers maintained for better context
            - Configurable chunk sizes with overlap optimization
            - Smart embedding generation with retry logic
            
            **🔍 Advanced Search:**
            - MMR (Maximal Marginal Relevance) reduces redundancy
            - Similarity threshold filtering
            - Debug mode for search result analysis
            - Multi-criteria ranking system
            """)
        
        with col2:
            st.markdown("""
            **🤖 Enhanced Responses:**
            - Strict relevance checking prevents hallucination
            - Source attribution with section information
            - Multiple model support for specialized tasks
            - Context-aware response generation
            
            **🛠️ Management Tools:**
            - Re-indexing for updated configurations
            - Export functionality for backup
            - Enhanced browsing with search
            - Performance monitoring and statistics
            """)
        
        # Use case examples
        st.subheader("🎯 Advanced Use Cases")
        
        use_cases = [
            {
                "title": "📊 Research Analysis",
                "description": "Upload multiple research papers and ask comparative questions",
                "example": "\"Compare the methodologies used in papers A and B\""
            },
            {
                "title": "📋 Policy Documentation",
                "description": "Create a searchable policy database with instant Q&A",
                "example": "\"What are the requirements for employee vacation requests?\""
            },
            {
                "title": "📚 Educational Content",
                "description": "Build interactive learning materials with instant explanations",
                "example": "\"Explain the concept of photosynthesis from the biology textbook\""
            },
            {
                "title": "⚖️ Legal Document Review",
                "description": "Quick reference and analysis of legal documents",
                "example": "\"What are the termination clauses in the employment contract?\""
            }
        ]
        
        for i, use_case in enumerate(use_cases):
            with st.expander(f"{use_case['title']}: {use_case['description']}"):
                st.write(f"**Example Query:** {use_case['example']}")
                st.write("**Features Used:**")
                st.write("- Smart document chunking preserves structure")
                st.write("- MMR search ensures diverse, relevant results")  
                st.write("- Source attribution shows exact references")
                st.write("- Strict relevance filtering prevents false information")

    # ---- Tab 5: Enhanced User Guide ----
    with tab5:
        st.header("📖 คู่มือการใช้งาน Typhoon OCR with Enhanced RAG")
        
        st.markdown("""
## 🌟 ภาพรวมระบบที่ปรับปรุงใหม่

Typhoon OCR with Enhanced RAG เป็นระบบที่ได้รับการปรับปรุงให้มีความสามารถที่ก้าวหน้ายิ่งขึ้น:

### 🆕 **ฟีเจอร์ใหม่ที่เพิ่มเข้ามา:**
- 🧠 **Smart Markdown Chunking**: แบ่งชิ้นเอกสารแบบเก็บหัวข้อไว้
- 🎯 **MMR Search**: ค้นหาแบบลดความซ้ำซ้อน  
- ⚙️ **Configurable Settings**: ตั้งค่า chunk size และ overlap ได้
- 🔄 **Re-indexing**: สร้าง embeddings ใหม่เมื่อเปลี่ยนการตั้งค่า
- 🗑️ **Safe Reset**: ลบข้อมูลแบบยืนยัน 2 ชั้น
- 📊 **Enhanced Analytics**: สถิติและการวิเคราะห์ที่ละเอียดขึ้น

---

## 🔄 ขั้นตอนการใช้งานแบบปรับปรุงใหม่

### ขั้นตอนที่ 1: การตั้งค่าระบบ RAG
1. **ไปที่ Sidebar → RAG Configuration**
2. **ปรับ Chunk Size**: ขนาดชิ้นข้อความ (แนะนำ 1000-1500)
3. **ปรับ Chunk Overlap**: ความซ้อนทับ (แนะนำ 200-300)
4. **หากเปลี่ยนการตั้งค่า**: ใช้ปุ่ม "Re-index" ใน Knowledge Base tab

### ขั้นตอนที่ 2: OCR Processing (เหมือนเดิม)
1. **เปิดแท็บ "📁 Upload & OCR"**
2. **อัพโหลดไฟล์** (PDF, PNG, JPG, JPEG)
3. **ตั้งค่าพารามิเตอร์** ตามต้องการ
4. **กดปุ่ม "🚀 Process Document(s)"**
5. **รอผลลัพธ์** และตรวจสอบความแม่นยำ

### ขั้นตอนที่ 3: Enhanced Knowledge Management
หลังจาก OCR เสร็จแล้ว มี **4 วิธี** ในการจัดการ Knowledge Base:

#### 🔹 วิธีที่ 1: Auto-add จากผล OCR
- หลังจาก OCR เสร็จ จะมีปุ่ม **"🧠 Add to Knowledge Base"**
- ระบบจะใช้ **Smart Chunking** แบ่งข้อความตามหัวข้อ Markdown

#### 🔹 วิธีที่ 2: Upload Markdown Files (แนะนำ)
- ไปที่แท็บ **"🧠 Knowledge Base"**
- อัพโหลดไฟล์ .md ที่จัดรูปแบบด้วย headers (# ## ###)
- ระบบจะแยกตามหัวข้อโดยอัตโนมัติ

#### 🔹 วิธีที่ 3: Direct Text Input
- ใส่ชื่อเอกสารและเนื้อหา
- **แนะนำ**: ใช้ Markdown format เพื่อการแบ่ง chunk ที่ดี
- ตัวอย่าง:
```markdown
# หัวข้อหลัก
## หัวข้อย่อย 1
เนื้อหาส่วนแรก...

## หัวข้อย่อย 2  
เนื้อหาส่วนที่สอง...
```

#### 🔹 วิธีที่ 4: Management Tools
- **🔄 Re-index Documents**: สร้าง embeddings ใหม่
- **🗑️ Reset Knowledge Base**: ลบข้อมูลทั้งหมด (ยืนยัน 2 ชั้น)
- **💾 Export Knowledge Base**: ส่งออกเป็นไฟล์ Markdown
- **🔍 Search Documents**: ค้นหาในเอกสารที่มีอยู่

### ขั้นตอนที่ 4: Enhanced AI Chat
1. **ไปที่แท็บ "💬 AI Chat"**
2. **เลือกโมเดล**:
   - **Qwen2.5 14B**: วิเคราะห์ซับซ้อน ภาษาอังกฤษ
   - **Typhoon2 8B**: เหมาะสำหรับภาษาไทย
   - **Nomic Embed**: สำหรับการค้นหาเฉพาะ
3. **ตั้งค่า Search**:
   - ✅ **Use MMR Search**: ลดความซ้ำซ้อนในผลลัพธ์
   - **Min Similarity**: กรองเอกสารที่ไม่เกี่ยวข้อง (แนะนำ 0.3)
4. **พิมพ์คำถาม** และ **กดปุ่ม "🚀 Ask"**
5. **ใช้ Debug Mode** เพื่อดูผลการค้นหาแบบละเอียด

---

## 🎯 เทคนิคการใช้งานขั้นสูง

### 📊 การตั้งค่าที่เหมาะสมสำหรับแต่ละประเภท

#### 🔹 เอกสารสั้น (1-5 หน้า):
- **Chunk Size**: 800-1000
- **Overlap**: 150-200  
- **Min Similarity**: 0.4-0.5

#### 🔹 เอกสารยาว (10+ หน้า):
- **Chunk Size**: 1200-1500
- **Overlap**: 250-300
- **Min Similarity**: 0.3-0.4

#### 🔹 เอกสารเทคนิค (คู่มือ, กฎหมาย):
- **Chunk Size**: 1000-1200
- **Overlap**: 200-250
- **Min Similarity**: 0.3
- **MMR**: เปิดใช้งานเสมอ

#### 🔹 เอกสารการวิจัย (บทความ, รายงาน):
- **Chunk Size**: 1500-2000  
- **Overlap**: 300-400
- **Min Similarity**: 0.25
- **Model**: Qwen2.5 14B

### 🧠 เทคนิค Markdown Chunking

ระบบใหม่จะแบ่ง chunk ตามโครงสร้าง Markdown:

```markdown
# หัวข้อใหญ่ (จะเป็น chunk แยก)
เนื้อหาส่วนแรก...

## หัวข้อย่อย A (chunk ใหม่)
รายละเอียดของ A...

### หัวข้อย่อยของ A (ถ้าเนื้อหาเยอะจะแยก chunk)
รายละเอียดเพิ่มเติม...

## หัวข้อย่อย B (chunk ใหม่)  
รายละเอียดของ B...
```

**ข้อดี**:
- ✅ เก็บบริบทของหัวข้อไว้
- ✅ ค้นหาแม่นยำขึ้น
- ✅ อ้างอิงแหล่งที่มาชัดเจน

### 🔍 เทคนิค MMR Search

**MMR (Maximal Marginal Relevance)** ช่วยลดความซ้ำซ้อน:

**เปิดใช้เมื่อ**:
- ✅ เอกสารหลายชิ้นที่คล้ายกัน
- ✅ ต้องการมุมมองหลากหลาย
- ✅ คำถามที่กว้าง เช่น "สรุปทั้งหมด"

**ปิดใช้เมื่อ**:
- ❌ ต้องการข้อมูลที่เฉพาะเจาะจง
- ❌ เอกสารมีเนื้อหาที่แตกต่างกันมาก
- ❌ คำถามที่ระบุชัดเจน

### 💬 เทคนิคการถามคำถามขั้นสูง

#### 🎯 **คำถามที่มีประสิทธิภาพ**:
1. **ระบุบริบท**: "จากเอกสารเรื่องการเงิน มีกำไรเท่าไหร่"
2. **ขอการเปรียบเทียบ**: "เปรียบเทียบวิธีการ A และ B"  
3. **ขอรายการ**: "จงแสดงขั้นตอนการ..."
4. **ระบุช่วงเวลา**: "ในไตรมาสแรก มียอดขายเท่าไหร่"

#### ❌ **คำถามที่ควรหลีกเลี่ยง**:
- คำถามที่กว้างเกินไป: "เอกสารนี้พูดถึงอะไร"
- คำถามที่ไม่เกี่ยวกับเอกสาร: "อากาศวันนี้เป็นยังไง"
- คำถามที่ต้องการข้อมูลนอกเอกสาร

### 🔧 การแก้ปัญหาขั้นสูง

#### ⚠️ **เมื่อไม่พบข้อมูลที่เกี่ยวข้อง**:
1. **ลดค่า Min Similarity** จาก 0.3 เป็น 0.2
2. **ใช้คำถามที่กว้างขึ้น**
3. **ตรวจสอบว่าเพิ่มเอกสารที่เกี่ยวข้องแล้ว**
4. **ใช้ Debug Mode ดูผลการค้นหา**

#### 🔄 **เมื่อเปลี่ยนการตั้งค่า Chunk**:
1. **บันทึกเอกสารสำคัญ** (Export ก่อน)
2. **ใช้ปุ่ม "Re-index Documents"**  
3. **หรือ Reset KB และเพิ่มเอกสารใหม่**

#### 🐛 **Debug Mode การใช้งาน**:
- ดู **Similarity Score** ของแต่ละผลลัพธ์
- ตรวจสอบว่า **Search Method** เป็น MMR หรือ Cosine
- เปรียบเทียบ **ผลลัพธ์ที่ผ่าน** และ **ไม่ผ่าน threshold**

---

## 🚀 Workflow แนะนำสำหรับการใช้งานจริง

### 📅 **วันแรก: Setup ระบบ**
1. ปรับการตั้งค่า RAG ตามประเภทเอกสาร
2. อัพโหลดเอกสารสำคัญทั้งหมด
3. ทดสอบคำถามพื้นฐาน
4. Export ข้อมูลเป็น backup

### 📊 **ทุกสัปดาห์: Maintenance**
1. เพิ่มเอกสารใหม่
2. ทดสอบคำถามที่ซับซ้อนขึ้น
3. ตรวจสอบสถิติและประสิทธิภาพ
4. ลบเอกสารที่ไม่จำเป็น

### 🎯 **ใช้งานประจำ: Optimization**
- ใช้ MMR สำหรับคำถามกว้าง ๆ
- ใช้ Cosine Similarity สำหรับคำถามเฉพาะ
- ปรับ Min Similarity ตามความต้องการ
- ใช้ Debug Mode เพื่อปรับปรุงคำถาม

### 🔄 **ทุกเดือน: Review**
1. ทบทวนการตั้งค่า Chunk Size
2. วิเคราะห์ Chat History
3. อัพเดตเอกสารเก่า
4. Re-index หากจำเป็น

---

## 📊 ตัวอย่างการใช้งานในสถานการณ์จริง

### 🏢 **บริษัท: จัดการเอกสารนโยบาย**
```
Setup:
- Chunk Size: 1200 (นโยบายมักมีหัวข้อชัดเจน)
- Overlap: 250
- Model: Typhoon2 8B (เนื้อหาภาษาไทย)
- MMR: เปิด (หลายนโยบายคล้ายกัน)

Sample Questions:
- "นโยบายการลาของพนักงานเป็นอย่างไร"
- "เปรียบเทียบสิทธิประโยชน์ระหว่างตำแหน่งต่าง ๆ"
- "ขั้นตอนการขอเลื่อนตำแหน่งมีอะไรบ้าง"
```

### 🎓 **มหาวิทยาลัย: ระบบ Q&A เอกสารวิชาการ**
```
Setup:
- Chunk Size: 1500 (เอกสารวิชาการมีเนื้อหาซับซ้อน)
- Overlap: 300  
- Model: Qwen2.5 14B (วิเคราะห์เชิงลึก)
- MMR: เปิด (เปรียบเทียบทฤษฎี)

Sample Questions:
- "อธิบายทฤษฎี X จากบทที่ 3"  
- "เปรียบเทียบแนวคิดของนักวิชาการ A และ B"
- "ยกตัวอย่างการประยุกต์ใช้หลักการ Y"
```

### ⚖️ **สำนักงานกฎหมาย: ค้นหาข้อกฎหมาย**
```
Setup:  
- Chunk Size: 1000 (ข้อกฎหมายมีโครงสร้างชัดเจน)
- Overlap: 200
- Model: Qwen2.5 14B (วิเคราะห์เชิงลึก)  
- MMR: ปิด (ต้องการข้อมูลที่แม่นยำ)
- Min Similarity: 0.4 (เข้มงวด)

Sample Questions:
- "มาตรา 25 ของพระราชบัญญัติ ABC กำหนดอย่างไร"
- "โทษสำหรับการกระทำ X มีอะไรบ้าง"  
- "ขั้นตอนการดำเนินคดีในกรณี Y"
```

### 🔬 **นักวิจัย: วิเคราะห์เอกสารงานวิจัย**
```
Setup:
- Chunk Size: 1800 (บทความวิจัยมีเนื้อหาเยอะ)
- Overlap: 400
- Model: Qwen2.5 14B (วิเคราะห์สลับซับซ้อน)
- MMR: เปิด (เปรียบเทียบงานวิจัย)

Sample Questions:
- "สรุป methodology ของงานวิจัยทั้งหมด"
- "ข้อจำกัดของการศึกษาเหล่านี้คืออะไร"
- "ผลการวิจัยสอดคล้องกันหรือไม่"
```

---

## ❓ Enhanced FAQ

### 🛠️ **Technical Questions**

**Q: ทำไมต้อง Re-index หลังเปลี่ยน Chunk Size?**  
A: เพราะ embedding vector จะไม่ตรงกับเนื้อหา chunk ใหม่ ทำให้การค้นหาไม่แม่นยำ

**Q: MMR ต่างจาก Cosine Similarity อย่างไร?**  
A: MMR พิจารณาทั้งความเกี่ยวข้อง และความหลากหลาย ในขณะที่ Cosine มองแค่ความเกี่ยวข้อง

**Q: Min Similarity 0.3 หมายความว่าอย่างไร?**  
A: เอกสารที่มีคะแนนความคล้ายน้อยกว่า 30% จะถูกปฏิเสธ ไม่นำมาตอบคำถาม

### 🎯 **Usage Questions**

**Q: ควรใช้โมเดลไหนสำหรับเอกสารภาษาไทย?**  
A: Typhoon2 8B เหมาะสำหรับภาษาไทยโดยเฉพาะ แต่ Qwen2.5 14B ดีกว่าสำหรับการวิเคราะห์ซับซ้อน

**Q: เมื่อไหร่ควรเปิด/ปิด MMR?**  
A: เปิดเมื่อมีเอกสารหลายชิ้นที่คล้ายกัน ปิดเมื่อต้องการข้อมูลที่แม่นยำเฉพาะเจาะจง

**Q: ทำไมระบบบอกว่า "ไม่พบข้อมูลที่เกี่ยวข้อง"?**  
A: เพราะไม่มีเอกสารใดผ่าน Min Similarity threshold ลองลดค่าหรือปรับคำถาม

### 🔧 **Troubleshooting**

**Q: จะรู้ได้อย่างไรว่าการตั้งค่าเหมาะสม?**  
A: ใช้ Debug Mode ดูคะแนน Similarity และปรับจนได้ผลลัพธ์ที่ต้องการ

**Q: เอกสารหายหลัง Reset ได้คืนมาไหม?**  
A: ไม่ได้ เพราะฉะนั้นควร Export ก่อน Reset เสมอ

**Q: ระบบช้าเมื่อมีเอกสารเยอะ?**  
A: ปกติ เพราะต้องค้นหาใน embedding space ที่ใหญ่ขึ้น ลองลด top_k ในการค้นหา

---

## 🎉 สรุป: Best Practices

### ✅ **ควรทำ**:
1. **ใช้ Markdown format** เมื่อเพิ่มเอกสาร
2. **Export backup** ก่อนทำการเปลี่ยนแปลงใหญ่
3. **ทดสอบคำถาม** หลังเพิ่มเอกสารใหม่
4. **ใช้ Debug Mode** เพื่อปรับปรุงผลลัพธ์
5. **ปรับการตั้งค่า** ตามประเภทเอกสาร

### ❌ **ไม่ควรทำ**:
1. **Reset โดยไม่ Export** ข้อมูลก่อน
2. **ใช้คำถามที่กว้างเกินไป** โดยไม่ระบุบริบท
3. **เพิ่มเอกสารที่ไม่เกี่ยวข้อง** เพราะจะรบกวนผลการค้นหา
4. **เปลี่ยน Chunk Size** บ่อยเกินไปโดยไม่ Re-index
5. **ตั้ง Min Similarity สูงเกินไป** จนไม่มีผลลัพธ์

### 🏆 **เป้าหมายสูงสุด**:
สร้างระบบ RAG ที่**แม่นยำ**, **ครอบคลุม**, และ**ใช้งานง่าย**  
เพื่อให้คุณค้นหาข้อมูลได้รวดเร็วและเชื่อถือได้ 100%
        """)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.error(f"Debug info: {traceback.format_exc()}")
