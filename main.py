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
    
    .source-reference {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        font-size: 0.9em;
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
    },
    "nomic-embed-text:latest": {
        "name": "Nomic Embed Text",
        "description": "โมเดลสำหรับการสร้าง Embeddings",
        "icon": "🔗",
        "best_for": "Text embeddings, Semantic search"
    }
}

# ==================== ENHANCED RAG SYSTEM FUNCTIONS ====================

def mmr_search(query_embedding: List[float], documents: List[Tuple], 
               top_k: int = 5, diversity_weight: float = 0.3) -> List[Tuple]:
    """
    Maximal Marginal Relevance search for balanced relevance and diversity
    """
    if not documents:
        return []
    
    # Calculate similarity scores
    doc_embeddings = [pickle.loads(doc[2]) for doc in documents]  # embedding is at index 2
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Initialize result set
    selected_docs = []
    remaining_docs = list(enumerate(zip(documents, similarities)))
    
    # Select first document (highest similarity)
    if remaining_docs:
        best_idx = max(remaining_docs, key=lambda x: x[1][1])[0]
        selected_docs.append(remaining_docs[best_idx][1][0])
        remaining_docs = [doc for i, doc in remaining_docs if i != best_idx]
    
    # Select remaining documents with MMR
    while len(selected_docs) < top_k and remaining_docs:
        best_score = -1
        best_idx = -1
        
        for i, (doc, sim) in remaining_docs:
            # Calculate diversity score (minimum similarity to already selected docs)
            selected_embeddings = [pickle.loads(selected_doc[2]) for selected_doc in selected_docs]
            doc_embedding = pickle.loads(doc[2])
            
            if selected_embeddings:
                diversity_scores = cosine_similarity([doc_embedding], selected_embeddings)[0]
                diversity = 1 - max(diversity_scores)  # 1 - max similarity = diversity
            else:
                diversity = 1.0
            
            # MMR score: balance between relevance and diversity
            mmr_score = (1 - diversity_weight) * sim + diversity_weight * diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx != -1:
            selected_docs.append(remaining_docs[best_idx][1][0])
            remaining_docs = [doc for i, doc in remaining_docs if i != best_idx]
        else:
            break
    
    return selected_docs

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
                header_level INTEGER DEFAULT 0,
                section_title TEXT DEFAULT ''
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
    
    def chunk_markdown(self, content: str) -> List[Dict[str, Any]]:
        """
        Smart markdown chunking that preserves headers and context
        """
        chunks = []
        lines = content.split('\n')
        
        current_chunk = ""
        current_headers = []
        chunk_id = 0
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if header_match:
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                # Save previous chunk if it exists and has content
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'headers': current_headers.copy(),
                        'section_title': current_headers[-1] if current_headers else ''
                    })
                    chunk_id += 1
                
                # Update headers hierarchy
                current_headers = current_headers[:header_level-1] + [header_text]
                current_chunk = line + '\n'
                
            else:
                current_chunk += line + '\n'
                
                # If chunk gets too long, split it but preserve context
                if len(current_chunk) > 1500:  # Adjustable chunk size
                    # Try to find a good break point
                    sentences = current_chunk.split('. ')
                    if len(sentences) > 1:
                        # Take most sentences, leave some overlap
                        break_point = len(sentences) * 2 // 3
                        chunk_content = '. '.join(sentences[:break_point]) + '.'
                        remaining_content = '. '.join(sentences[break_point:])
                        
                        chunks.append({
                            'content': chunk_content.strip(),
                            'chunk_id': chunk_id,
                            'headers': current_headers.copy(),
                            'section_title': current_headers[-1] if current_headers else ''
                        })
                        chunk_id += 1
                        
                        # Start new chunk with header context
                        header_context = '\n'.join([f"{'#' * (i+1)} {h}" for i, h in enumerate(current_headers)])
                        current_chunk = header_context + '\n\n' + remaining_content
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'chunk_id': chunk_id,
                'headers': current_headers.copy(),
                'section_title': current_headers[-1] if current_headers else ''
            })
        
        return chunks
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks (for non-markdown content)"""
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
        """Add document to knowledge base with enhanced chunking"""
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
            
            # Choose chunking strategy based on file type
            if filename.endswith('.md'):
                # Use smart markdown chunking
                chunks_data = self.chunk_markdown(content)
                chunk_type = "markdown"
            else:
                # Use regular text chunking
                plain_chunks = self.chunk_text(content)
                chunks_data = [
                    {
                        'content': chunk,
                        'chunk_id': i,
                        'headers': [],
                        'section_title': ''
                    }
                    for i, chunk in enumerate(plain_chunks)
                ]
                chunk_type = "text"
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk_data in enumerate(chunks_data):
                status_text.text(f"Processing {chunk_type} chunk {i+1}/{len(chunks_data)}...")
                progress_bar.progress((i + 1) / len(chunks_data))
                
                # Get embedding
                embedding = self.get_embedding(chunk_data['content'])
                if embedding:
                    # Store in database with enhanced metadata
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'headers': chunk_data['headers'],
                        'chunk_type': chunk_type
                    })
                    
                    cursor.execute('''
                        INSERT INTO documents 
                        (filename, content, embedding, chunk_id, file_hash, metadata, 
                         header_level, section_title)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        filename, 
                        chunk_data['content'], 
                        pickle.dumps(embedding), 
                        chunk_data['chunk_id'],
                        file_hash,
                        json.dumps(chunk_metadata),
                        len(chunk_data['headers']) if chunk_data['headers'] else 0,
                        chunk_data['section_title']
                    ))
            
            conn.commit()
            conn.close()
            
            status_text.text(f"✅ Added {len(chunks_data)} {chunk_type} chunks from {filename}")
            progress_bar.progress(1.0)
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding document: {str(e)}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5, use_mmr: bool = True, 
                      min_similarity: float = 0.3) -> List[Tuple[str, str, float, str]]:
        """
        Enhanced search with MMR and strict relevance filtering
        Returns: List of (filename, content, similarity, section_title)
        """
        try:
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT filename, content, embedding, section_title 
                FROM documents 
                ORDER BY created_at DESC
            """)
            
            documents = cursor.fetchall()
            conn.close()
            
            if not documents:
                return []
            
            # Calculate similarities and filter by minimum threshold
            results = []
            for filename, content, embedding_blob, section_title in documents:
                stored_embedding = pickle.loads(embedding_blob)
                similarity = cosine_similarity(
                    [query_embedding], 
                    [stored_embedding]
                )[0][0]
                
                # Only include results above minimum similarity threshold
                if similarity >= min_similarity:
                    results.append((filename, content, embedding_blob, section_title, similarity))
            
            if not results:
                return []  # No results above threshold
            
            # Sort by similarity for MMR or regular search
            if use_mmr and len(results) > 1:
                # Prepare documents for MMR (filename, content, embedding_blob, section_title)
                mmr_docs = [(r[0], r[1], r[2], r[3]) for r in results]
                selected_docs = mmr_search(query_embedding, mmr_docs, top_k)
                
                # Convert back to expected format and recalculate similarity for display
                final_results = []
                for filename, content, embedding_blob, section_title in selected_docs:
                    stored_embedding = pickle.loads(embedding_blob)
                    similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                    final_results.append((filename, content, similarity, section_title))
                
                return final_results
            else:
                # Regular similarity search
                results.sort(key=lambda x: x[4], reverse=True)
                return [(r[0], r[1], r[4], r[3]) for r in results[:top_k]]
            
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
            
            # Get chunk type distribution
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN metadata LIKE '%markdown%' THEN 1 ELSE 0 END) as markdown_chunks,
                    SUM(CASE WHEN metadata NOT LIKE '%markdown%' THEN 1 ELSE 0 END) as text_chunks
                FROM documents
            """)
            chunk_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_chat_sessions": total_chats,
                "markdown_chunks": chunk_stats[0] or 0,
                "text_chunks": chunk_stats[1] or 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def reset_knowledge_base(self) -> bool:
        """Reset entire knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM chat_sessions")
            cursor.execute("VACUUM")  # Reclaim space
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"❌ Error resetting knowledge base: {str(e)}")
            return False
    
    def reindex_documents(self) -> bool:
        """Re-create embeddings for all documents"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all documents
            cursor.execute("SELECT id, content FROM documents")
            documents = cursor.fetchall()
            
            if not documents:
                conn.close()
                st.info("ℹ️ No documents to reindex")
                return True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            for i, (doc_id, content) in enumerate(documents):
                status_text.text(f"Re-indexing document {i+1}/{len(documents)}...")
                progress_bar.progress((i + 1) / len(documents))
                
                # Get new embedding
                embedding = self.get_embedding(content)
                if embedding:
                    # Update embedding in database
                    cursor.execute(
                        "UPDATE documents SET embedding = ? WHERE id = ?",
                        (pickle.dumps(embedding), doc_id)
                    )
                    success_count += 1
            
            conn.commit()
            conn.close()
            
            status_text.text(f"✅ Re-indexed {success_count}/{len(documents)} documents")
            progress_bar.progress(1.0)
            return True
            
        except Exception as e:
            st.error(f"❌ Error re-indexing: {str(e)}")
            return False

def generate_rag_response(query: str, context_docs: List[Tuple[str, str, float, str]], 
                         model: str = "qwen2.5:14b", strict_mode: bool = True) -> Optional[str]:
    """Generate response using RAG context with enhanced source attribution"""
    try:
        if not context_docs:
            if strict_mode:
                return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในฐานข้อมูลความรู้ กรุณาลองปรับคำถาม หรือเพิ่มเอกสารที่เกี่ยวข้องเข้าสู่ระบบ"
            else:
                return None
        
        # Prepare context with source attribution
        context_parts = []
        for i, (filename, content, similarity, section_title) in enumerate(context_docs[:3]):
            section_info = f" (หัวข้อ: {section_title})" if section_title else ""
            context_parts.append(
                f"[เอกสารที่ {i+1}: {filename}{section_info} - ความเกี่ยวข้อง: {similarity:.3f}]\n{content}"
            )
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Create enhanced prompt
        prompt = f"""คุณเป็น AI Assistant ที่ช่วยตอบคำถามจากเอกสารที่กำหนดให้เท่านั้น

กฎสำคัญ:
1. ตอบคำถามโดยใช้ข้อมูลจากเอกสารที่ให้มาเท่านั้น
2. หากไม่มีข้อมูลที่เกี่ยวข้องในเอกสาร ให้บอกว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่กำหนด"
3. อ้างอิงแหล่งที่มาทุกครั้ง โดยระบุชื่อเอกสารและหัวข้อ (ถ้ามี)
4. ห้ามแต่งหรือคาดเดาข้อมูลที่ไม่ได้ระบุในเอกสาร

บริบทจากเอกสาร:
{context_text}

คำถาม: {query}

คำตอบ (พร้อมแหล่งอ้างอิง):"""
        
        # Call API
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.2,  # Lower temperature for more factual responses
                "top_p": 0.8,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result.get('response', '')
        
        if answer:
            # Add source references at the end
            source_refs = "\n\n📚 **แหล่งอ้างอิง:**\n"
            for i, (filename, content, similarity, section_title) in enumerate(context_docs[:3]):
                section_info = f" → {section_title}" if section_title else ""
                source_refs += f"- **เอกสารที่ {i+1}:** {filename}{section_info} (ความเกี่ยวข้อง: {similarity:.1%})\n"
            
            return answer + source_refs
        
        return None
        
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
                        combined_content += f"## 🚀 ขั้นตอนการใช้งานแบบครบวงจร

### ขั้นตอนที่ 1: OCR Processing (การแปลงเอกสาร)
1. **เปิดแท็บ "📁 Upload & OCR"**
2. **อัพโหลดไฟล์** (PDF, PNG, JPG, JPEG - สูงสุด 10MB)
3. **ตั้งค่าพารามิเตอร์:**
   - **Model:** Typhoon OCR 7B สำหรับ OCR
   - **Temperature:** 0.1 (ความแม่นยำสูงสุด)
   - **Prompt Type:** Structure สำหรับตารางซับซ้อน
4. **กดปุ่ม "🚀 Process Document(s)"**
5. **รอผลลัพธ์** และตรวจสอบความแม่นยำ

### ขั้นตอนที่ 2: เพิ่มเข้า Enhanced Knowledge Base
มี **3 วิธีหลัก** ในการเพิ่มข้อมูล พร้อมระบบ Smart Chunking:

#### 🔹 วิธีที่ 1: Auto-add จากผล OCR
- หลังจาก OCR เสร็จ จะมีปุ่ม **"🧠 Add to Knowledge Base"**
- ระบบจะใช้ Regular Text Chunking สำหรับผลลัพธ์ OCR

#### 🔹 วิธีที่ 2: Upload Markdown Files (แนะนำ!)
- ไปที่แท็บ **"🧠 Knowledge Base"**
- อัพโหลดไฟล์ .md ในส่วน **"Method 1"**
- ระบบจะใช้ **Header-Aware Chunking** อัตโนมัติ
- เก็บรักษาโครงสร้างหัวข้อและบริบท

#### 🔹 วิธีที่ 3: Direct Text Input
- เลือก **"Content Format: markdown"** สำหรับ Smart Chunking
- ใส่เนื้อหาในรูปแบบ Markdown พร้อมหัวข้อ
- ระบบจะแยก chunk ตามโครงสร้างหัวข้อ

### ขั้นตอนที่ 3: ถามตอบด้วย Enhanced AI Chat
1. **ไปที่แท็บ "💬 AI Chat"**
2. **ตั้งค่า Chat Settings:**
   - **Model:** Qwen2.5 14B สำหรับการวิเคราะห์ทั่วไป หรือ Typhoon2 8B สำหรับภาษาไทย
   - **MMR Search:** เปิดใช้งานสำหรับผลลัพธ์ที่หลากหลาย
   - **Strict Mode:** เปิดเพื่อป้องกันการ hallucination
3. **ปรับแต่ง Advanced Settings:**
   - **Top K:** 5-7 สำหรับคำถามทั่วไป
   - **Min Similarity:** 0.3 (ปรับลดถ้าไม่เจอข้อมูล)
   - **Diversity Weight:** 0.3 สำหรับความสมดุล
4. **พิมพ์คำถาม** และรอคำตอบพร้อมแหล่งอ้างอิง

---

## 🎯 เทคนิคการตั้งค่าขั้นสูง

### 📊 การเลือก Model ให้เหมาะสม

#### OCR Processing:
- **Typhoon OCR 7B:** เชี่ยวชาญ OCR ไทย-อังกฤษ
- **Temperature: 0.1, Top P: 0.6** → ความแม่นยำสูงสุด

#### Q&A Chat:
- **Typhoon2 8B:** ดีที่สุดสำหรับภาษาไทย
- **Qwen2.5 14B:** เหมาะสำหรับการวิเคราะห์ซับซ้อน
- **Temperature: 0.2-0.3** → สมดุลระหว่างแม่นยำและสร้างสรรค์

#### Embeddings:
- **Nomic Embed Text:** ใช้สำหรับสร้าง embeddings (ทำงานอัตโนมัติ)

### 🔍 การตั้งค่า Search Algorithm

#### MMR Search (แนะนำ):
- **ข้อดี:** ผลลัพธ์หลากหลาย ไม่ซ้ำซ้อน
- **Top K:** 5-7 สำหรับคำถามทั่วไป, 3-5 สำหรับคำถามเฉพาะ
- **Diversity Weight:** 0.2-0.4 (0.3 เป็นค่าที่ดี)

#### Similarity Search:
- **ข้อดี:** เน้นความเกี่ยวข้องสูงสุด
- **ใช้เมื่อ:** ต้องการข้อมูลที่เฉพาะเจาะจงมาก

#### Strict Mode:
- **เปิด:** ป้องกัน AI แต่งคำตอบ (แนะนำ)
- **ปิด:** ให้ AI ตอบได้แม้ไม่มีข้อมูลที่เกี่ยวข้อง

---

## 🧠 ระบบ Smart RAG ทำงานอย่างไร?

### 🔧 Header-Aware Markdown Chunking:
```markdown
# หัวข้อใหญ่
เนื้อหาส่วนที่ 1...

## หัวข้อย่อย A
เนื้อหาส่วนที่ 2...

## หัวข้อย่อย B  
เนื้อหาส่วนที่ 3...
```

**ระบบจะแยกเป็น:**
- Chunk 1: รวมหัวข้อใหญ่ + เนื้อหาส่วนที่ 1
- Chunk 2: หัวข้อใหญ่ + หัวข้อย่อย A + เนื้อหาส่วนที่ 2  
- Chunk 3: หัวข้อใหญ่ + หัวข้อย่อย B + เนื้อหาส่วนที่ 3

**ประโยชน์:**
- เก็บบริบทของหัวข้อไว้ในทุก chunk
- ค้นหาได้แม่นยำตามโครงสร้างเอกสาร
- AI เข้าใจบริบทได้ดีขึ้น

### 🎯 MMR Algorithm:
1. **หาความเกี่ยวข้อง** (Relevance) กับคำถาม
2. **วัดความหลากหลาย** (Diversity) ระหว่างผลลัพธ์  
3. **สร้างสมดุล** ระหว่างทั้งสองปัจจัย
4. **คืนผลลัพธ์** ที่เกี่ยวข้องและไม่ซ้ำซ้อน

---

## 📱 วิธีใช้แต่ละแท็บอย่างละเอียด

### 📁 Upload & OCR
**จุดประสงค์:** แปลงเอกสารเป็นข้อความ

**ขั้นตอน:**
1. อัพโหลดไฟล์ (เปิด Batch Processing สำหรับหลายไฟล์)
2. เลือก Model: Typhoon OCR 7B  
3. ตั้งค่า:
   - **Structure mode** สำหรับเอกสารที่มีตาราง
   - **High quality** สำหรับเอกสารที่มีข้อความเล็ก
4. ประมวลผลและตรวจสอบผลลัพธ์
5. กด "🧠 Add to Knowledge Base" หากต้องการ

### 🧠 Knowledge Base (แท็บสำคัญ!)
**ฟีเจอร์หลัก:**

#### การดูสถิติ:
- **Total Documents:** จำนวนเอกสารทั้งหมด
- **MD Chunks:** จำนวน chunk จาก Markdown (Smart Chunking)
- **Text Chunks:** จำนวน chunk จาก Text ธรรมดา

#### เครื่องมือจัดการ:
- **🔄 Re-index:** สร้าง embeddings ใหม่ (ใช้เมื่อเปลี่ยน embedding model)
- **🗑️ Reset KB:** ลบข้อมูลทั้งหมด (มีการยืนยัน 3 ขั้นตอน)

#### การเพิ่มเอกสาร:
- **Markdown Files:** แนะนำสำหรับเอกสารที่มีโครงสร้าง
- **Direct Input:** เลือก "markdown format" สำหรับ Smart Chunking

#### Document Browser:
- ดูรายละเอียดเอกสารแต่ละไฟล์
- แสดงจำนวน sections และ chunk type
- ลบเอกสารที่ไม่ต้องการได้

### 💬 AI Chat (แท็บหลักสำหรับใช้งาน)
**การตั้งค่าที่สำคัญ:**

#### Chat Settings:
- **Chat Model:** เลือกตาม use case
- **Use MMR Search:** เปิดเสมอ (ยกเว้นกรณีพิเศษ)
- **Strict Mode:** เปิดเพื่อป้องกัน hallucination

#### Advanced Settings:
- **Top K (5-7):** จำนวนผลลัพธ์ที่ค้นหา
- **Min Similarity (0.3):** กรองผลลัพธ์ที่ไม่เกี่ยวข้อง
- **Diversity Weight (0.3):** สมดุลระหว่างความเกี่ยวข้องและหลากหลาย

#### ปุ่มต่าง ๆ:
- **🚀 Ask:** ถามคำถามและได้คำตอบพร้อมแหล่งอ้างอิง
- **🔍 Search Only:** ดูผลการค้นหาอย่างเดียว (ไม่สร้างคำตอบ)
- **💾 Export Chat:** ส่งออกประวัติการสนทนา
- **🗑️ Clear Chat:** ล้างการสนทนา

---

## ⚡ เคล็ดลับและเทคนิคขั้นสูง

### 🎯 เพิ่มประสิทธิภาพ OCR:
1. **สแกนด้วยความละเอียดสูง** (300 DPI ขึ้นไป)
2. **ปรับแสงให้เหมาะสม** หลีกเลี่ยงเงาหรือแสงสะท้อน
3. **เอกสารตรง** ไม่เอียงหรือบิดเบี้ยว
4. **ใช้ Structure mode** สำหรับเอกสารที่มีตารางซับซ้อน
5. **แบ่งไฟล์ใหญ่** เป็นหลายหน้าสำหรับผลลัพธ์ที่ดีกว่า

### 🧠 เพิ่มประสิทธิภาพ Knowledge Base:

#### การจัดเตรียมเอกสาร Markdown:
```markdown
# เรื่องหลัก: รายงานการเงิน Q1 2024

## สรุปผลการดำเนินงาน
ยอดขายในไตรมาสแรกปี 2024 อยู่ที่ 150 ล้านบาท...

### รายได้แยกตามส่วนงาน
- ส่วนงานขาย: 100 ล้านบาท
- ส่วนงานบริการ: 50 ล้านบาท

## ค่าใช้จ่ายและกำไร
ค่าใช้จ่ายรวม 120 ล้านบาท กำไรสุทธิ 30 ล้านบาท...
```

#### การตั้งชื่อไฟล์:
- ✅ **ดี:** "รายงานการเงิน_Q1_2024.md"
- ✅ **ดี:** "คู่มือการใช้งาน_ระบบ_CRM.md"  
- ❌ **ไม่ดี:** "document1.md", "ไฟล์ใหม่.md"

#### การจัดกลุ่มเอกสาร:
- **หัวข้อเดียวกัน** ควรอยู่ในไฟล์เดียวกัน
- **แยกหัวข้อใหญ่** เป็นไฟล์แยก
- **เพิ่มบริบท** ในแต่ละส่วน

### 💬 เทคนิคการถามคำถาม:

#### คำถามที่ดี:
- **เฉพาะเจาะจง:** "มียอดขายในเดือนมกราคม 2024 เท่าไหร่?"
- **อ้างอิงเอกสาร:** "จากรายงานการเงิน Q1 มีกำไรเท่าไหร่?"
- **เปรียบเทียบ:** "เปรียบเทียบผลงาน Q1 กับ Q2 2024"
- **สรุป:** "สรุปจุดเด่นของผลิตภัณฑ์ใหม่"

#### คำถามที่ควรหลีกเลี่ยง:
- **กว้างเกินไป:** "เล่าทุกอย่างในเอกสาร"
- **ไม่เกี่ยวข้อง:** "อากาศวันนี้เป็นอย่างไร?"
- **คลุมเครือ:** "มีอะไรน่าสนใจบ้าง?"

### 🔧 การแก้ปัญหาเบื้องต้น:

#### OCR ไม่แม่นยำ:
- เพิ่ม Image Quality เป็น "high"
- ใช้ Structure mode สำหรับเอกสารซับซ้อน
- ตรวจสอบความชัดของภาพต้นฉบับ
- ลองเปลี่ยนเป็นไฟล์ภาพแทน PDF

#### ไม่พบข้อมูลในการค้นหา:
- ลด Min Similarity เหลือ 0.2 หรือ 0.1
- ปิด Strict Mode ชั่วคราว
- เพิ่ม Top K เป็น 8-10
- ตรวจสอบว่าเพิ่มเอกสารเข้า Knowledge Base แล้ว

#### AI ตอบไม่ตรงคำถาม:
- ทำให้คำถามชัดเจนและเฉพาะเจาะจงขึ้น
- เปลี่ยนจาก Qwen2.5 เป็น Typhoon2 สำหรับภาษาไทย
- ใช้ Search Only เพื่อดูว่าค้นหาเจอข้อมูลที่ถูกต้องหรือไม่
- ตรวจสอบแหล่งอ้างอิงในคำตอบ

#### ระบบช้า:
- ลด Top K เหลือ 3-5
- ใช้ Similarity Search แทน MMR สำหรับความเร็ว
- ตรวจสอบขนาดไฟล์ที่อัพโหลด (ไม่เกิน 10MB)

---

## 🚀 การใช้งานขั้นสูงและ Best Practices

### 📊 Workflow สำหรับองค์กร:

#### Week 1: Setup Phase
1. **รวบรวมเอกสารสำคัญ** ทั้งหมดขององค์กร
2. **แปลงเป็น Markdown** ด้วยโครงสร้างหัวข้อที่ชัดเจน
3. **อัพโหลดเป็นชุด** ผ่าน Knowledge Base
4. **ทดสอบการค้นหา** ด้วยคำถามตัวอย่าง

#### Ongoing: Maintenance Phase
1. **เพิ่มเอกสารใหม่** ทุกสัปดาห์
2. **ทำความสะอาด** เอกสารที่ล้าสมัย
3. **Re-index** เมื่อมีการเปลี่ยนแปลงใหญ่
4. **Export chat history** เป็น knowledge repository

### 🎛️ การปรับแต่งตาม Use Case:

#### การวิจัยและวิเคราะห์:
- **Model:** Qwen2.5 14B
- **MMR:** เปิด, Diversity Weight 0.4
- **Top K:** 7-10
- **Min Similarity:** 0.2

#### การหาข้อมูลเฉพาะ:
- **Model:** Typhoon2 8B (ไทย) หรือ Qwen2.5 (อังกฤษ)  
- **MMR:** ปิด (ใช้ Similarity)
- **Top K:** 3-5
- **Min Similarity:** 0.4

#### การสรุปและเปรียบเทียบ:
- **Model:** Qwen2.5 14B
- **MMR:** เปิด, Diversity Weight 0.3
- **Top K:** 5-7  
- **Strict Mode:** ปิด (ให้ flexibility)

### 🔄 Data Management Best Practices:

#### การตั้งชื่อและจัดกลุ่ม:
```
├── รายงานการเงิน/
│   ├── รายงานการเงิน_Q1_2024.md
│   ├── รายงานการเงิน_Q2_2024.md
│   └── วิเคราะห์แนวโน้ม_2024.md
├── นโยบายองค์กร/
│   ├── นโยบาย_HR_2024.md
│   ├── ระเบียบการจัดซื้อ.md
│   └── แนวทางการทำงาน_Remote.md
└── คู่มือเทคนิค/
    ├── คู่มือ_ระบบ_CRM.md
    ├── วิธีใช้_Analytics_Platform.md
    └── Troubleshooting_Guide.md
```

#### การสำรองและกู้คืน:
1. **Export chat history** เป็นประจำ
2. **สำรอง knowledge base** ด้วยการ export เป็น Markdown
3. **ทดสอบ re-indexing** ก่อนการใช้งานจริง
4. **เก็บ backup** ของเอกสารต้นฉบับ

---

## ❓ FAQ และการแก้ปัญหา

**Q: ฐานข้อมูลจะหายไหมถ้าปิดเบราว์เซอร์?**  
A: ไม่หาย ระบบใช้ SQLite เก็บข้อมูลแบบถาวรในเครื่อง

**Q: สามารถใช้งานออฟไลน์ได้ไหม?**  
A: ไม่ได้ ต้องเชื่อมต่ออินเทอร์เน็ตเพื่อเข้าถึง AI models

**Q: ทำไม MMR ช้ากว่า Similarity Search?**  
A: MMR ต้องคำนวณ diversity ซึ่งใช้เวลามากกว่า แต่ได้ผลลัพธ์ที่หลากหลายกว่า

**Q: Re-indexing คืออะไร และเมื่อไหร่ต้องใช้?**  
A: การสร้าง embeddings ใหม่ ใช้เมื่อ:
- เปลี่ยน embedding model
- เพิ่มเอกสารจำนวนมาก
- ผลการค้นหาไม่แม่นยำ

**Q: Strict Mode กับ Flexible Mode ต่างกันอย่างไร?**  
A: 
- **Strict:** ตอบเฉพาะจากข้อมูลในเอกสาร, ปฏิเสธถ้าไม่มีข้อมูล
- **Flexible:** AI อาจตอบจากความรู้ทั่วไปถ้าไม่เจอข้อมูล

**Q: จำกัดขนาดไฟล์เท่าไหร่?**  
A: ไฟล์เดี่ยว 10MB, ไม่จำกัดจำนวนไฟล์ แต่ประสิทธิภาพจะลดลงถ้าข้อมูลมากเกินไป

**Q: ลบข้อมูลใน Knowledge Base ได้ไหม?**  
A: ได้ ผ่าน Document Browser หรือใช้ Reset KB สำหรับลบทั้งหมด

**Q: แนะนำการจัดเก็บเอกสารอย่างไร?**  
A: 
- **ไฟล์เล็ก (<50 หน้า):** เก็บรวมกันในไฟล์เดียว
- **ไฟล์ใหญ่:** แยกตามหัวข้อหลัก
- **ใช้ Markdown** เสมอเพื่อใช้ประโยชน์จาก Smart Chunking

---

## 🎯 สรุปสำหรับผู้ใช้งานมือใหม่

### เริ่มต้นใช้งาน 5 ขั้นตอน:
1. **อัพโหลดเอกสาร** → แท็บ Upload & OCR
2. **Add to Knowledge Base** → กดปุ่มหลัง OCR เสร็จ
3. **ไปที่ AI Chat** → ตั้งค่าเป็น MMR + Strict Mode
4. **ถามคำถามง่าย ๆ** → ทดสอบระบบ
5. **ปรับแต่งตามผลลัพธ์** → เพิ่ม/ลดความเคร่งครัด

### การใช้งานขั้นสูง:
1. **เตรียมไฟล์ Markdown** → โครงสร้างหัวข้อชัดเจน
2. **อัพโหลดผ่าน Knowledge Base** → ได้ Smart Chunking
3. **ปรับแต่ง Search Parameters** → ตาม use case
4. **ใช้ Search Only** → ตรวจสอบก่อนถามคำถาบ
5. **Export Chat History** → เก็บเป็น knowledge base

ระบบนี้เหมาะสำหรับ:
- ✅ **การค้นหาข้อมูลในเอกสาร**
- ✅ **การวิเคราะห์และสรุปเนื้อหา**  
- ✅ **การเปรียบเทียบข้อมูลข้ามเอกสาร**
- ✅ **การสร้าง knowledge base ขององค์กร**

        """)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.error(f"Debug info: {traceback.format_exc()}")Page {page['page_number']}\n\n"
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
        <p>AI-Powered Thai-English Document Parser + Smart Knowledge Q&A</p>
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
                st.metric("📄 Total Chunks", stats["total_chunks"])
            with col2:
                st.metric("📝 MD Chunks", stats.get("markdown_chunks", 0))
                st.metric("📋 Text Chunks", stats.get("text_chunks", 0))
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
                st.metric("📝 Markdown Chunks", stats.get("markdown_chunks", 0))
            with col4:
                st.metric("💬 Chat Sessions", stats["total_chat_sessions"])
        
        # Knowledge Base Actions
        st.markdown("---")
        st.subheader("🔧 Knowledge Base Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Re-index All Documents", help="Recreate embeddings for all documents"):
                if stats.get("total_documents", 0) > 0:
                    with st.spinner("Re-indexing documents..."):
                        if kb.reindex_documents():
                            st.success("✅ Successfully re-indexed all documents!")
                            st.rerun()
                else:
                    st.info("ℹ️ No documents to re-index")
        
        with col2:
            if st.button("🗑️ Reset Knowledge Base", help="Delete ALL documents and chat history"):
                # Double confirmation system
                if 'reset_confirm_step' not in st.session_state:
                    st.session_state.reset_confirm_step = 0
                
                if st.session_state.reset_confirm_step == 0:
                    st.session_state.reset_confirm_step = 1
                    st.warning("⚠️ First confirmation: Click again to proceed to final confirmation.")
                elif st.session_state.reset_confirm_step == 1:
                    st.session_state.reset_confirm_step = 2
                    st.error("🚨 FINAL WARNING: This will DELETE ALL data! Click once more to confirm.")
                else:
                    # Final confirmation - actually reset
                    with st.spinner("Resetting knowledge base..."):
                        if kb.reset_knowledge_base():
                            st.success("✅ Knowledge base has been completely reset!")
                            st.session_state.chat_history = []  # Clear chat history
                            st.session_state.reset_confirm_step = 0  # Reset confirmation
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset knowledge base")
                            st.session_state.reset_confirm_step = 0
        
        st.markdown("---")
        
        # Add documents manually
        st.subheader("📤 Add Documents to Knowledge Base")
        
        # Method 1: Upload Markdown files
        st.write("**Method 1: Upload Markdown Files (Smart Chunking)**")
        st.info("📝 Markdown files will be processed with header-aware chunking for better context preservation.")
        
        markdown_files = st.file_uploader(
            "Upload Markdown files",
            type=['md', 'txt'],
            accept_multiple_files=True,
            help="Upload .md or .txt files - .md files get smart header-based chunking"
        )
        
        if markdown_files and st.button("➕ Add Markdown Files"):
            added_count = 0
            progress_bar = st.progress(0)
            
            for i, file in enumerate(markdown_files):
                progress_bar.progress((i + 1) / len(markdown_files))
                try:
                    content = file.read().decode('utf-8')
                    file_type = "markdown" if file.name.endswith('.md') else "text"
                    
                    if kb.add_document(
                        filename=file.name,
                        content=content,
                        metadata={
                            "source": "manual_upload", 
                            "file_type": file_type,
                            "upload_method": "file_upload"
                        }
                    ):
                        added_count += 1
                except Exception as e:
                    st.error(f"❌ Error adding {file.name}: {str(e)}")
            
            if added_count > 0:
                st.success(f"✅ Added {added_count} files to knowledge base with smart chunking!")
                st.rerun()
        
        # Method 2: Direct text input
        st.write("**Method 2: Direct Text Input**")
        with st.form("add_text_form"):
            doc_title = st.text_input("Document Title", placeholder="Enter document title")
            doc_content = st.text_area(
                "Document Content",
                height=200,
                placeholder="Paste or type your document content here... (Use Markdown format for better structure)"
            )
            
            # Choose file format
            content_format = st.selectbox(
                "Content Format:",
                ["markdown", "text"],
                help="Markdown format enables smart header-based chunking"
            )
            
            submitted = st.form_submit_button("➕ Add to Knowledge Base")
            if submitted and doc_title and doc_content:
                filename = f"{doc_title}.{content_format}"
                if kb.add_document(
                    filename=filename,
                    content=doc_content,
                    metadata={
                        "source": "manual_input",
                        "file_type": content_format,
                        "upload_method": "direct_input"
                    }
                ):
                    st.success(f"✅ Added '{doc_title}' to knowledge base!")
                    st.rerun()
        
        st.markdown("---")
        
        # Enhanced knowledge base browser
        st.subheader("📖 Browse Knowledge Base")
        
        if stats.get("total_documents", 0) > 0:
            # Get document list with enhanced info
            conn = sqlite3.connect(st.session_state.rag_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    filename, 
                    COUNT(*) as chunk_count, 
                    MAX(created_at) as latest,
                    AVG(header_level) as avg_header_level,
                    COUNT(DISTINCT section_title) as sections
                FROM documents 
                GROUP BY filename 
                ORDER BY latest DESC
            """)
            docs = cursor.fetchall()
            conn.close()
            
            # Display documents with enhanced info
            for filename, chunk_count, created_at, avg_header_level, sections in docs:
                chunk_type = "📝 MD" if filename.endswith('.md') else "📋 TXT"
                with st.expander(f"{chunk_type} {filename} ({chunk_count} chunks, {sections} sections)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Created:** {created_at}")
                        st.write(f"**Chunks:** {chunk_count}")
                        st.write(f"**Sections:** {sections}")
                    with col2:
                        if avg_header_level and avg_header_level > 0:
                            st.write(f"**Avg Header Level:** {avg_header_level:.1f}")
                        st.write(f"**Type:** {'Markdown' if filename.endswith('.md') else 'Plain Text'}")
                    
                    # Show first chunk as preview
                    conn = sqlite3.connect(st.session_state.rag_db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, section_title 
                        FROM documents 
                        WHERE filename = ? 
                        ORDER BY chunk_id 
                        LIMIT 1
                    """, (filename,))
                    preview = cursor.fetchone()
                    if preview:
                        content, section_title = preview
                        st.write("**Preview:**")
                        if section_title:
                            st.write(f"*Section: {section_title}*")
                        st.text(content[:300] + "..." if len(content) > 300 else content)
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

    # ---- Tab 3: Enhanced AI Chat with RAG ----
    with tab3:
        st.header("💬 Enhanced AI Chat with Knowledge Base")
        
        # Check if knowledge base has content
        if stats.get("total_documents", 0) == 0:
            st.warning("⚠️ No documents in knowledge base. Please add documents in the Knowledge Base tab first.")
            return
        
        # Enhanced chat settings
        st.subheader("⚙️ Chat Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chat_model = st.selectbox(
                "Chat Model:",
                options=[k for k in AVAILABLE_MODELS.keys() if k != "scb10x/typhoon-ocr-7b:latest"],
                index=0,
                format_func=lambda x: f"{AVAILABLE_MODELS[x]['icon']} {AVAILABLE_MODELS[x]['name']}",
                help="Choose model for answering questions"
            )
        
        with col2:
            use_mmr = st.checkbox(
                "Use MMR Search", 
                value=True,
                help="Maximal Marginal Relevance for diverse, relevant results"
            )
        
        with col3:
            strict_mode = st.checkbox(
                "Strict Mode", 
                value=True,
                help="Refuse to answer if no relevant information found"
            )
        
        # Advanced search settings
        with st.expander("🔧 Advanced Search Settings"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Results to retrieve", 3, 10, 5)
                min_similarity = st.slider("Min similarity threshold", 0.1, 0.8, 0.3, 0.1)
            with col2:
                diversity_weight = st.slider("MMR diversity weight", 0.0, 1.0, 0.3, 0.1)
                st.write(f"**Current Settings:**")
                st.write(f"- Model: {AVAILABLE_MODELS[chat_model]['name']}")
                st.write(f"- Search: {'MMR' if use_mmr else 'Similarity'}")
                st.write(f"- Mode: {'Strict' if strict_mode else 'Flexible'}")
        
        st.markdown("---")
        
        # Chat interface
        st.subheader("🗣️ Ask Questions About Your Documents")
        
        # Display chat history with enhanced formatting
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
                <strong>🤖 Assistant ({AVAILABLE_MODELS.get(chat_model, {}).get('name', 'Unknown')}):</strong><br>
                {answer.replace('📚 **แหล่งอ้างอิง:**', '<br><strong>📚 แหล่งอ้างอิง:</strong>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Show context sources with enhanced display
            if context:
                with st.expander(f"📚 Source Context (Chat {i+1})", expanded=False):
                    for j, (filename, content, similarity, section_title) in enumerate(context):
                        st.markdown(f"""
                        <div class="source-reference">
                            <strong>📄 Source {j+1}: {filename}</strong><br>
                            <em>Section: {section_title if section_title else 'N/A'} | Relevance: {similarity:.1%}</em><br>
                            <small>{content[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Enhanced chat input
        with st.form("enhanced_chat_form", clear_on_submit=True):
            user_question = st.text_area(
                "Ask a question:",
                placeholder="What would you like to know about your documents? \n\nTips:\n- Be specific about what you're looking for\n- Mention document names if you want to focus on specific files\n- Ask for comparisons or summaries across documents",
                height=100,
                key="chat_input"
            )
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            with col1:
                ask_button = st.form_submit_button("🚀 Ask Question", type="primary")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
            with col3:
                export_button = st.form_submit_button("💾 Export Chat")
            with col4:
                search_only = st.form_submit_button("🔍 Search Only")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            if export_button and st.session_state.chat_history:
                # Export chat history
                chat_export = "# Chat History Export\n\n"
                for i, (q, a, ctx) in enumerate(st.session_state.chat_history):
                    chat_export += f"## Question {i+1}\n**Q:** {q}\n\n**A:** {a}\n\n"
                    if ctx:
                        chat_export += "**Sources:**\n"
                        for j, (fname, content, sim, section) in enumerate(ctx):
                            chat_export += f"- {fname} ({section or 'N/A'}) - {sim:.1%}\n"
                        chat_export += "\n"
                
                st.download_button(
                    "📥 Download Chat History",
                    chat_export,
                    "chat_history.md",
                    "text/markdown"
                )
        
        if search_only and user_question:
            # Search only mode - show results without generating answer
            with st.spinner("🔍 Searching knowledge base..."):
                if use_mmr:
                    # Use enhanced search method directly
                    context_docs = kb.search_similar(
                        user_question, 
                        top_k=top_k, 
                        use_mmr=True, 
                        min_similarity=min_similarity
                    )
                else:
                    context_docs = kb.search_similar(
                        user_question, 
                        top_k=top_k, 
                        use_mmr=False, 
                        min_similarity=min_similarity
                    )
                
                if context_docs:
                    st.success(f"✅ Found {len(context_docs)} relevant results:")
                    for i, (filename, content, similarity, section_title) in enumerate(context_docs):
                        st.markdown(f"""
                        <div class="knowledge-card">
                            <strong>📄 Result {i+1}: {filename}</strong><br>
                            <em>Section: {section_title or 'N/A'} | Relevance: {similarity:.1%}</em><br>
                            <small>{content[:300]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No relevant information found in knowledge base")
        
        elif ask_button and user_question:
            with st.spinner("🔍 Searching knowledge base and generating response..."):
                # Enhanced search with user settings
                context_docs = kb.search_similar(
                    user_question, 
                    top_k=top_k, 
                    use_mmr=use_mmr, 
                    min_similarity=min_similarity
                )
                
                if context_docs or not strict_mode:
                    # Generate RAG response with selected model
                    response = generate_rag_response(
                        user_question, 
                        context_docs, 
                        chat_model,
                        strict_mode=strict_mode
                    )
                    
                    if response:
                        # Add to chat history with enhanced context info
                        enhanced_context = [
                            (filename, content, similarity, section_title)
                            for filename, content, similarity, section_title in context_docs[:3]
                        ]
                        
                        st.session_state.chat_history.append((
                            user_question,
                            response,
                            enhanced_context
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
                            json.dumps({
                                "context": [(doc[0], doc[1][:200], doc[2], doc[3]) for doc in enhanced_context],
                                "model": chat_model,
                                "settings": {
                                    "use_mmr": use_mmr,
                                    "strict_mode": strict_mode,
                                    "top_k": top_k,
                                    "min_similarity": min_similarity
                                }
                            })
                        ))
                        conn.commit()
                        conn.close()
                        
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate response")
                else:
                    st.warning("⚠️ No relevant information found in knowledge base (try lowering similarity threshold or disabling strict mode)")

    # ---- Tab 4: Features ----
    with tab4:
        st.header("✨ Enhanced Features with Smart RAG")
        
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
                "title": "Smart Knowledge Base with MMR",
                "description": "สร้างฐานความรู้อัจฉริยะพร้อมการค้นหาแบบ MMR",
                "items": [
                    "Auto-import OCR results to knowledge base",
                    "Smart markdown chunking with header awareness",
                    "MMR search for balanced relevance and diversity",
                    "Configurable similarity thresholds and strict mode"
                ]
            },
            {
                "icon": "💬",
                "title": "Enhanced AI Q&A System",
                "description": "ถามตอบกับเอกสารด้วย AI ที่ฉลาดขึ้น",
                "items": [
                    "Context-aware responses with source attribution",
                    "Multiple model support (Typhoon2, Qwen2.5, Nomic)",
                    "Strict mode prevents hallucination",
                    "Enhanced chat history with export functionality"
                ]
            },
            {
                "icon": "🔧",
                "title": "Advanced Management Tools", 
                "description": "เครื่องมือจัดการขั้นสูงสำหรับฐานความรู้",
                "items": [
                    "Re-indexing with updated embeddings",
                    "Safe knowledge base reset with double confirmation",
                    "Document browser with section tracking",
                    "Performance metrics and chunk type analysis"
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
        
        # Enhanced workflow diagram
        st.subheader("🔄 Smart RAG Workflow")
        st.markdown("""
        ```mermaid
        graph TD
            A[📄 Upload Document] --> B[🌪️ OCR Processing]
            B --> C[📝 Extract Text/Markdown]
            C --> D{Add to Knowledge Base?}
            D -->|Yes| E[🧠 Smart Chunking]
            E --> F[Header-Aware MD] --> G[🔗 Generate Embeddings]
            E --> H[Regular Text] --> G
            G --> I[💾 Store in RAG DB]
            D -->|No| J[💾 Download Results]
            
            K[📤 Manual MD Upload] --> E
            L[✍️ Direct Input] --> E
            
            I --> M[💬 Enhanced Chat]
            M --> N{Search Method}
            N -->|MMR| O[🎯 MMR Search]
            N -->|Similarity| P[📊 Cosine Search]
            O --> Q[🤖 Generate Answer]
            P --> Q
            Q --> R{Strict Mode?}
            R -->|Yes| S[✅ Answer with Sources]
            R -->|No| T[🔄 Flexible Response]
        ```
        """)
        
        # Enhanced model comparison
        st.subheader("⚡ Enhanced Model Performance")
        
        performance_data = {
            "Model": ["Typhoon OCR 7B", "Qwen2.5 14B", "Typhoon2 8B", "Nomic Embed"],
            "Thai OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "N/A"],
            "English OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "N/A"],
            "Complex Tables": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "N/A"],
            "Thai Q&A": ["⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "N/A"],
            "Processing Speed": ["⭐⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            "Best For": ["OCR Tasks", "Complex Reasoning", "Thai Content", "Embeddings"]
        }
        
        st.table(performance_data)
        
        # Smart RAG benefits
        st.subheader("🎯 Smart RAG System Benefits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Intelligent Processing:**
            - Header-aware markdown chunking
            - Context preservation across chunks
            - Smart overlap for continuity
            - Section-based organization
            
            **🔍 Advanced Search:**
            - MMR for relevance + diversity
            - Configurable similarity thresholds
            - Multi-model support
            - Source attribution with sections
            """)
        
        with col2:
            st.markdown("""
            **⚙️ Enhanced Management:**
            - Re-indexing capabilities
            - Safe reset with confirmations
            - Document type tracking
            - Performance monitoring
            
            **🎯 Quality Assurance:**
            - Strict mode prevents hallucination
            - Source transparency
            - Chat history export
            - Embedding version control
            """)
        
        # Technical specifications
        st.subheader("🔬 Technical Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Chunking Strategy:**
            - Markdown: Header-aware, 1500 char max
            - Text: Sentence-boundary, 1000 char + 200 overlap
            - Context preservation with header hierarchy
            - Smart overlap for continuity
            """)
        
        with col2:
            st.markdown("""
            **Search Algorithm:**
            - MMR with configurable diversity (default: 0.3)
            - Cosine similarity baseline
            - Minimum similarity filtering (default: 0.3)
            - Top-K retrieval (configurable: 3-10)
            """)

    # ---- Tab 5: Comprehensive User Guide ----
    with tab5:
        st.header("📖 คู่มือการใช้งาน Typhoon OCR with Enhanced RAG")
        
        st.markdown("""
## 🌟 ภาพรวมระบบที่ปรับปรุงใหม่

Typhoon OCR with Enhanced RAG เป็นระบบที่รวม **OCR ขั้นสูง** กับ **Smart RAG System** ที่มีความสามารถ:

1. **📄 ประมวลผลเอกสาร** - แปลง PDF/รูปภาพเป็นข้อความด้วย AI
2. **🧠 ฐานความรู้อัจฉริยะ** - จัดเก็บและจัดระเบียบข้อมูลแบบ semantic
3. **💬 ถามตอบขั้นสูง** - ค้นหาและตอบคำถามด้วย MMR algorithm
4. **🔧 จัดการขั้นสูง** - เครื่องมือสำหรับจัดการและดูแลระบบ

---

## 
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
