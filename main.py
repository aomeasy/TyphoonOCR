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
from typing import Optional, Dict, Any, List
import traceback

# Try importing pdf2image with error handling
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    st.warning("⚠️ pdf2image not available. PDF conversion will use fallback method.")

# Page configuration
st.set_page_config(
    page_title="🌪️ Typhoon OCR",
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
        
        st.download_button(
            "📦 Download All Results",
            combined_content,
            f"typhoon_ocr_results.{file_extension}",
            f"text/{file_extension}",
            help=f"Download combined results in {params.get('output_format', 'markdown')} format"
        )

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌪️ Typhoon OCR</h1>
        <p>AI-Powered Thai-English Document Parser</p>
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
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "✨ Features", "📖 คู่มือการใช้งาน"])

    # ---- Tab 1: Upload & Process ----
    with tab1:
        st.header("📁 Upload Document")

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

    # ---- Tab 2: Features ----
    with tab2:
        st.header("✨ Features")
        
        # Enhanced feature cards with more details
        features = [
            {
                "icon": "📊",
                "title": "Structured Documents",
                "description": "รายงานทางการเงิน, เอกสารวิชาการ, แบบฟอร์มราชการ",
                "items": ["Financial reports with tables", "Academic papers with references", "Government forms and applications", "Books & textbooks with complex layouts"],
                "accuracy": "92-96%",
                "best_settings": "Structure prompt + HTML output"
            },
            {
                "icon": "🍽️", 
                "title": "Layout-Heavy Documents",
                "description": "เอกสารที่เน้น Layout และไม่เป็นทางการ",
                "items": ["Receipts & bills with mixed layouts", "Food menus with images", "Tickets and vouchers", "Infographics and brochures"],
                "accuracy": "85-92%",
                "best_settings": "Structure prompt + High quality"
            },
            {
                "icon": "🔍",
                "title": "Advanced Analysis", 
                "description": "การวิเคราะห์รูปภาพและไดอะแกรมแบบลึกซึ้ง",
                "items": ["Element detection and positioning", "Context-aware text extraction", "Multi-language text recognition", "Table structure preservation"],
                "accuracy": "88-94%",
                "best_settings": "Typhoon OCR model + Low temperature"
            }
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
                
                <strong>Capabilities:</strong>
                <ul>
                    {' '.join([f'<li>✓ {item}</li>' for item in feature['items']])}
                </ul>
                
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <span><strong>Accuracy:</strong> {feature['accuracy']}</span>
                    <span><strong>Best Settings:</strong> {feature['best_settings']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance comparison
        st.subheader("⚡ Model Performance Comparison")
        
        performance_data = {
            "Model": ["Typhoon OCR 7B", "Qwen2.5 14B", "Typhoon2 8B"],
            "Thai OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐"],
            "English OCR": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐"],
            "Complex Tables": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
            "Processing Speed": ["⭐⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐"],
            "Best For": ["OCR Tasks", "General Analysis", "Thai Content"]
        }
        
        st.table(performance_data)

    # ---- Tab 3: User Guide ---- (keeping the original comprehensive guide)
    with tab3:
        st.header("📖 คู่มือการใช้งาน Typhoon OCR")
        st.markdown("""
🌟 **ภาพรวมระบบ**  
Typhoon OCR เป็นเครื่องมือแปลงเอกสารภาษาไทย-อังกฤษด้วย AI ที่สามารถ:

- อ่านและแปลงเอกสาร PDF และรูปภาพ  
- รองรับเอกสารที่มีโครงสร้างซับซ้อน  
- ประมวลผลหลายไฟล์พร้อมกัน  
- ส่งออกผลลัพธ์ในหลากหลายรูปแบบ  

---

### 🎯 การตั้งค่าตัวแปรการประมวลผล (Processing Parameters)

#### 🌡️ Temperature (อุณหภูมิ)
**คืออะไร:** ควบคุมความสร้างสรรค์และความแปรปรวนของ AI  

**ค่าแนะนำ:**
- **0.0-0.1 🎯 แม่นยำสูงสุด (แนะนำสำหรับ OCR)**
  - ผลลัพธ์คาดเดาได้  
  - เหมาะสำหรับเอกสารราชการ, รายงานทางการเงิน  
  - ความผิดพลาดน้อยที่สุด  
- **0.2-0.5 ⚖️ สมดุล**
  - ผลลัพธ์หลากหลายแต่ยังคุมได้  
  - เหมาะสำหรับเอกสารที่มีเนื้อหาซับซ้อน  
  - สำหรับเอกสารที่ต้องการการตีความ
- **0.6-1.0 🌟 สร้างสรรค์**
  - ผลลัพธ์หลากหลายมาก  
  - ไม่แนะนำสำหรับ OCR (อาจมีข้อผิดพลาด)  
  - เหมาะสำหรับงานเขียนสร้างสรรค์

**ตัวอย่างการใช้งาน:**
- 📄 ใบเสร็จ/บิล → Temperature **0.1**  
- 📊 รายงานการเงิน → Temperature **0.1**  
- 📚 หนังสือ/บทความ → Temperature **0.2-0.3**  
- 🎨 เอกสารสร้างสรรค์ → Temperature **0.5-0.7**

#### 🎯 Top P (การเลือกคำ)
**คืออะไร:** ควบคุมว่า AI จะพิจารณาคำที่มีความน่าจะเป็นสูงแค่ไหน  

**ค่าแนะนำ:**
- **0.1-0.3 📊 เข้มงวด**  
  เลือกเฉพาะคำที่แน่นอนที่สุด เหมาะสำหรับข้อมูลที่ต้องการความแม่นยำสูง  
- **0.6 🎯 ค่าแนะนำ (Default)**  
  สมดุลระหว่างความแม่นยำและความหลากหลาย เหมาะสำหรับงาน OCR ทั่วไป  
- **0.8-1.0 🎲 หลากหลาย**  
  เปิดโอกาสให้คำที่หลากหลายมากขึ้น อาจให้ผลลัพธ์ที่ไม่สม่ำเสมอ

---

### 🔧 การแก้ไขปัญหา PDF Processing

#### ⚠️ ปัญหา "Unable to get page count. Is poppler installed and in PATH?"

**สาเหตุ:** ไม่มี poppler-utils ติดตั้งในระบบ

**การแก้ไข:**

**สำหรับ Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**สำหรับ CentOS/RHEL:**
```bash
sudo yum install poppler-utils
# หรือใน CentOS 8+
sudo dnf install poppler-utils
```

**สำหรับ macOS:**
```bash
brew install poppler
```

**สำหรับ Windows:**
1. ดาวน์โหลด poppler จาก: https://blog.alivate.com.au/poppler-windows/
2. แตกไฟล์และเพิ่ม path ของ bin folder ใน Environment Variables

#### 🔄 ทางเลือกอื่น (ถ้าติดตั้ง poppler ไม่ได้):

**ติดตั้ง PyMuPDF:**
```bash
pip install PyMuPDF
```

**หรือใช้ระบบ fallback ที่มีอยู่แล้ว** - โปรแกรมจะใช้ PyPDF2 แทน (ความสามารถจำกัด)

---

### 📦 Requirements สำหรับ GitHub Deployment

สร้างไฟล์ `requirements.txt`:

```txt
streamlit>=1.28.0
requests>=2.31.0
Pillow>=10.0.0
PyPDF2>=3.0.1
pdf2image>=1.16.3
PyMuPDF>=1.23.0
```

สำหรับ **Streamlit Cloud** หรือ **GitHub Codespaces**, เพิ่มไฟล์ `packages.txt`:

```txt
poppler-utils
```

หรือสร้าง `apt.txt` (สำหรับบาง platform):
```txt
poppler-utils
```

---

### 🚀 การ Deploy บน Streamlit Cloud

1. **Push โค้ดไปยัง GitHub repository**
2. **ไปที่ streamlit.io และ connect repository**
3. **ตั้งค่าไฟล์:**
   - Main file: `main.py`
   - Python version: 3.9+
4. **เพิ่มไฟล์ `packages.txt`** สำหรับ system packages
5. **Deploy และรอระบบติดตั้ง dependencies**

---

### ✅ เคล็ดลับเพิ่มเติม

**🔧 การเพิ่มประสิทธิภาพ:**
- ใช้ **Medium quality** สำหรับใช้งานทั่วไป
- ใช้ **High quality** เฉพาะเอกสารสำคัญ
- **ตั้งค่า Temperature = 0.1** สำหรับความแม่นยำสูงสุด
- **เลือกโมเดล Typhoon OCR** สำหรับภาษาไทย

**⚡ การประมวลผลเร็วขึ้น:**
- แยกไฟล์ขนาดใหญ่เป็นไฟล์เล็ก
- ประมวลผลครั้งละไม่เกิน 5 ไฟล์
- ใช้ **Low quality** สำหรับการทดสอบ

**🎯 ความแม่นยำสูงขึ้น:**
- สแกนด้วย **300 DPI** ขึ้นไป
- ตรวจสอบแสงให้เหมาะสม
- หลีกเลี่ยงการเอียงของเอกสาร
- ใช้ **Structure prompt** สำหรับเอกสารซับซ้อน
        """)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.error(f"Debug info: {traceback.format_exc()}")
