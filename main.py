import streamlit as st
import requests
import base64
import json
from io import BytesIO
from PIL import Image
import PyPDF2
import pdf2image
import tempfile
import os
import subprocess
import sys
from typing import Optional, Dict, Any
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Check if required system dependencies are installed"""
    dependencies = {
        'poppler': False,
        'pdf2image': True,  # This is a Python package, assumed to be installed
        'PyPDF2': True      # This is a Python package, assumed to be installed
    }
    
    # Check poppler
    try:
        result = subprocess.run(['pdfinfo', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            dependencies['poppler'] = True
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            # Try alternative poppler command
            result = subprocess.run(['pdftoppm', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                dependencies['poppler'] = True
        except (subprocess.SubprocessError, FileNotFoundError):
            dependencies['poppler'] = False
    
    return dependencies

def display_system_status():
    """Display system dependency status"""
    deps = check_system_dependencies()
    
    if not deps['poppler']:
        st.warning("""
        ⚠️ **Poppler ไม่ได้ติดตั้ง**: การแปลง PDF อาจไม่ทำงาน
        
        **วิธีแก้ไข:**
        
        **Ubuntu/Debian:**
        ```bash
        sudo apt-get update
        sudo apt-get install poppler-utils
        ```
        
        **CentOS/RHEL:**
        ```bash
        sudo yum install poppler-utils
        # หรือ
        sudo dnf install poppler-utils
        ```
        
        **macOS:**
        ```bash
        brew install poppler
        ```
        
        **Windows:**
        1. ดาวน์โหลด poppler จาก: https://github.com/oschwartz10612/poppler-windows/releases/
        2. แตกไฟล์และเพิ่ม bin folder ใน PATH
        
        **Docker:**
        ```dockerfile
        RUN apt-get update && apt-get install -y poppler-utils
        ```
        """)
        return False
    else:
        st.success("✅ ระบบพร้อมใช้งาน - Poppler ติดตั้งแล้ว")
        return True

def safe_convert_pdf_to_images(pdf_file, quality="high") -> tuple[list, str]:
    """
    Safely convert PDF to images with comprehensive error handling
    Returns: (images_list, error_message)
    """
    images = []
    error_msg = ""
    
    try:
        # Check if poppler is available
        deps = check_system_dependencies()
        if not deps['poppler']:
            return [], "Poppler is not installed. Please install poppler-utils to process PDF files."
        
        # Validate file size (limit to 50MB)
        file_size = len(pdf_file.getvalue()) if hasattr(pdf_file, 'getvalue') else 0
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return [], f"File too large ({file_size / (1024*1024):.1f}MB). Maximum size is 50MB."
        
        # Reset file pointer
        pdf_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            try:
                # Write file content
                content = pdf_file.read()
                if not content:
                    return [], "PDF file is empty or corrupted."
                
                tmp_file.write(content)
                tmp_file.flush()
                
                # Verify it's a valid PDF using PyPDF2
                pdf_file.seek(0)
                try:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(pdf_reader.pages)
                    if num_pages == 0:
                        return [], "PDF has no pages."
                    if num_pages > 50:  # Limit pages
                        return [], f"PDF has too many pages ({num_pages}). Maximum is 50 pages."
                        
                except Exception as e:
                    return [], f"Invalid PDF file: {str(e)}"
                
                # Set DPI based on quality
                dpi_mapping = {"high": 300, "medium": 200, "low": 150}
                dpi = dpi_mapping.get(quality, 200)
                
                # Convert PDF to images with timeout and error handling
                try:
                    images = pdf2image.convert_from_path(
                        tmp_file.name, 
                        dpi=dpi,
                        first_page=1,
                        last_page=min(50, num_pages),  # Limit to 50 pages
                        timeout=300,  # 5 minute timeout
                        poppler_path=None,  # Use system poppler
                        thread_count=1  # Single thread to avoid issues
                    )
                    
                    if not images:
                        return [], "Failed to convert PDF pages to images."
                        
                    logger.info(f"Successfully converted {len(images)} pages from PDF")
                    
                except Exception as conv_error:
                    error_details = str(conv_error)
                    if "poppler" in error_details.lower():
                        return [], f"Poppler error: {error_details}. Please ensure poppler is properly installed."
                    elif "timeout" in error_details.lower():
                        return [], "PDF conversion timed out. File may be too large or complex."
                    else:
                        return [], f"PDF conversion failed: {error_details}"
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass  # File already deleted or doesn't exist
                    
    except Exception as e:
        error_msg = f"Unexpected error during PDF processing: {str(e)}"
        logger.error(f"PDF conversion error: {error_msg}")
        logger.error(traceback.format_exc())
        return [], error_msg
    
    return images, ""

def validate_image_file(image_file) -> tuple[Optional[Image.Image], str]:
    """
    Validate and load image file
    Returns: (image_object, error_message)
    """
    try:
        # Check file size (limit to 20MB)
        file_size = len(image_file.getvalue()) if hasattr(image_file, 'getvalue') else 0
        if file_size > 20 * 1024 * 1024:  # 20MB limit
            return None, f"Image file too large ({file_size / (1024*1024):.1f}MB). Maximum size is 20MB."
        
        # Reset file pointer
        image_file.seek(0)
        
        # Try to open and validate image
        try:
            image = Image.open(image_file)
            # Validate image
            image.verify()
            
            # Reopen image (verify() closes the file)
            image_file.seek(0)
            image = Image.open(image_file)
            
            # Check image dimensions
            width, height = image.size
            if width < 50 or height < 50:
                return None, f"Image too small ({width}x{height}). Minimum size is 50x50 pixels."
            if width > 10000 or height > 10000:
                return None, f"Image too large ({width}x{height}). Maximum size is 10000x10000 pixels."
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'RGBA', 'L'):
                image = image.convert('RGB')
            
            logger.info(f"Successfully loaded image: {width}x{height}, mode: {image.mode}")
            return image, ""
            
        except Exception as img_error:
            return None, f"Invalid image file: {str(img_error)}"
            
    except Exception as e:
        return None, f"Error validating image: {str(e)}"

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string with error handling"""
    try:
        buffer = BytesIO()
        # Optimize image format and quality
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Save with optimized settings
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        
        # Check base64 size (limit to 10MB encoded)
        if len(encoded) > 10 * 1024 * 1024:
            # Resize image if too large
            image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=70, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode()
        
        return encoded
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

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

def call_ollama_api(model: str, prompt: str, image_base64: str, params: dict) -> tuple[Optional[str], str]:
    """
    Call Ollama API for OCR processing with comprehensive error handling
    Returns: (response_text, error_message)
    """
    try:
        # Validate inputs
        if not model or not prompt or not image_base64:
            return None, "Missing required parameters (model, prompt, or image)"
        
        if len(image_base64) > 15 * 1024 * 1024:  # 15MB limit
            return None, "Image too large for API processing"
        
        # For OCR models, we need to use chat completion format
        if "typhoon-ocr" in model:
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "temperature": max(0.0, min(1.0, params.get('temperature', 0.1))),
                "top_p": max(0.1, min(1.0, params.get('top_p', 0.6))),
                "num_predict": max(100, min(16384, params.get('max_tokens', 12000))),
                "stream": False,
                "options": {
                    "repeat_penalty": max(1.0, min(2.0, params.get('repetition_penalty', 1.2))),
                    "stop": ["<|im_end|>", "</response>"]
                }
            }
        else:
            # For other models, use standard format
            payload = {
                "model": model,
                "prompt": f"Analyze this image and extract all text content. Return the result in markdown format.\n\nImage: data:image/png;base64,{image_base64}\n\n{prompt}",
                "temperature": max(0.0, min(1.0, params.get('temperature', 0.1))),
                "top_p": max(0.1, min(1.0, params.get('top_p', 0.6))),
                "num_predict": max(100, min(16384, params.get('max_tokens', 12000))),
                "stream": False,
                "options": {
                    "repeat_penalty": max(1.0, min(2.0, params.get('repetition_penalty', 1.2)))
                }
            }
        
        # Make API request with proper timeout and error handling
        try:
            response = requests.post(
                OLLAMA_API_URL, 
                json=payload, 
                timeout=300,  # 5 minute timeout
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            # Check response status
            if response.status_code == 404:
                return None, f"Model '{model}' not found. Please check if the model is available."
            elif response.status_code == 400:
                return None, f"Bad request to API. Please check your parameters."
            elif response.status_code == 500:
                return None, "Internal server error. The API service may be down."
            elif response.status_code != 200:
                return None, f"API returned status code {response.status_code}: {response.text}"
            
            # Parse response
            try:
                result = response.json()
                api_response = result.get('response', '')
                
                if not api_response:
                    return None, "API returned empty response"
                
                # Check for API-level errors
                if 'error' in result:
                    return None, f"API error: {result['error']}"
                
                logger.info(f"Successfully got response from {model}, length: {len(api_response)}")
                return api_response, ""
                
            except json.JSONDecodeError as je:
                return None, f"Invalid JSON response from API: {str(je)}"
                
        except requests.exceptions.Timeout:
            return None, "Request timed out. The model may be processing a complex image."
        except requests.exceptions.ConnectionError:
            return None, "Cannot connect to API server. Please check if the server is running."
        except requests.exceptions.RequestException as re:
            return None, f"Network error: {str(re)}"
        
    except Exception as e:
        error_msg = f"Unexpected error calling API: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

def process_single_document(file, model: str, params: dict) -> Dict[str, Any]:
    """Process a single document with comprehensive error handling"""
    results = {
        'filename': file.name,
        'pages': [],
        'total_pages': 0,
        'success': False,
        'error': None,
        'warnings': []
    }
    
    try:
        # Validate file
        if not file.name or not hasattr(file, 'type'):
            results['error'] = "Invalid file object"
            return results
        
        # Handle different file types
        images = []
        if file.type == 'application/pdf':
            images, error_msg = safe_convert_pdf_to_images(file, params.get('image_quality', 'high'))
            if error_msg:
                results['error'] = error_msg
                return results
            results['total_pages'] = len(images)
        else:
            # Image file
            image, error_msg = validate_image_file(file)
            if error_msg:
                results['error'] = error_msg
                return results
            images = [image]
            results['total_pages'] = 1
        
        if not images:
            results['error'] = "No images to process"
            return results
        
        # Limit number of pages for performance
        max_pages = 20
        if len(images) > max_pages:
            results['warnings'].append(f"Processing only first {max_pages} pages out of {len(images)}")
            images = images[:max_pages]
            results['total_pages'] = max_pages
        
        # Process each page/image
        successful_pages = 0
        for i, image in enumerate(images):
            page_result = {
                'page_number': i + 1,
                'success': False,
                'content': '',
                'raw_response': '',
                'error': None,
                'processing_time': 0
            }
            
            try:
                import time
                start_time = time.time()
                
                # Convert image to base64
                try:
                    image_base64 = image_to_base64(image)
                except Exception as img_error:
                    page_result['error'] = f"Failed to encode image: {str(img_error)}"
                    results['pages'].append(page_result)
                    continue
                
                # Get appropriate prompt
                prompt = get_ocr_prompt(params.get('prompt_type', 'default'))
                
                # Call API
                response, api_error = call_ollama_api(model, prompt, image_base64, params)
                
                if api_error:
                    page_result['error'] = api_error
                    results['pages'].append(page_result)
                    continue
                
                if response:
                    page_result['raw_response'] = response
                    
                    # Try to parse JSON response
                    try:
                        json_response = json.loads(response)
                        page_result['content'] = json_response.get('natural_text', response)
                    except json.JSONDecodeError:
                        # If not JSON, use raw response
                        page_result['content'] = response
                    
                    # Validate content
                    if not page_result['content'].strip():
                        page_result['error'] = "Empty response from API"
                    else:
                        page_result['success'] = True
                        successful_pages += 1
                else:
                    page_result['error'] = "No response from API"
                
                page_result['processing_time'] = time.time() - start_time
                
            except Exception as e:
                page_result['error'] = f"Processing error: {str(e)}"
                logger.error(f"Error processing page {i+1}: {str(e)}")
            
            results['pages'].append(page_result)
        
        # Check if any pages were successful
        results['success'] = successful_pages > 0
        if results['success'] and successful_pages < len(images):
            results['warnings'].append(f"Only {successful_pages} out of {len(images)} pages processed successfully")
        
    except Exception as e:
        results['error'] = f"Document processing failed: {str(e)}"
        logger.error(f"Error processing document {file.name}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return results

def process_documents(uploaded_files, model: str, params: dict):
    """Process uploaded documents with enhanced error handling and progress tracking"""
    
    # Handle single file vs multiple files
    files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    
    st.header("🔄 Processing Results")
    
    # Validate batch size
    max_files = 10
    if len(files_to_process) > max_files:
        st.warning(f"⚠️ Too many files ({len(files_to_process)}). Processing only first {max_files} files.")
        files_to_process = files_to_process[:max_files]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    processing_start_time = time.time()
    
    for i, file in enumerate(files_to_process):
        try:
            status_text.text(f"Processing {file.name} ({i+1}/{len(files_to_process)})...")
            progress_bar.progress(i / len(files_to_process))
            
            # Reset file pointer
            file.seek(0)
            
            # Process document
            result = process_single_document(file, model, params)
            all_results.append(result)
            
            # Display results for this file
            status_icon = "✅ Success" if result['success'] else "❌ Failed"
            with st.expander(f"📄 {file.name} - {status_icon}", expanded=result['success']):
                
                # Show warnings if any
                if result.get('warnings'):
                    for warning in result['warnings']:
                        st.warning(f"⚠️ {warning}")
                
                if result['success']:
                    # Show summary
                    successful_pages = len([p for p in result['pages'] if p['success']])
                    total_time = sum(p.get('processing_time', 0) for p in result['pages'])
                    
                    st.info(f"""
                    **Processing Summary:**
                    - Total pages: {result['total_pages']}
                    - Successful pages: {successful_pages}
                    - Processing time: {total_time:.1f} seconds
                    """)
                    
                    # Show results for each page
                    for page in result['pages']:
                        if page['success']:
                            st.subheader(f"📄 Page {page['page_number']} ({page['processing_time']:.1f}s)")
                            
                            # Tabs for different views
                            tab1, tab2, tab3 = st.tabs(["📖 Preview", "📝 Raw Content", "🔧 Debug"])
                            
                            with tab1:
                                if params.get('output_format') == 'html':
                                    try:
                                        st.markdown(page['content'], unsafe_allow_html=True)
                                    except Exception:
                                        st.code(page['content'], language='html')
                                else:
                                    st.markdown(page['content'])
                            
                            with tab2:
                                st.code(page['content'], language='markdown')
                            
                            with tab3:
                                st.code(page['raw_response'], language='json')
                            
                            # Download button
                            if page['content'].strip():
                                file_extension = params.get('output_format', 'md')
                                if file_extension == 'markdown':
                                    file_extension = 'md'
                                    
                                st.download_button(
                                    f"💾 Download Page {page['page_number']}",
                                    page['content'],
                                    f"{file.name}_page_{page['page_number']}.{file_extension}",
                                    f"text/{file_extension}",
                                    key=f"download_{file.name}_{page['page_number']}"
                                )
                        else:
                            st.error(f"❌ Page {page['page_number']} failed: {page.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ Failed to process {file.name}: {result.get('error', 'Unknown error')}")
                    
                    # Show any partial results
                    if result['pages']:
                        st.info("**Partial results available:**")
                        for page in result['pages']:
                            if page['success']:
                                st.success(f"✅ Page {page['page_number']}: OK")
                            else:
                                st.error(f"❌ Page {page['page_number']}: {page.get('error', 'Failed')}")
                                
        except Exception as e:
            st.error(f"❌ Critical error processing {file.name}: {str(e)}")
            logger.error(f"Critical error processing file {file.name}: {str(e)}")
    
    # Final progress
    progress_bar.progress(1.0)
    total_processing_time = time.time() - processing_start_time
    status_text.text(f"✅ Processing complete! ({total_processing_time:.1f}s)")
    
    # Summary
    successful_files = sum(1 for r in all_results if r['success'])
    total_pages = sum(r['total_pages'] for r in all_results)
    successful_pages = sum(len([p for p in r['pages'] if p['success']]) for r in all_results)
    
    # Display summary
    if successful_files > 0:
        st.success(f"""
        **📊 Processing Summary:**
        - ✅ Files processed: {successful_files}/{len(files_to_process)}
        - 📄 Pages processed: {successful_pages}/{total_pages}
        - 🤖 Model used: {AVAILABLE_MODELS[model]['name']}
        - ⏱️ Total time: {total_processing_time:.1f} seconds
        """)
    else:
        st.error(f"""
        **❌ Processing Failed:**
        - Files failed: {len(files_to_process)}
        - Check error messages above for details
        """)
    
    # Bulk download for successful results
    if successful_pages > 0:
        try:
            # Combine all successful content
            combined_content = f"# Typhoon OCR Results\n\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\nModel: {AVAILABLE_MODELS[model]['name']}\n\n"
            
            for result in all_results:
                if result['success']:
                    combined_content += f"## {result['filename']}\n\n"
                    for page in result['pages']:
                        if page['success']:
                            combined_content += f"### Page {page['page_number']}\n\n{page['content']}\n\n---\n\n"
            
            file_extension = params.get('output_format', 'md')
            if file_extension == 'markdown':
                file_extension = 'md'
            
            st.download_button(
                "📦 Download All Results",
                combined_content,
                f"typhoon_ocr_results_{time.strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                f"text/{file_extension}",
                key="download_all_results"
            )
        except Exception as e:
            st.error(f"Error creating combined download: {str(e)}")

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
    display_system_status()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
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
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, 
                               help="Lower = more precise, Higher = more creative. Recommended: 0.1 for OCR")
        top_p = st.slider("Top P", 0.0, 1.0, 0.6, 0.1,
                         help="Controls word selection diversity. Recommended: 0.6")
        max_tokens = st.slider("Max Tokens", 1000, 16384, 12000, 500,
                              help="Maximum response length. More tokens = longer processing time")
        
        # OCR specific settings
        st.subheader("📄 OCR Settings")
        prompt_type = st.selectbox(
            "Prompt Type:",
            ["default", "structure"],
            format_func=lambda x: "Default (Simple)" if x == "default" else "Structure (Complex Documents)",
            help="Default: Simple text. Structure: Complex tables and images"
        )
        
        output_format = st.selectbox(
            "Output Format:",
            ["markdown", "html", "json"],
            help="Format for the extracted text output"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1,
                                          help="Prevents repetitive text. Higher = less repetition")
            image_quality = st.selectbox(
                "Image Quality", 
                ["high", "medium", "low"],
                index=1,  # Default to medium
                help="Higher quality = better OCR but slower processing"
            )
            batch_processing = st.checkbox("Enable Batch Processing", 
                                          help="Process multiple files at once (up to 10 files)")
            
            # File size limits info
            st.info("""
            **File Limits:**
            - PDF: Max 50MB, 50 pages
            - Images: Max 20MB, 10000x10000px
            - Batch: Max 10 files
            """)

    # Main content area with Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "✨ Features", "📖 คู่มือการใช้งาน", "🔧 System Status"])

    # Tab 1: Upload & Process
    with tab1:
        st.header("📁 Upload Document")

        # File upload with enhanced validation
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=batch_processing,
            help="Support: PDF, PNG, JPG, JPEG\nLimits: PDF (50MB, 50 pages), Images (20MB)"
        )
        
        # Preview uploaded files
        if uploaded_files:
            files_to_show = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            
            # Validate files before showing preview
            valid_files = []
            invalid_files = []
            
            for file in files_to_show:
                try:
                    file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
                    if file.type == 'application/pdf' and file_size > 50 * 1024 * 1024:
                        invalid_files.append(f"{file.name}: Too large ({file_size/(1024*1024):.1f}MB > 50MB)")
                    elif file.type.startswith('image') and file_size > 20 * 1024 * 1024:
                        invalid_files.append(f"{file.name}: Too large ({file_size/(1024*1024):.1f}MB > 20MB)")
                    else:
                        valid_files.append(file)
                        file.seek(0)  # Reset after reading
                except Exception as e:
                    invalid_files.append(f"{file.name}: Error - {str(e)}")
            
            # Show validation results
            if valid_files:
                st.success(f"✅ {len(valid_files)} valid files uploaded")
                
                # Show file details
                for i, file in enumerate(valid_files[:5]):  # Show first 5
                    file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
                    st.write(f"{i+1}. **{file.name}** ({file_size/(1024*1024):.1f}MB, {file.type})")
                    file.seek(0)  # Reset after reading
                    
                if len(valid_files) > 5:
                    st.write(f"... and {len(valid_files) - 5} more files")
                
                # Show preview for first valid file
                if len(valid_files) > 0:
                    first_file = valid_files[0]
                    if first_file.type.startswith('image'):
                        try:
                            first_file.seek(0)
                            image = Image.open(first_file)
                            st.image(image, caption=f"Preview: {first_file.name}", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Could not preview {first_file.name}: {str(e)}")
                    elif first_file.type == 'application/pdf':
                        st.info("📄 PDF file uploaded - preview will be shown during processing")
            
            if invalid_files:
                st.error("❌ Invalid files detected:")
                for error in invalid_files:
                    st.write(f"• {error}")
        
        # Process button
        col1, col2 = st.columns([1, 3])
        with col1:
            process_btn = st.button("🚀 Process Document(s)", type="primary", disabled=not uploaded_files)
        
        with col2:
            if uploaded_files:
                files_count = len(uploaded_files) if isinstance(uploaded_files, list) else 1
                estimated_time = files_count * 30  # 30 seconds per file estimate
                st.info(f"📊 Estimated processing time: {estimated_time//60}m {estimated_time%60}s")
        
        if process_btn:
            if uploaded_files:
                # Final validation before processing
                files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                
                # Check system dependencies one more time
                deps = check_system_dependencies()
                if any(f.type == 'application/pdf' for f in files_to_process) and not deps['poppler']:
                    st.error("❌ Cannot process PDF files: Poppler is not installed!")
                    st.info("Please install Poppler (see System Status tab) or use image files instead.")
                else:
                    process_documents(files_to_process, selected_model, {
                        'temperature': temperature,
                        'top_p': top_p,
                        'max_tokens': max_tokens,
                        'repetition_penalty': repetition_penalty,
                        'prompt_type': prompt_type,
                        'output_format': output_format,
                        'image_quality': image_quality
                    })
            else:
                st.error("❌ Please upload at least one file!")

    # Tab 2: Features
    with tab2:
        st.header("✨ Features")
        
        # Feature cards
        features = [
            {
                "icon": "📊",
                "title": "Structured Documents",
                "description": "รายงานทางการเงิน, เอกสารวิชาการ, แบบฟอร์มราชการ",
                "items": ["Financial reports", "Academic papers", "Government forms", "Books & textbooks"],
                "color": "#e3f2fd"
            },
            {
                "icon": "🍽️", 
                "title": "Layout-Heavy Documents",
                "description": "เอกสารที่เน้น Layout และไม่เป็นทางการ",
                "items": ["Receipts & bills", "Food menus", "Tickets", "Infographics"],
                "color": "#f3e5f5"
            },
            {
                "icon": "🔍",
                "title": "Advanced Analysis", 
                "description": "การวิเคราะห์รูปภาพและไดอะแกรมแบบลึกซึ้ง",
                "items": ["Element detection", "Context analysis", "Text recognition", "Structure analysis"],
                "color": "#e8f5e8"
            },
            {
                "icon": "⚡",
                "title": "Enhanced Error Handling",
                "description": "ระบบจัดการข้อผิดพลาดที่ครอบคลุม",
                "items": ["File validation", "Size limits", "Dependency checks", "Graceful failures"],
                "color": "#fff3e0"
            }
        ]
        
        for feature in features:
            st.markdown(f"""
            <div style="background: {feature['color']}; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; margin: 1rem 0;">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
                <ul>
                    {' '.join([f'<li>✓ {item}</li>' for item in feature['items']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Tab 3: User Guide (keep existing content)
    with tab3:
        st.header("📖 คู่มือการใช้งาน Typhoon OCR")
        st.markdown("""
🌟 **ภาพรวมระบบ**  
Typhoon OCR เป็นเครื่องมือแปลงเอกสารภาษาไทย-อังกฤษด้วย AI ที่สามารถ:

- อ่านและแปลงเอกสาร PDF และรูปภาพ  
- รองรับเอกสารที่มีโครงสร้างซับซ้อน  
- ประมวลผลหลายไฟล์พร้อมกัน  
- ส่งออกผลลัพธ์ในหลากหลายรูปแบบ
- ระบบตรวจสอบความผิดพลาดแบบครอบคลุม

---

### 🛠️ การเตรียมระบบ

#### ข้อกำหนดระบบ:
- **Python 3.8+**
- **Poppler** (สำหรับประมวลผล PDF)
- **Pillow** (สำหรับประมวลผลรูปภาพ)
- **Streamlit** และ dependencies อื่นๆ

#### การติดตั้ง Poppler:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**CentOS/RHEL/Fedora:**
```bash
# CentOS/RHEL 7
sudo yum install poppler-utils

# CentOS/RHEL 8+ / Fedora
sudo dnf install poppler-utils
```

**macOS:**
```bash
# ติดตั้ง Homebrew ก่อน (หากยังไม่มี)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# ติดตั้ง poppler
brew install poppler
```

**Windows:**
1. ดาวน์โหลดจาก: https://github.com/oschwartz10612/poppler-windows/releases/
2. แตกไฟล์ไปยังโฟลเดอร์ เช่น `C:\poppler`
3. เพิ่ม `C:\poppler\Library\bin` ใน PATH environment variable
4. รีสตาร์ทคอมพิวเตอร์

**Docker:**
```dockerfile
FROM python:3.9-slim

# ติดตั้ง system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

### 📊 ขีดจำกัดและข้อกำหนด

#### 📁 ขนาดไฟล์:
- **PDF:** สูงสุด 50MB, 50 หน้า
- **รูปภาพ:** สูงสุด 20MB, 10,000×10,000 พิกเซล
- **Batch Processing:** สูงสุด 10 ไฟล์ต่อครั้ง

#### 🖼️ รูปแบบไฟล์ที่รองรับ:
- **PDF:** `.pdf`
- **รูปภาพ:** `.png`, `.jpg`, `.jpeg`
- **ความละเอียดแนะนำ:** ≥300 DPI สำหรับ OCR ที่แม่นยำ

#### ⚡ ประสิทธิภาพ:
- **การประมวลผล:** 1-3 วินาที/หน้า (ขึ้นอยู่กับขนาดและความซับซ้อน)
- **หน่วยความจำ:** ~500MB-2GB ขึ้นอยู่กับขนาดไฟล์
- **Network:** ต้องเชื่อมต่อ Ollama API

---

### 🎯 การตั้งค่าตัวแปรการประมวลผล

#### 🌡️ Temperature (อุณหภูมิ) - ความสำคัญสูง
**คำอธิบาย:** ควบคุมความสร้างสรรค์และความแปรปรวนของ AI

**ค่าแนะนำสำหรับ OCR:**
- **0.0-0.1 🎯 แนะนำสำหรับ OCR**
  - ผลลัพธ์แม่นยำและสม่ำเสมอที่สุด
  - เหมาะสำหรับ: ใบเสร็จ, เอกสารราชการ, รายงานการเงิน
  - ความผิดพลาดน้อยที่สุด

- **0.2-0.3 ⚖️ สำหรับเอกสารซับซ้อน**
  - สมดุลระหว่างความแม่นยำและการตีความ
  - เหมาะสำหรับ: บทความยาว, เอกสารวิชาการ
  
- **0.4-1.0 ⚠️ ไม่แนะนำสำหรับ OCR**
  - อาจทำให้ผลลัพธ์คลาดเคลื่อน

#### 🎯 Top P (การเลือกคำ)
**คำอธิบาย:** ควบคุมความหลากหลายในการเลือกคำ

**ค่าแนะนำ:**
- **0.3-0.6:** เหมาะสำหรับ OCR ทั่วไป
- **0.6:** ค่า default ที่แนะนำ
- **0.8-1.0:** สำหรับเอกสารที่ต้องการการตีความ

#### 📝 Max Tokens
**ขึ้นอยู่กับประเภทเอกสาร:**
- **1,000-3,000:** ใบเสร็จ, ฟอร์มสั้น
- **4,000-8,000:** รายงาน 2-3 หน้า  
- **9,000-16,384:** เอกสารยาว, หนังสือ

---

### 📄 การตั้งค่า OCR

#### 🎭 ประเภท Prompt
**Default (เอกสารธรรมดา):**
- เหมาะสำหรับ: จดหมาย, บทความ, ฟอร์มพื้นฐาน
- ได้ผลลัพธ์: Markdown แบบธรรมดา
- ประมวลผลเร็วกว่า

**Structure (เอกสารซับซ้อน):**
- เหมาะสำหรับ: รายงานการเงิน, ตาราง, อินโฟกราฟิก
- ได้ผลลัพธ์: HTML + Markdown พร้อมการวิเคราะห์รูปภาพ
- ใช้เวลาประมวลผลนานกว่า

#### 📤 รูปแบบผลลัพธ์
- **Markdown:** อ่านง่าย, แก้ไขง่าย
- **HTML:** เหมาะสำหรับตารางซับซ้อน
- **JSON:** สำหรับนักพัฒนา, การประมวลผลต่อ

---

### 🚨 การจัดการข้อผิดพลาด

#### ข้อผิดพลาดที่พบบ่อยและวิธีแก้:

**1. "Poppler is not installed"**
```
❌ ปัญหา: ไม่สามารถประมวลผล PDF ได้
✅ วิธีแก้: ติดตั้ง poppler-utils (ดูขั้นตอนด้านบน)
🔄 ทางเลือก: ใช้รูปภาพแทน PDF
```

**2. "File too large"**
```
❌ ปัญหา: ไฟล์ใหญ่เกินกำหนด
✅ วิธีแก้: 
   - PDF: แยกไฟล์หรือลดคุณภาพ
   - รูปภาพ: ลดความละเอียดหรือบีบอัด
```

**3. "Model not found"**
```
❌ ปัญหา: โมเดล AI ไม่พร้อมใช้งาน
✅ วิธีแก้: ตรวจสอบการเชื่อมต่อ Ollama API
```

**4. "API timeout"**
```
❌ ปัญหา: ประมวลผลใช้เวลานานเกินไป
✅ วิธีแก้:
   - ลดคุณภาพรูปภาพ
   - แยกไฟล์ใหญ่เป็นไฟล์เล็ก
   - ลด Max Tokens
```

**5. "Empty response"**
```
❌ ปัญหา: ได้ผลลัพธ์เปล่า
✅ วิธีแก้:
   - ตรวจสอบคุณภาพรูปภาพ
   - เปลี่ยนเป็น Structure prompt
   - เพิ่ม Max Tokens
```

#### ระดับการจัดการข้อผิดพลาด:
1. **File Validation:** ตรวจสอบไฟล์ก่อนประมวลผล
2. **Size Limits:** จำกัดขนาดเพื่อป้องกันปัญหา
3. **Dependency Checks:** ตรวจสอบ system dependencies
4. **API Error Handling:** จัดการข้อผิดพลาด API
5. **Graceful Degradation:** แสดงผลบางส่วนหากมีข้อผิดพลาด

---

### 💡 เคล็ดลับการใช้งานที่ดี

#### 📋 การเตรียมเอกสาร:
1. **สแกนด้วยความละเอียดสูง** (≥300 DPI)
2. **ปรับแสงให้เหมาะสม** - ไม่สว่างหรือมืดจนเกินไป
3. **วางเอกสารให้ตรง** - หลีกเลี่ยงการเอียง
4. **ตัดขอบที่ไม่จำเป็น** - เน้นที่เนื้อหาสำคัญ
5. **ใช้พื้นหลังสีขาวหรือ contrasting สี**

#### 🎯 การเลือกตั้งค่า:
- **เอกสารธรรมดา:** Default + Markdown + Temperature 0.1
- **เอกสารซับซ้อน:** Structure + HTML + Temperature 0.1
- **การทดสอบ:** Medium quality + Temperature 0.1
- **งานสำคัญ:** High quality + Temperature 0.1

#### 🚀 เพิ่มประสิทธิภาพ:
- **Batch Processing** สำหรับไฟล์หลายๆ ไฟล์
- **เลือกโมเดลที่เหมาะสม** - Typhoon OCR สำหรับภาษาไทย
- **ตั้งค่า Max Tokens ให้พอดี** - ไม่สูงเกินความจำเป็น
- **ตรวจสอบผลลัพธ์** ก่อนนำไปใช้งาน

---

### 📞 การแก้ไขปัญหาเร่งด่วน

**ขั้นตอนการแก้ไขปัญหาเบื้องต้น:**
1. **ตรวจสอบ System Status tab**
2. **ลองไฟล์ง่ายๆ ก่อน** (รูปภาพเล็ก, ข้อความชัด)
3. **ลด Temperature เป็น 0.1**
4. **ใช้ Default prompt และ Medium quality**
5. **ตรวจสอบขนาดไฟล์และความละเอียด**

**หากยังมีปัญหา:**
- ดู error message ใน console
- ตรวจสอบ network connection
- ลองรีสตาร์ทแอพพลิเคชั่น
- ติดต่อทีม AI NT North

---

### 🔍 การตรวจสอบคุณภาพผลลัพธ์

#### ตัวชี้วัดความสำเร็จ:
- **ความแม่นยำ:** >90% สำหรับข้อความที่ชัดเจน
- **ความครบถ้วน:** จับข้อความทั้งหมดได้
- **การจัดรูปแบบ:** รักษา structure เดิม
- **ตาราง:** แสดงผลถูกต้องใน HTML format

#### การตรวจสอบ:
1. **เปรียบเทียบกับต้นฉบับ**
2. **ตรวจสอบตัวเลขและวันที่**
3. **ดูการจัดรูปแบบตาราง**
4. **ตรวจสอบภาษาไทยและอังกฤษ**

---

### 📈 อนาคตของระบบ

**Features ที่กำลังพัฒนา:**
- รองรับไฟล์ Word และ Excel
- OCR แบบ real-time ผ่าน webcam
- การแปลภาษาอัตโนมัติ
- API สำหรับนักพัฒนา
- การประมวลผลแบบ cloud

**การปรับปรุงประสิทธิภาพ:**
- เพิ่มความเร็วประมวลผล
- ลดการใช้หน่วยความจำ
- รองรับไฟล์ขนาดใหญ่ขึ้น
- ปรับปรุงความแม่นยำ

ระบบนี้ได้รับการปรับปรุงอย่างต่อเนื่อง หากมีข้อเสนะแนะหรือพบปัญหา กรุณาแจ้งให้ทีมพัฒนาทราบ
        """)

    # Tab 4: System Status
    with tab4:
        st.header("🔧 System Status & Diagnostics")
        
        # System Dependencies
        st.subheader("📦 System Dependencies")
        deps = check_system_dependencies()
        
        dep_status = {
            'poppler': {
                'name': 'Poppler Utils',
                'description': 'Required for PDF processing',
                'status': deps['poppler'],
                'install_cmd': {
                    'Ubuntu/Debian': 'sudo apt-get install poppler-utils',
                    'CentOS/RHEL': 'sudo yum install poppler-utils',
                    'macOS': 'brew install poppler',
                    'Windows': 'Download from GitHub releases'
                }
            },
            'pdf2image': {
                'name': 'PDF2Image',
                'description': 'Python library for PDF conversion',
                'status': True,  # Assume installed if we reach this point
                'install_cmd': 'pip install pdf2image'
            },
            'PyPDF2': {
                'name': 'PyPDF2', 
                'description': 'Python library for PDF reading',
                'status': True,  # Assume installed
                'install_cmd': 'pip install PyPDF2'
            }
        }
        
        for dep_key, dep_info in dep_status.items():
            if dep_info['status']:
                st.success(f"✅ **{dep_info['name']}**: Installed")
            else:
                st.error(f"❌ **{dep_info['name']}**: Not installed")
                st.write(f"**Description:** {dep_info['description']}")
                
                if isinstance(dep_info['install_cmd'], dict):
                    st.write("**Installation commands:**")
                    for os_name, cmd in dep_info['install_cmd'].items():
                        st.code(f"{os_name}: {cmd}")
                else:
                    st.code(f"Install: {dep_info['install_cmd']}")
        
        # API Connectivity
        st.subheader("🌐 API Connectivity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Ollama API Connection"):
                with st.spinner("Testing API connection..."):
                    try:
                        response = requests.get(f"{OLLAMA_API_URL.replace('/api/generate', '')}", timeout=10)
                        if response.status_code == 200:
                            st.success("✅ Ollama API is reachable")
                        else:
                            st.warning(f"⚠️ Ollama API returned status {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to Ollama API")
                        st.write(f"URL: {OLLAMA_API_URL}")
                    except requests.exceptions.Timeout:
                        st.error("❌ API connection timeout")
                    except Exception as e:
                        st.error(f"❌ API test failed: {str(e)}")
        
        with col2:
            if st.button("Test Model Availability"):
                with st.spinner("Checking available models..."):
                    try:
                        # Test with a simple request to check if models are available
                        test_payload = {"model": list(AVAILABLE_MODELS.keys())[0], "prompt": "test", "stream": False}
                        response = requests.post(OLLAMA_API_URL, json=test_payload, timeout=30)
                        
                        if response.status_code == 404:
                            st.error("❌ Model not found - check if models are downloaded")
                        elif response.status_code == 200:
                            st.success("✅ Models are available")
                        else:
                            st.warning(f"⚠️ Unexpected response: {response.status_code}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to model server")
                    except requests.exceptions.Timeout:
                        st.error("❌ Model test timeout")
                    except Exception as e:
                        st.error(f"❌ Model test failed: {str(e)}")
        
        # System Information
        st.subheader("💻 System Information")
        
        import platform
        import psutil
        
        sys_info = {
            "Platform": platform.system(),
            "Platform Version": platform.version(),
            "Architecture": platform.machine(),
            "Python Version": platform.python_version(),
            "CPU Cores": psutil.cpu_count(),
            "Memory Total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "Memory Available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
        }
        
        for key, value in sys_info.items():
            st.write(f"**{key}:** {value}")
        
        # Configuration Summary
        st.subheader("⚙️ Current Configuration")
        
        config_info = {
            "Ollama API URL": OLLAMA_API_URL,
            "Embedding API URL": EMBEDDING_API_URL,
            "Available Models": len(AVAILABLE_MODELS),
            "File Size Limits": {
                "PDF": "50 MB, 50 pages",
                "Images": "20 MB, 10000x10000px",
                "Batch": "10 files max"
            }
        }
        
        st.json(config_info)
        
        # Performance Monitoring
        st.subheader("📊 Performance Monitoring")
        
        if st.button("Show Resource Usage"):
            import time
            
            # Create a placeholder for real-time updates
            placeholder = st.empty()
            
            for i in range(10):  # Monitor for 10 seconds
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                with placeholder.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
                    
                    with col2:
                        st.metric("Memory Usage", f"{memory_percent:.1f}%")
                
                if i < 9:  # Don't sleep on last iteration
                    time.sleep(1)
        
        # Log Viewer
        st.subheader("📋 Recent Logs")
        
        if st.button("Show Application Logs"):
            # This would show recent logs if logging to file is implemented
            st.info("Log viewer would show recent application events, errors, and performance metrics here.")
            
            # Sample log entries for demonstration
            sample_logs = [
                "2024-01-20 10:30:15 - INFO - Application started",
                "2024-01-20 10:30:16 - INFO - Dependencies checked successfully",
                "2024-01-20 10:32:45 - INFO - PDF processed: document.pdf (3 pages, 2.1s)",
                "2024-01-20 10:33:12 - WARNING - Large file uploaded: report.pdf (45MB)",
                "2024-01-20 10:35:22 - ERROR - API timeout for model qwen2.5:14b",
                "2024-01-20 10:36:01 - INFO - Fallback to typhoon-ocr successful"
            ]
            
            for log in reversed(sample_logs):
                if "ERROR" in log:
                    st.error(log)
                elif "WARNING" in log:
                    st.warning(log)
                else:
                    st.info(log)

# Import required modules for the updated functions
import time
import psutil

# Run the app
if __name__ == "__main__":
    main()
