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
from typing import Optional, Dict, Any

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

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌪️ Typhoon OCR</h1>
        <p>AI-Powered Thai-English Document Parser</p>
        <p>Powered by SCB 10X Advanced AI Models</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.6, 0.1)
        max_tokens = st.slider("Max Tokens", 1000, 16384, 12000, 500)
        
        # OCR specific settings
        st.subheader("📄 OCR Settings")
        prompt_type = st.selectbox(
            "Prompt Type:",
            ["default", "structure"],
            format_func=lambda x: "Default (Simple)" if x == "default" else "Structure (Complex Documents)"
        )
        
        output_format = st.selectbox(
            "Output Format:",
            ["markdown", "html", "json"]
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1)
            image_quality = st.selectbox("Image Quality", ["high", "medium", "low"])
            batch_processing = st.checkbox("Enable Batch Processing")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
                st.success(f"✅ {len(uploaded_files)} files uploaded")
                for i, file in enumerate(uploaded_files[:3]):  # Show first 3
                    st.write(f"{i+1}. {file.name} ({file.size} bytes)")
                if len(uploaded_files) > 3:
                    st.write(f"... and {len(uploaded_files) - 3} more files")
            else:
                st.success(f"✅ File uploaded: {uploaded_files.name}")
                
                # Show preview for single file
                if uploaded_files.type.startswith('image'):
                    image = Image.open(uploaded_files)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                elif uploaded_files.type == 'application/pdf':
                    st.info("📄 PDF file uploaded - preview will be shown during processing")
        
        # Process button
        if st.button("🚀 Process Document(s)", type="primary"):
            if uploaded_files:
                process_documents(uploaded_files, selected_model, {
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
    
    with col2:
        st.header("✨ Features")
        
        # Feature cards
        features = [
            {
                "icon": "📊",
                "title": "Structured Documents",
                "description": "รายงานทางการเงิน, เอกสารวิชาการ, แบบฟอร์มราชการ",
                "items": ["Financial reports", "Academic papers", "Government forms", "Books & textbooks"]
            },
            {
                "icon": "🍽️", 
                "title": "Layout-Heavy Documents",
                "description": "เอกสารที่เน้น Layout และไม่เป็นทางการ",
                "items": ["Receipts & bills", "Food menus", "Tickets", "Infographics"]
            },
            {
                "icon": "🔍",
                "title": "Advanced Analysis", 
                "description": "การวิเคราะห์รูปภาพและไดอะแกรมแบบลึกซึ้ง",
                "items": ["Element detection", "Context analysis", "Text recognition", "Structure analysis"]
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

def convert_pdf_to_images(pdf_file, quality="high") -> list:
    """Convert PDF to images"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file.flush()
            
            # Convert PDF to images
            if quality == "high":
                dpi = 300
            elif quality == "medium":
                dpi = 200
            else:
                dpi = 150
                
            images = pdf2image.convert_from_path(tmp_file.name, dpi=dpi)
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return []

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

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
    """Call Ollama API for OCR processing"""
    try:
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
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
        
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
        return None

def process_single_document(file, model: str, params: dict) -> Dict[str, Any]:
    """Process a single document"""
    results = {
        'filename': file.name,
        'pages': [],
        'total_pages': 0,
        'success': False,
        'error': None
    }
    
    try:
        # Handle different file types
        if file.type == 'application/pdf':
            images = convert_pdf_to_images(file, params.get('image_quality', 'high'))
            results['total_pages'] = len(images)
        else:
            # Image file
            image = Image.open(file)
            images = [image]
            results['total_pages'] = 1
        
        if not images:
            results['error'] = "Could not process file"
            return results
        
        # Process each page/image
        for i, image in enumerate(images):
            page_result = {
                'page_number': i + 1,
                'success': False,
                'content': '',
                'raw_response': ''
            }
            
            try:
                # Convert image to base64
                image_base64 = image_to_base64(image)
                
                # Get appropriate prompt
                prompt = get_ocr_prompt(params.get('prompt_type', 'default'))
                
                # Call API
                response = call_ollama_api(model, prompt, image_base64, params)
                
                if response:
                    page_result['raw_response'] = response
                    
                    # Try to parse JSON response
                    try:
                        json_response = json.loads(response)
                        page_result['content'] = json_response.get('natural_text', response)
                    except json.JSONDecodeError:
                        page_result['content'] = response
                    
                    page_result['success'] = True
                else:
                    page_result['error'] = "No response from API"
                
            except Exception as e:
                page_result['error'] = str(e)
            
            results['pages'].append(page_result)
        
        # Check if any pages were successful
        results['success'] = any(page['success'] for page in results['pages'])
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def process_documents(uploaded_files, model: str, params: dict):
    """Process uploaded documents"""
    
    # Handle single file vs multiple files
    files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    
    st.header("🔄 Processing Results")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    
    for i, file in enumerate(files_to_process):
        status_text.text(f"Processing {file.name}...")
        progress_bar.progress((i) / len(files_to_process))
        
        # Reset file pointer
        file.seek(0)
        
        # Process document
        result = process_single_document(file, model, params)
        all_results.append(result)
        
        # Display results for this file
        with st.expander(f"📄 {file.name} - {'✅ Success' if result['success'] else '❌ Failed'}", expanded=True):
            if result['success']:
                # Show results for each page
                for page in result['pages']:
                    if page['success']:
                        st.subheader(f"Page {page['page_number']}")
                        
                        # Tabs for different views
                        tab1, tab2, tab3 = st.tabs(["📖 Preview", "📝 Markdown", "🔧 Raw"])
                        
                        with tab1:
                            if params.get('output_format') == 'html':
                                st.markdown(page['content'], unsafe_allow_html=True)
                            else:
                                st.markdown(page['content'])
                        
                        with tab2:
                            st.code(page['content'], language='markdown')
                        
                        with tab3:
                            st.code(page['raw_response'], language='json')
                        
                        # Download button
                        st.download_button(
                            f"💾 Download Page {page['page_number']}",
                            page['content'],
                            f"{file.name}_page_{page['page_number']}.{params.get('output_format', 'md')}",
                            f"text/{params.get('output_format', 'markdown')}"
                        )
                    else:
                        st.error(f"Page {page['page_number']} failed: {page.get('error', 'Unknown error')}")
            else:
                st.error(f"Failed to process {file.name}: {result.get('error', 'Unknown error')}")
    
    # Final progress
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Summary
    successful_files = sum(1 for r in all_results if r['success'])
    total_pages = sum(r['total_pages'] for r in all_results)
    successful_pages = sum(len([p for p in r['pages'] if p['success']]) for r in all_results)
    
    st.success(f"""
    **Processing Summary:**
    - Files processed: {successful_files}/{len(files_to_process)}
    - Pages processed: {successful_pages}/{total_pages}
    - Model used: {AVAILABLE_MODELS[model]['name']}
    """)
    
    # Bulk download for successful results
    if successful_pages > 0:
        # Combine all successful content
        combined_content = ""
        for result in all_results:
            if result['success']:
                combined_content += f"# {result['filename']}\n\n"
                for page in result['pages']:
                    if page['success']:
                        combined_content += f"## Page {page['page_number']}\n\n{page['content']}\n\n---\n\n"
        
        st.download_button(
            "📦 Download All Results",
            combined_content,
            f"typhoon_ocr_results.{params.get('output_format', 'md')}",
            f"text/{params.get('output_format', 'markdown')}"
        )

# Run the app
if __name__ == "__main__":
    main()
