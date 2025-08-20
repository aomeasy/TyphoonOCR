import streamlit as st
import requests
import base64
import json
from io import BytesIO
from PIL import Image
import tempfile
import os
from typing import Optional, Dict, Any, List
import logging

# Multiple PDF processing libraries for fallback
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    st.warning("⚠️ pdf2image not available. Some PDF features may be limited.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pdf_dependencies():
    """ตรวจสอบ dependencies ที่จำเป็นสำหรับการประมวลผล PDF"""
    status = {
        'pdf2image': PDF2IMAGE_AVAILABLE,
        'pymupdf': PYMUPDF_AVAILABLE, 
        'pypdf2': PYPDF2_AVAILABLE,
        'pdfplumber': PDFPLUMBER_AVAILABLE,
        'poppler_available': False
    }
    
    # ตรวจสอบ poppler สำหรับ pdf2image
    if PDF2IMAGE_AVAILABLE:
        try:
            # ทดสอบ poppler โดยการสร้าง PDF dummy
            test_pdf = create_test_pdf()
            if test_pdf:
                pdf2image.convert_from_bytes(test_pdf, first_page=1, last_page=1, dpi=50)
                status['poppler_available'] = True
                logger.info("✅ Poppler is available and working")
        except Exception as e:
            logger.warning(f"⚠️ Poppler test failed: {str(e)}")
            status['poppler_available'] = False
    
    return status

def create_test_pdf() -> Optional[bytes]:
    """สร้าง PDF ทดสอบขนาดเล็ก"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(100, 750, "Test PDF")
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer.getvalue()
    except ImportError:
        # สร้าง minimal PDF header สำหรับการทดสอบ
        minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f 
0000000010 00000 n 
0000000053 00000 n 
0000000100 00000 n 
trailer<</Size 4/Root 1 0 R>>
startxref
159
%%EOF"""
        return minimal_pdf

def convert_pdf_to_images_fallback(pdf_file, quality="high") -> tuple[list, str]:
    """
    แปลง PDF เป็น images โดยใช้หลายวิธี fallback
    Returns: (images_list, method_used)
    """
    images = []
    method_used = "none"
    error_messages = []
    
    # ตั้งค่า DPI ตามคุณภาพ
    dpi_settings = {
        "high": 300,
        "medium": 200, 
        "low": 150
    }
    dpi = dpi_settings.get(quality, 200)
    
    # อ่านไฟล์ PDF
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
    except Exception as e:
        return [], f"Error reading PDF file: {str(e)}"
    
    # วิธีที่ 1: pdf2image (ต้องการ poppler)
    if PDF2IMAGE_AVAILABLE:
        try:
            logger.info("🔄 Trying pdf2image...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_file.flush()
                
                images = pdf2image.convert_from_path(tmp_file.name, dpi=dpi)
                method_used = "pdf2image"
                logger.info(f"✅ pdf2image successful - {len(images)} pages")
                
                # Clean up
                os.unlink(tmp_file.name)
                return images, method_used
                
        except Exception as e:
            error_messages.append(f"pdf2image failed: {str(e)}")
            logger.warning(f"❌ pdf2image failed: {str(e)}")
            # Clean up on error
            try:
                os.unlink(tmp_file.name)
            except:
                pass
    
    # วิธีที่ 2: PyMuPDF (fitz) - ไม่ต้องการ poppler
    if PYMUPDF_AVAILABLE and not images:
        try:
            logger.info("🔄 Trying PyMuPDF...")
            doc = fitz.open("pdf", pdf_bytes)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # แปลงเป็น image
                mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                images.append(img)
            
            doc.close()
            method_used = "pymupdf"
            logger.info(f"✅ PyMuPDF successful - {len(images)} pages")
            return images, method_used
            
        except Exception as e:
            error_messages.append(f"PyMuPDF failed: {str(e)}")
            logger.warning(f"❌ PyMuPDF failed: {str(e)}")
    
    # วิธีที่ 3: pdfplumber + PIL (สำหรับ text-heavy PDFs)
    if PDFPLUMBER_AVAILABLE and not images:
        try:
            logger.info("🔄 Trying pdfplumber...")
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # สร้าง image จาก text content (สำหรับ OCR)
                    # นี่เป็นวิธีการ fallback สำหรับ text-only PDFs
                    img = create_text_image(page.extract_text() or f"Page {i+1} - No text extracted")
                    images.append(img)
            
            method_used = "pdfplumber_text"
            logger.info(f"✅ pdfplumber successful - {len(images)} pages (text-based)")
            return images, method_used
            
        except Exception as e:
            error_messages.append(f"pdfplumber failed: {str(e)}")
            logger.warning(f"❌ pdfplumber failed: {str(e)}")
    
    # วิธีที่ 4: PyPDF2 + text conversion (last resort)
    if PYPDF2_AVAILABLE and not images:
        try:
            logger.info("🔄 Trying PyPDF2...")
            with open(temp_file_path, 'rb') if 'temp_file_path' in locals() else BytesIO(pdf_bytes) as file:
                reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        img = create_text_image(text or f"Page {i+1} - No text extracted")
                        images.append(img)
                    except Exception as page_error:
                        logger.warning(f"Failed to process page {i+1}: {page_error}")
                        # สร้าง placeholder image
                        img = create_text_image(f"Page {i+1} - Processing failed")
                        images.append(img)
            
            method_used = "pypdf2_text"
            logger.info(f"✅ PyPDF2 successful - {len(images)} pages (text-based)")
            return images, method_used
            
        except Exception as e:
            error_messages.append(f"PyPDF2 failed: {str(e)}")
            logger.warning(f"❌ PyPDF2 failed: {str(e)}")
    
    # ถ้าทุกวิธีล้มเหลว
    error_summary = " | ".join(error_messages)
    return [], f"All PDF conversion methods failed: {error_summary}"

def create_text_image(text: str, width: int = 800, height: int = 1000) -> Image.Image:
    """สร้าง image จาก text (สำหรับ fallback)"""
    try:
        from PIL import ImageDraw, ImageFont
        
        # สร้าง image พื้นหลังสีขาว
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # พยายามใช้ font ที่มี
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
        
        # เขียนข้อความ
        lines = text.split('\n')
        y_pos = 20
        line_height = 15
        
        for line in lines[:50]:  # จำกัดจำนวนบรรทัด
            if y_pos + line_height > height - 20:
                break
            draw.text((20, y_pos), line[:80], fill='black', font=font)  # จำกัดความยาวบรรทัด
            y_pos += line_height
        
        return img
        
    except Exception as e:
        logger.error(f"Error creating text image: {str(e)}")
        # สร้าง blank image
        return Image.new('RGB', (width, height), 'white')

def convert_pdf_to_images(pdf_file, quality="high") -> list:
    """
    Main function สำหรับแปลง PDF เป็น images
    พร้อม comprehensive error handling
    """
    try:
        # ตรวจสอบ dependencies
        deps = check_pdf_dependencies()
        
        if not any([deps['pdf2image'], deps['pymupdf'], deps['pdfplumber'], deps['pypdf2']]):
            st.error("❌ No PDF processing libraries available. Please install pdf2image, PyMuPDF, pdfplumber, or PyPDF2")
            return []
        
        # แสดงสถานะ dependencies
        with st.expander("🔍 PDF Processing Status", expanded=False):
            st.write("**Available Libraries:**")
            for lib, status in deps.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {lib}: {'Available' if status else 'Not available'}")
        
        # เรียกใช้ fallback converter
        images, method_used = convert_pdf_to_images_fallback(pdf_file, quality)
        
        if images:
            st.success(f"✅ PDF converted successfully using {method_used} ({len(images)} pages)")
            return images
        else:
            st.error(f"❌ Failed to convert PDF: {method_used}")
            return []
            
    except Exception as e:
        logger.error(f"Unexpected error in PDF conversion: {str(e)}")
        st.error(f"❌ Unexpected error: {str(e)}")
        return []

def display_pdf_help():
    """แสดงคำแนะนำสำหรับการแก้ปัญหา PDF"""
    st.info("""
    **🛠️ การแก้ไขปัญหา PDF:**
    
    **สำหรับ Linux/Ubuntu/WSL:**
    ```bash
    sudo apt update
    sudo apt install poppler-utils
    pip install PyMuPDF pdfplumber
    ```
    
    **สำหรับ macOS:**
    ```bash
    brew install poppler
    pip install PyMuPDF pdfplumber
    ```
    
    **สำหรับ Windows:**
    1. ดาวน์โหลด poppler: https://github.com/oschwartz10612/poppler-windows
    2. เพิ่ม poppler/bin ไปยัง System PATH
    3. `pip install PyMuPDF pdfplumber`
    
    **วิธีทางเลือก:** ใช้ PyMuPDF (ไม่ต้องการ poppler)
    ```bash
    pip install PyMuPDF
    ```
    """)

# เพิ่ม function นี้ไปยัง main() function
def enhanced_main():
    """Enhanced main function with PDF debugging"""
    
    # เพิ่มการตรวจสอบ PDF capabilities ใน sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # ตรวจสอบ PDF dependencies
        deps = check_pdf_dependencies()
        
        if not any(deps.values()):
            st.error("❌ No PDF processing available")
            if st.button("📖 Show PDF Setup Guide"):
                display_pdf_help()
        else:
            working_methods = [k for k, v in deps.items() if v]
            st.success(f"✅ PDF support: {', '.join(working_methods)}")
    
    # เพิ่ม debug tab
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "✨ Features", "📖 คู่มือการใช้งาน", "🔧 Debug"])
    
    with tab4:
        st.header("🔧 Debug Information")
        
        st.subheader("📋 System Dependencies")
        deps = check_pdf_dependencies()
        
        for lib, status in deps.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{lib}**")
            with col2:
                if status:
                    st.success("✅")
                else:
                    st.error("❌")
        
        st.subheader("🧪 Test PDF Conversion")
        if st.button("Test PDF Processing"):
            # สร้าง test PDF
            test_pdf_bytes = create_test_pdf()
            if test_pdf_bytes:
                # แปลงเป็น BytesIO object
                from io import BytesIO
                test_file = BytesIO(test_pdf_bytes)
                test_file.name = "test.pdf"
                
                images, method = convert_pdf_to_images_fallback(test_file, "low")
                if images:
                    st.success(f"✅ Test successful using {method}")
                    st.image(images[0], caption="Test PDF converted", width=200)
                else:
                    st.error(f"❌ Test failed: {method}")
            else:
                st.error("Could not create test PDF")
        
        display_pdf_help()
