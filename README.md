# 🌪️ Typhoon OCR Web Application

A powerful web application for Thai-English document parsing using advanced AI models, built with Streamlit.

## 🚀 Features

### 📊 Structured Documents
- Financial reports (รายงานทางการเงิน)
- Academic papers (เอกสารวิชาการ)  
- Government forms (แบบฟอร์มราชการ)
- Books & textbooks (หนังสือและตำรา)

### 🍽️ Layout-Heavy Documents
- Receipts & bills (ใบเสร็จและบิล)
- Food menus (เมนูอาหาร)
- Tickets (ตั๋วต่างๆ)
- Infographics (อินโฟกราฟิก)

### 🔍 Advanced Analysis
- Multi-layer figure analysis
- Context-aware text recognition
- Complex table structure preservation
- Bilingual content support

## 🤖 Available AI Models

- **Typhoon OCR 7B** - Specialized Thai-English OCR model
- **Qwen2.5 14B** - General purpose large language model
- **Typhoon2 8B** - Latest Thai language model

## 🛠️ Installation & Setup

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Deploy from your forked repository
5. Set environment variables in Streamlit Cloud settings

### Option 2: Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd typhoon-ocr-webapp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## ⚙️ Configuration

### Environment Variables

Set these in your Streamlit Cloud secrets or local `.env` file:

```toml
EMBEDDING_API_URL = "http://209.15.123.47:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_API_URL = "http://209.15.123.47:11434/api/generate"
```

### Supported File Types

- **PDF**: Multi-page document support
- **Images**: PNG, JPG, JPEG
- **Maximum file size**: 10MB per file
- **Batch processing**: Multiple files simultaneously

## 📖 Usage Guide

### 1. Upload Documents
- Drag & drop files or use the file picker
- Support for single or multiple files
- Preview uploaded content

### 2. Configure Settings
- Choose AI model based on your needs
- Adjust processing parameters:
  - Temperature (0.0-1.0)
  - Top P (0.0-1.0) 
  - Max tokens (1000-16384)
- Select prompt type:
  - Default: Simple documents
  - Structure: Complex layouts

### 3. Process & Download
- Click "Process Document(s)"
- View results in multiple formats
- Download individual pages or combined results

## 🎯 Processing Parameters

### Temperature
- **Low (0.0-0.1)**: More focused, deterministic output
- **Medium (0.3-0.7)**: Balanced creativity and accuracy
- **High (0.8-1.0)**: More creative, varied output

### Prompt Types
- **Default**: Best for simple documents with standard layouts
- **Structure**: Optimized for complex documents with tables, figures, and mixed content

### Output Formats
- **Markdown**: Clean, readable text format
- **HTML**: Preserves complex table structures
- **JSON**: Structured data with metadata

## 🔧 Advanced Features

### Batch Processing
- Process multiple documents simultaneously
- Progress tracking with status updates
- Bulk download of all results

### Quality Settings
- **High**: 300 DPI (best quality, slower)
- **Medium**: 200 DPI (balanced)
- **Low**: 150 DPI (faster processing)

### Multi-language Support
- Seamless Thai-English document processing
- Context-aware language detection
- Preserve original formatting

## 📊 Model Comparison

| Model | Best For | Strengths | Use Cases |
|-------|----------|-----------|-----------|
| Typhoon OCR 7B | OCR Tasks | Thai-English expertise, Table handling | Documents, Forms, Reports |
| Qwen2.5 14B | General Tasks | Large context, Reasoning | Complex analysis, Q&A |
| Typhoon2 8B | Thai Content | Thai language mastery | Thai documents, Instructions |

## 🚨 Limitations & Considerations

### Current Limitations
- Specific prompt requirements for Typhoon OCR
- No built-in guardrails (raw model output)
- Potential hallucination with complex layouts
- Processing time varies with document complexity

### Best Practices
- Use low temperature (0.0-0.1) for accuracy
- Choose appropriate prompt type for document complexity
- Verify critical information from original documents
- Use structure prompt for documents with complex layouts

## 🔄 API Integration

The application connects to your private Ollama server:

```python
# Example API call structure
payload = {
    "model": "scb10x/typhoon-ocr-7b:latest",
    "prompt": ocr_prompt,
    "images": [image_base64],
    "temperature": 0.1,
    "stream": false
}
```

## 📝 Project Structure

```
typhoon-ocr-webapp/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # This file
└── .env.example          # Environment variables template
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is built upon the Typhoon OCR model by SCB 10X. Please refer to their licensing terms.

## 🆘 Support

For issues and questions:
1. Check the [Issues](../../issues) page
2. Review the Typhoon OCR documentation
3. Contact the development team

## 🎉 Acknowledgments

- **SCB 10X Team** for the Typhoon OCR model
- **Qwen Team** for the base models
- **Streamlit** for the amazing web framework
- **Open source community** for various tools and libraries

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Developed with**: ❤️ and ☕
