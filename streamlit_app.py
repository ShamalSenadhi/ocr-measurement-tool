
import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from scipy import ndimage

# Page configuration
st.set_page_config(
    page_title="✍️ Handwriting OCR Extractor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.tips-box {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 15px;
    margin: 15px 0;
    border-radius: 5px;
}
.result-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px;
    margin: 15px 0;
    font-family: monospace;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def preprocess_for_handwriting(img, mode='auto'):
    """Apply specialized preprocessing for handwriting recognition"""
    # Convert PIL to OpenCV format
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    if mode == 'high_contrast':
        # Aggressive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif mode == 'noise_reduction':
        # Focus on noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        enhanced = cv2.equalizeHist(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif mode == 'edge_enhance':
        # Enhance edges for handwriting
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif mode == 'minimal':
        # Minimal processing
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    else:  # auto
        # Auto mode - comprehensive processing
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        
        # 2. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 4. Adaptive threshold
        binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    return Image.fromarray(binary)

def get_ocr_config(mode, language):
    """Get OCR configuration based on mode"""
    # Define character whitelist for handwriting
    handwriting_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-=()[]{}/"' + "'"
    
    configs = {
        'handwriting': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={handwriting_chars}',
        'print': f'--oem 3 --psm 6 -l {language}',
        'mixed': f'--oem 3 --psm 3 -l {language}',
        'numbers': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist=0123456789.,+-=()[]m',
        'single_word': f'--oem 3 --psm 8 -l {language}'
    }
    return configs.get(mode, configs['handwriting'])

def extract_text_from_image(img, ocr_mode, language, preprocess_mode):
    """Extract text from image using specified parameters"""
    try:
        # Apply preprocessing
        processed_img = preprocess_for_handwriting(img, preprocess_mode)
        
        # Get OCR config
        config = get_ocr_config(ocr_mode, language)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=config)
        cleaned_text = text.strip()
        
        # If result is poor, try with original image
        if len(cleaned_text) < 2:
            text_original = pytesseract.image_to_string(img, config=config)
            if len(text_original.strip()) > len(cleaned_text):
                cleaned_text = text_original.strip()
                processed_img = img
        
        return cleaned_text, processed_img
        
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", img

def multiple_attempts_ocr(img, language):
    """Try multiple OCR approaches and return all results"""
    results = []
    modes = ['handwriting', 'numbers', 'single_word', 'print']
    preprocess_modes = ['auto', 'high_contrast', 'minimal', 'edge_enhance']
    
    progress_bar = st.progress(0)
    total_attempts = len(modes) * len(preprocess_modes)
    current_attempt = 0
    
    for ocr_mode in modes:
        for prep_mode in preprocess_modes:
            try:
                current_attempt += 1
                progress_bar.progress(current_attempt / total_attempts)
                
                processed = preprocess_for_handwriting(img, prep_mode)
                config = get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(processed, config=config).strip()
                
                if text and text not in results:
                    results.append((text, f"{ocr_mode} + {prep_mode}"))
                    
            except Exception as e:
                continue
    
    progress_bar.empty()
    return results

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'multi_results' not in st.session_state:
    st.session_state.multi_results = []

# Main header
st.markdown('<h1 class="main-header">✍️ Handwriting & Text OCR Extractor</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # OCR Mode selection
    ocr_mode = st.selectbox(
        "OCR Mode",
        options=['handwriting', 'print', 'mixed', 'numbers', 'single_word'],
        format_func=lambda x: {
            'handwriting': '📝 Handwriting Optimized',
            'print': '🖨️ Printed Text',
            'mixed': '🔀 Mixed Text',
            'numbers': '🔢 Numbers/Measurements',
            'single_word': '📄 Single Word'
        }[x],
        index=0
    )
    
    # Language selection
    language = st.selectbox(
        "Language",
        options=['eng', 'eng+ara', 'eng+chi_sim', 'eng+fra', 'eng+deu', 'eng+spa', 'eng+rus'],
        format_func=lambda x: {
            'eng': 'English',
            'eng+ara': 'English + Arabic',
            'eng+chi_sim': 'English + Chinese',
            'eng+fra': 'English + French',
            'eng+deu': 'English + German',
            'eng+spa': 'English + Spanish',
            'eng+rus': 'English + Russian'
        }[x],
        index=0
    )
    
    # Preprocessing mode
    preprocess_mode = st.selectbox(
        "Preprocessing",
        options=['auto', 'high_contrast', 'noise_reduction', 'edge_enhance', 'minimal'],
        format_func=lambda x: {
            'auto': '🤖 Auto Enhance',
            'high_contrast': '⚡ High Contrast',
            'noise_reduction': '🧹 Noise Reduction',
            'edge_enhance': '📐 Edge Enhancement',
            'minimal': '🎯 Minimal Processing'
        }[x],
        index=0
    )
    
    st.markdown("---")
    
    # Tips section
    st.markdown("""
    <div class="tips-box">
    <h4>💡 Tips for Better Results:</h4>
    <ul>
    <li>Upload clear, high-contrast images</li>
    <li>Ensure good lighting</li>
    <li>Try different modes for difficult text</li>
    <li>Use "Numbers/Measurements" for values like "12.51m"</li>
    <li>Crop images tightly around text area</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing handwritten or printed text"
    )
    
    if uploaded_file is not None:
        # Display original image
        img = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(img, caption=f"Size: {img.size[0]}×{img.size[1]} pixels", use_column_width=True)
        
        # Action buttons
        st.subheader("🎯 Actions")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✍️ Extract Text", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    extracted_text, processed_img = extract_text_from_image(
                        img, ocr_mode, language, preprocess_mode
                    )
                    st.session_state.extracted_text = extracted_text
                    st.session_state.processed_image = processed_img
                    st.session_state.multi_results = []
        
        with col_btn2:
            if st.button("🔄 Multiple Attempts", use_container_width=True):
                with st.spinner("Trying multiple approaches..."):
                    multi_results = multiple_attempts_ocr(img, language)
                    st.session_state.multi_results = multi_results
                    st.session_state.extracted_text = ""
                    st.session_state.processed_image = None

with col2:
    st.header("📊 Results")
    
    # Single extraction results
    if st.session_state.extracted_text:
        st.subheader("✅ Extracted Text")
        
        # Metrics
        text_length = len(st.session_state.extracted_text)
        word_count = len(st.session_state.extracted_text.split())
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Characters", text_length)
        col_m2.metric("Words", word_count)
        
        # Text result
        st.text_area(
            "Result:",
            value=st.session_state.extracted_text,
            height=200,
            help="Copy this text to use elsewhere"
        )
        
        # Download button
        if st.session_state.extracted_text:
            st.download_button(
                "💾 Download as Text File",
                data=st.session_state.extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Multiple attempts results
    if st.session_state.multi_results:
        st.subheader("🔄 Multiple Attempt Results")
        
        # Find best result (longest non-empty)
        best_result = ""
        if st.session_state.multi_results:
            valid_results = [r[0] for r in st.session_state.multi_results if r[0].strip()]
            if valid_results:
                best_result = max(valid_results, key=len)
        
        if best_result:
            st.success(f"🎯 Best Result ({len(best_result)} chars): {best_result}")
            
            if st.button("📋 Use Best Result", use_container_width=True):
                st.session_state.extracted_text = best_result
                st.session_state.multi_results = []
                st.experimental_rerun()
        
        # Show all results
        with st.expander("View All Attempts", expanded=True):
            for i, (result, method) in enumerate(st.session_state.multi_results, 1):
                if result.strip():
                    st.text(f"Attempt {i} ({method}): {result}")
                    
        # Download all results
        if st.session_state.multi_results:
            all_results_text = "\n".join([
                f"Attempt {i} ({method}): {result}"
                for i, (result, method) in enumerate(st.session_state.multi_results, 1)
            ])
            st.download_button(
                "💾 Download All Results",
                data=all_results_text,
                file_name="all_ocr_attempts.txt",
                mime="text/plain",
                use_container_width=True
            )

# Show processed image if available
if st.session_state.processed_image is not None:
    st.header("🔧 Processed Image")
    col_proc1, col_proc2 = st.columns(2)
    
    with col_proc1:
        st.subheader("Before Processing")
        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), use_column_width=True)
    
    with col_proc2:
        st.subheader("After Processing")
        st.image(st.session_state.processed_image, use_column_width=True)

# Footer information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p>✍️ <strong>Handwriting OCR Extractor</strong> - Built with Streamlit & Tesseract</p>
<p>🎯 Optimized for handwritten text, measurements, and mixed content</p>
</div>
""", unsafe_allow_html=True)

# Instructions for setup (shown in expander)
with st.expander("📋 Setup Instructions"):
    st.markdown("""
    ### Required Dependencies
    
    Install the following packages:
    ```bash
    pip install streamlit pytesseract pillow opencv-python numpy scipy scikit-image
    ```
    
    ### Tesseract Installation
    
    **Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr tesseract-ocr-all
    ```
    
    **Windows:**
    Download from: https://github.com/UB-Mannheim/tesseract/wiki
    
    **macOS:**
    ```bash
    brew install tesseract tesseract-lang
    ```
    
    ### Running the App
    ```bash
    streamlit run app.py
    ```
    """)
