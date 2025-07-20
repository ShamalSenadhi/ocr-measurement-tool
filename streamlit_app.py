import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas

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
.selection-info {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def image_to_base64_url(img):
    """Convert PIL image to base64 data URL for canvas background"""
    buffered = io.BytesIO()
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

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

def crop_image_from_canvas(original_image, canvas_result):
    """Extract cropped region from canvas selection"""
    if canvas_result is None:
        # Check for manual coordinates fallback
        if hasattr(st.session_state, 'manual_coords'):
            left, top, width, height = st.session_state.manual_coords
            cropped = original_image.crop((left, top, left + width, top + height))
            return cropped, (left, top, width, height)
        return None, None
    
    if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0:
        return None, None
    
    # Get the rectangle from canvas
    rect = canvas_result.json_data["objects"][0]
    if rect["type"] != "rect":
        return None, None
    
    # Get original image dimensions
    orig_width, orig_height = original_image.size
    
    # Get canvas dimensions (we'll scale the selection)
    canvas_width = 600  # Fixed canvas width
    canvas_height = int(600 * orig_height / orig_width) if orig_width > orig_height else 600
    
    # Calculate scaling factors
    scale_x = orig_width / canvas_width
    scale_y = orig_height / canvas_height
    
    # Get rectangle coordinates and scale them
    left = int(rect["left"] * scale_x)
    top = int(rect["top"] * scale_y)
    width = int(rect["width"] * scale_x)
    height = int(rect["height"] * scale_y)
    
    # Ensure coordinates are within image bounds
    left = max(0, min(left, orig_width - 1))
    top = max(0, min(top, orig_height - 1))
    width = min(width, orig_width - left)
    height = min(height, orig_height - top)
    
    # Crop the original image
    cropped = original_image.crop((left, top, left + width, top + height))
    
    return cropped, (left, top, width, height)

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
    status_text = st.empty()
    total_attempts = len(modes) * len(preprocess_modes)
    current_attempt = 0
    
    for ocr_mode in modes:
        for prep_mode in preprocess_modes:
            try:
                current_attempt += 1
                progress_bar.progress(current_attempt / total_attempts)
                status_text.text(f"Trying {ocr_mode} with {prep_mode}... ({current_attempt}/{total_attempts})")
                
                processed = preprocess_for_handwriting(img, prep_mode)
                config = get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(processed, config=config).strip()
                
                if text and text not in [r[0] for r in results]:
                    results.append((text, f"{ocr_mode} + {prep_mode}"))
                    
            except Exception as e:
                continue
    
    progress_bar.empty()
    status_text.empty()
    return results

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'multi_results' not in st.session_state:
    st.session_state.multi_results = []
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None
if 'selection_coords' not in st.session_state:
    st.session_state.selection_coords = None
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
if 'manual_coords' not in st.session_state:
    st.session_state.manual_coords = None

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
    
    # Processing options
    st.subheader("📋 Processing Options")
    process_full_image = st.radio(
        "Process:",
        ["Selected Area Only", "Full Image"],
        help="Choose whether to process the entire image or just the selected region"
    )
    
    st.markdown("---")
    
    # Tips section
    st.markdown("""
    <div class="tips-box">
    <h4>💡 Tips for Better Results:</h4>
    <ul>
    <li><strong>Make tight selections</strong> around handwritten text</li>
    <li>Ensure good contrast between text and background</li>
    <li>Try different preprocessing modes if first attempt fails</li>
    <li>Use "Numbers/Measurements" for values like "12.51m"</li>
    <li>Use "Multiple Attempts" for difficult handwriting</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content area
uploaded_file = st.file_uploader(
    "📤 Choose an image file",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Upload an image containing handwritten or printed text"
)

if uploaded_file is not None:
    # Load and display original image
    original_img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Select Text Region")
        st.markdown("**Instructions:** Draw a rectangle around the text you want to extract")
        
        # Calculate canvas dimensions maintaining aspect ratio
        canvas_width = 600
        canvas_height = int(canvas_width * original_img.height / original_img.width)
        
        # Limit height to prevent extremely tall canvases
        if canvas_height > 800:
            canvas_height = 800
            canvas_width = int(canvas_height * original_img.width / original_img.height)
        
        try:
            # Resize image for canvas while maintaining aspect ratio
            canvas_img = original_img.copy()
            canvas_img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Create drawable canvas with PIL image
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.2)",  # Orange with transparency
                stroke_width=2,
                stroke_color="#FF4444",
                background_color="#FFFFFF",
                background_image=canvas_img,
                update_streamlit=True,
                width=canvas_width,
                height=canvas_height,
                drawing_mode="rect",
                point_display_radius=0,
                key=f"canvas_{st.session_state.canvas_key}",
            )
        except Exception as e:
            st.error(f"Canvas error: {str(e)}")
            st.info("Fallback: Using image display without canvas selection")
            canvas_result = None
            st.image(original_img, width=canvas_width, caption="Original Image")
            
            # Simple coordinate input fallback
            st.subheader("Manual Selection (Fallback)")
            col_x1, col_y1 = st.columns(2)
            col_x2, col_y2 = st.columns(2)
            
            with col_x1:
                x1 = st.number_input("Left (x1)", min_value=0, max_value=original_img.width-1, value=0)
            with col_y1:
                y1 = st.number_input("Top (y1)", min_value=0, max_value=original_img.height-1, value=0)
            with col_x2:
                x2 = st.number_input("Right (x2)", min_value=x1+1, max_value=original_img.width, value=min(200, original_img.width))
            with col_y2:
                y2 = st.number_input("Bottom (y2)", min_value=y1+1, max_value=original_img.height, value=min(100, original_img.height))
            
            # Create mock canvas result for fallback
            if st.button("Apply Manual Selection"):
                st.session_state.manual_coords = (x1, y1, x2-x1, y2-y1)
        
        # Action buttons
        st.subheader("🎯 Actions")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            extract_btn = st.button("✍️ Extract Text", type="primary", use_container_width=True)
            
        with col_btn2:
            multi_btn = st.button("🔄 Multi-Try", use_container_width=True)
            
        with col_btn3:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        # Clear selection
        if clear_btn:
            st.session_state.extracted_text = ""
            st.session_state.processed_image = None
            st.session_state.multi_results = []
            st.session_state.cropped_image = None
            st.session_state.selection_coords = None
            st.session_state.manual_coords = None
            st.session_state.canvas_key += 1
            st.rerun()
    
    with col2:
        st.subheader("📊 Results & Preview")
        
        # Process image based on selection
        target_image = original_img
        processing_type = "Full Image"
        
        # Check if there's a selection and user wants to process selected area
        if (((canvas_result is not None and
              canvas_result.json_data is not None and 
              len(canvas_result.json_data["objects"]) > 0) or
             st.session_state.manual_coords is not None) and 
            process_full_image == "Selected Area Only"):
            
            cropped_img, coords = crop_image_from_canvas(original_img, canvas_result)
            if cropped_img is not None:
                target_image = cropped_img
                processing_type = "Selected Area"
                st.session_state.cropped_image = cropped_img
                st.session_state.selection_coords = coords
                
                # Show selection info
                st.markdown(f"""
                <div class="selection-info">
                <strong>📏 Selection Info:</strong><br>
                Size: {coords[2]}×{coords[3]} pixels<br>
                Position: ({coords[0]}, {coords[1]})
                </div>
                """, unsafe_allow_html=True)
        
        # Show current processing target
        if st.session_state.cropped_image is not None and process_full_image == "Selected Area Only":
            st.image(st.session_state.cropped_image, caption="Selected Region for Processing", use_column_width=True)
        
        # Extract text button action
        if extract_btn:
            with st.spinner(f"Processing {processing_type.lower()}..."):
                extracted_text, processed_img = extract_text_from_image(
                    target_image, ocr_mode, language, preprocess_mode
                )
                st.session_state.extracted_text = extracted_text
                st.session_state.processed_image = processed_img
                st.session_state.multi_results = []
        
        # Multiple attempts button action
        if multi_btn:
            if (process_full_image == "Selected Area Only" and 
                st.session_state.cropped_image is None and 
                st.session_state.manual_coords is None):
                st.warning("⚠️ Please make a selection first!")
            else:
                with st.spinner(f"Trying multiple approaches on {processing_type.lower()}..."):
                    multi_results = multiple_attempts_ocr(target_image, language)
                    st.session_state.multi_results = multi_results
                    st.session_state.extracted_text = ""
                    st.session_state.processed_image = None
        
        # Display results
        if st.session_state.extracted_text:
            st.success("✅ Text Extraction Complete!")
            
            # Metrics
            text_length = len(st.session_state.extracted_text)
            word_count = len(st.session_state.extracted_text.split())
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Characters", text_length)
            col_m2.metric("Words", word_count)
            
            # Text result
            st.text_area(
                f"Extracted Text ({processing_type}):",
                value=st.session_state.extracted_text,
                height=150,
                help="Copy this text to use elsewhere"
            )
            
            # Download button
            st.download_button(
                "💾 Download Text",
                data=st.session_state.extracted_text,
                file_name=f"extracted_text_{processing_type.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Multiple attempts results
        elif st.session_state.multi_results:
            st.success("🔄 Multiple Attempts Complete!")
            
            # Find best result (longest non-empty)
            best_result = ""
            if st.session_state.multi_results:
                valid_results = [r[0] for r in st.session_state.multi_results if r[0].strip()]
                if valid_results:
                    best_result = max(valid_results, key=len)
            
            if best_result:
                st.info(f"🎯 Best Result ({len(best_result)} chars): {best_result}")
                
                if st.button("📋 Use Best Result", use_container_width=True):
                    st.session_state.extracted_text = best_result
                    st.session_state.multi_results = []
                    st.rerun()
            
            # Show all results in expander
            with st.expander("View All Attempts", expanded=True):
                for i, (result, method) in enumerate(st.session_state.multi_results, 1):
                    if result.strip():
                        st.text(f"Attempt {i} ({method}): {result}")
                        
            # Download all results
            if st.session_state.multi_results:
                all_results_text = f"Multiple OCR Attempts - {processing_type}\n" + "="*50 + "\n\n"
                all_results_text += "\n".join([
                    f"Attempt {i} ({method}): {result}"
                    for i, (result, method) in enumerate(st.session_state.multi_results, 1)
                ])
                st.download_button(
                    "💾 Download All Results",
                    data=all_results_text,
                    file_name=f"all_ocr_attempts_{processing_type.lower().replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Show processed image if available
    if st.session_state.processed_image is not None:
        st.markdown("---")
        st.subheader("🔧 Image Processing Preview")
        
        col_proc1, col_proc2 = st.columns(2)
        
        with col_proc1:
            st.subheader("Before Processing")
            display_img = st.session_state.cropped_image if st.session_state.cropped_image is not None else original_img
            st.image(display_img, use_column_width=True)
        
        with col_proc2:
            st.subheader("After Processing")
            st.image(st.session_state.processed_image, use_column_width=True)

else:
    # Welcome screen when no image is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 50px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
    <h3>👋 Welcome to Handwriting OCR Extractor!</h3>
    <p>Upload an image to get started with text extraction</p>
    <p>📝 Perfect for handwritten notes, measurements, and mixed text</p>
    </div>
    """, unsafe_allow_html=True)

# Footer information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p>✍️ <strong>Handwriting OCR Extractor</strong> - Built with Streamlit & Tesseract</p>
<p>🎯 Optimized for handwritten text, measurements, and mixed content</p>
</div>
""", unsafe_allow_html=True)

# Setup instructions
with st.expander("📋 Setup Instructions & Dependencies"):
    st.markdown("""
    ### Required Dependencies
    
    Install the following packages:
    ```bash
    pip install streamlit pytesseract pillow opencv-python numpy scipy scikit-image streamlit-drawable-canvas
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
    streamlit run streamlit_app.py
    ```
    
    ### Features
    - 🖱️ **Interactive Selection**: Draw rectangles to select specific text regions
    - 🔄 **Multiple OCR Modes**: Optimized for different text types
    - 🎨 **Image Preprocessing**: 5 different enhancement modes
    - 🌐 **Multi-language Support**: 7 language combinations
    - 📊 **Detailed Results**: Character/word counts and processing previews
    - 💾 **Export Options**: Download extracted text and results
    
    ### Troubleshooting
    - If canvas doesn't work, the app will fall back to simple image display
    - Make sure Tesseract is properly installed and in your PATH
    - For better handwriting recognition, try different preprocessing modes
    """)
