import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from scipy import ndimage
from skimage import morphology
import re
import io
import base64
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(
    page_title="📏 Measurement & Cable Text OCR Extractor",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    .result-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
        padding: 15px;
        margin-top: 15px;
        border-radius: 5px;
    }
    .tips-box {
        background: rgba(30, 60, 114, 0.1);
        border: 1px solid rgba(30, 60, 114, 0.2);
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
    }
    .measurement-item {
        background: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def enhance_for_measurements(img, mode='auto'):
    """Enhanced preprocessing for measurement extraction"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if mode == 'cable_optimized':
        # Optimized for printed text on dark cables
        enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=50)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(enhanced)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 10)

    elif mode == 'handwriting':
        # Optimized for handwritten measurements on paper
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    elif mode == 'high_contrast':
        # Maximum contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6,6))
        enhanced = clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=30)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif mode == 'minimal':
        # Minimal processing
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:  # auto
        # Auto mode - balanced approach
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(binary)

def get_measurement_config(mode, language):
    """Get OCR configuration for different measurement types"""
    configs = {
        'measurement': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist=0123456789.,-+m',
        'cable_text': f'--oem 3 --psm 7 -l {language}',
        'both': f'--oem 3 --psm 6 -l {language}',
        'numbers_only': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist=0123456789.,-+'
    }
    return configs.get(mode, configs['measurement'])

def extract_measurement_values(text):
    """Extract measurement values from OCR text using regex patterns"""
    measurements = []

    # Patterns for different measurement formats
    patterns = [
        r'(\d+\.?\d*)\s*m(?:\s|$)',  # "645m", "12.5m", "155 m"
        r'(\d+\.?\d*)\s*mm(?:\s|$)', # "645mm", "12.5mm"
        r'(\d+\.?\d*)\s*cm(?:\s|$)', # "64.5cm"
        r'(\d+\.?\d*)\s*km(?:\s|$)', # "1.5km"
        r'(\d+\.?\d*)\s*ft(?:\s|$)', # "645ft"
        r'(\d+\.?\d*)\s*in(?:\s|$)', # "12.5in"
        r'(\d+)\s*-\s*(\d+)', # Range like "645-650"
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) == 2:  # Range pattern
                measurements.append(f"{match.group(1)}-{match.group(2)}")
            else:
                # Extract the unit from the pattern itself
                unit_match = re.search(r'm(?:m)?|cm|km|ft|in', pattern)
                unit = unit_match.group(0) if unit_match else ""
                measurements.append(f"{match.group(1)}{unit}")

    # Also look for standalone numbers that might be measurements
    standalone_numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d+)(?![a-zA-Z])', text)
    for num in standalone_numbers:
        try:
            if float(num) > 1 and float(num) < 10000:  # Reasonable measurement range
                measurements.append(f"{num} (unit unknown)")
        except ValueError:
            continue

    return list(set(measurements))  # Remove duplicates

def extract_measurements(img, detection_mode='measurement', language='eng', enhance_mode='auto'):
    """Extract measurements from images with specialized processing"""
    try:
        # Apply specialized enhancement
        processed_img = enhance_for_measurements(img, enhance_mode)
        
        # Get appropriate OCR config
        config = get_measurement_config(detection_mode, language)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=config)
        cleaned_text = text.strip()
        
        # If poor results, try with original image
        if len(cleaned_text) < 2:
            text_original = pytesseract.image_to_string(img, config=config)
            if len(text_original.strip()) > len(cleaned_text):
                cleaned_text = text_original.strip()
        
        # Extract measurements using regex
        measurements = extract_measurement_values(cleaned_text)
        
        return measurements, cleaned_text, processed_img
        
    except Exception as e:
        st.error(f"Extraction Error: {str(e)}")
        return [], "", None

def smart_measurement_extract(img, language='eng'):
    """Smart extraction that tries multiple approaches and combines results"""
    try:
        all_measurements = []
        processing_details = []
        
        # Try different combinations of modes and enhancements
        combinations = [
            ('measurement', 'handwriting'),
            ('measurement', 'cable_optimized'),
            ('cable_text', 'cable_optimized'),
            ('both', 'auto'),
            ('numbers_only', 'high_contrast')
        ]
        
        for detection_mode, enhance_mode in combinations:
            try:
                processed_img = enhance_for_measurements(img, enhance_mode)
                config = get_measurement_config(detection_mode, language)
                text = pytesseract.image_to_string(processed_img, config=config).strip()
                
                if text:
                    measurements = extract_measurement_values(text)
                    if measurements:
                        all_measurements.extend(measurements)
                        processing_details.append({
                            'mode': f"{detection_mode} + {enhance_mode}",
                            'status': 'success',
                            'result': ', '.join(measurements)
                        })
                    else:
                        processing_details.append({
                            'mode': f"{detection_mode} + {enhance_mode}",
                            'status': 'text_only',
                            'result': text
                        })
                        
            except Exception as e:
                processing_details.append({
                    'mode': f"{detection_mode} + {enhance_mode}",
                    'status': 'error',
                    'result': str(e)
                })
                continue
        
        # Remove duplicates
        unique_measurements = list(set(all_measurements))
        
        return unique_measurements, processing_details
        
    except Exception as e:
        st.error(f"Smart extraction error: {str(e)}")
        return [], []

def crop_image_from_canvas(img, canvas_result):
    """Crop image based on canvas selection"""
    if canvas_result.json_data is None:
        return img, None
    
    objects = canvas_result.json_data["objects"]
    if not objects:
        return img, None
    
    # Get the last drawn rectangle
    rect = None
    for obj in reversed(objects):
        if obj["type"] == "rect":
            rect = obj
            break
    
    if rect is None:
        return img, None
    
    # Extract coordinates
    left = int(rect["left"])
    top = int(rect["top"])
    width = int(rect["width"])
    height = int(rect["height"])
    
    # Calculate crop coordinates
    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(img.size[0], left + width)
    y2 = min(img.size[1], top + height)
    
    # Crop the image
    cropped_img = img.crop((x1, y1, x2, y2))
    crop_info = f"Cropped area: ({x1}, {y1}) to ({x2}, {y2}) - Size: {x2-x1}x{y2-y1}"
    
    return cropped_img, crop_info

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📏 Measurement & Cable Text OCR Extractor</h1>
        <p>Extract handwritten measurements from paper labels AND printed text from cables/wires</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎯 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing measurements or cable text"
    )
    
    if uploaded_file is not None:
        # Load image
        img = Image.open(uploaded_file)
        
        # Sidebar settings
        detection_mode = st.sidebar.selectbox(
            "🎯 Detection Mode",
            ['measurement', 'cable_text', 'both', 'numbers_only'],
            format_func=lambda x: {
                'measurement': '📏 Measurement Labels',
                'cable_text': '🔌 Cable/Wire Text',
                'both': '🔀 Both Labels & Cable Text',
                'numbers_only': '🔢 Numbers Only'
            }[x]
        )
        
        language = st.sidebar.selectbox(
            "🌐 Language",
            ['eng', 'eng+ara', 'eng+chi_sim', 'eng+fra', 'eng+deu'],
            format_func=lambda x: {
                'eng': 'English',
                'eng+ara': 'English + Arabic',
                'eng+chi_sim': 'English + Chinese',
                'eng+fra': 'English + French',
                'eng+deu': 'English + German'
            }[x]
        )
        
        enhance_mode = st.sidebar.selectbox(
            "🔧 Enhancement",
            ['auto', 'high_contrast', 'cable_optimized', 'handwriting', 'minimal'],
            format_func=lambda x: {
                'auto': '🤖 Auto Enhance',
                'high_contrast': '⚡ High Contrast',
                'cable_optimized': '🔌 Cable Text Optimized',
                'handwriting': '✍️ Handwriting Optimized',
                'minimal': '🎯 Minimal Processing'
            }[x]
        )
        
        # Main content area
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📸 Image & Area Selection")
            
            # Selection mode options
            selection_mode = st.radio(
                "🎯 Selection Mode:",
                ["full_image", "manual_selection", "slider_crop"],
                format_func=lambda x: {
                    "full_image": "📄 Process Full Image",
                    "manual_selection": "✂️ Draw Selection Box",
                    "slider_crop": "📐 Slider-based Crop"
                }[x],
                horizontal=True
            )
            
            if selection_mode == "manual_selection":
                st.info("👆 Draw a rectangle on the image to select the area for OCR processing")
                
                # Calculate display size to fit the container
                max_width = 600
                max_height = 400
                img_width, img_height = img.size
                
                # Calculate scaling to fit within max dimensions
                scale = min(max_width / img_width, max_height / img_height, 1.0)
                display_width = int(img_width * scale)
                display_height = int(img_height * scale)
                
                # Create canvas for selection
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.1)",  # Transparent red fill
                    stroke_width=2,
                    stroke_color="#FF0000",  # Red border
                    background_image=img,
                    width=display_width,
                    height=display_height,
                    drawing_mode="rect",
                    key="canvas",
                    display_toolbar=True
                )
                
                # Process the selection
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        # Scale coordinates back to original image size
                        scale_x = img_width / display_width
                        scale_y = img_height / display_height
                        
                        # Get the last rectangle
                        rect = None
                        for obj in reversed(objects):
                            if obj["type"] == "rect":
                                rect = obj
                                break
                        
                        if rect:
                            # Scale coordinates
                            left = int(rect["left"] * scale_x)
                            top = int(rect["top"] * scale_y)
                            width = int(rect["width"] * scale_x)
                            height = int(rect["height"] * scale_y)
                            
                            # Calculate crop coordinates
                            x1 = max(0, left)
                            y1 = max(0, top)
                            x2 = min(img.size[0], left + width)
                            y2 = min(img.size[1], top + height)
                            
                            # Crop the image
                            cropped_img = img.crop((x1, y1, x2, y2))
                            
                            # Show cropped preview
                            st.success(f"✅ Selected area: {x2-x1}×{y2-y1} pixels")
                            st.image(cropped_img, caption="Selected Area Preview", width=300)
                        else:
                            cropped_img = img
                    else:
                        cropped_img = img
                        st.warning("⚠️ No selection made. Will process full image.")
                else:
                    cropped_img = img
            
            elif selection_mode == "slider_crop":
                st.info("Use the sliders below to define crop area (as percentages)")
                
                col1a, col1b = st.columns(2)
                with col1a:
                    crop_left = st.slider("Left %", 0, 100, 0) / 100
                    crop_top = st.slider("Top %", 0, 100, 0) / 100
                with col1b:
                    crop_right = st.slider("Right %", 0, 100, 100) / 100
                    crop_bottom = st.slider("Bottom %", 0, 100, 100) / 100
                
                # Calculate crop coordinates
                w, h = img.size
                x1 = int(crop_left * w)
                y1 = int(crop_top * h)
                x2 = int(crop_right * w)
                y2 = int(crop_bottom * h)
                
                # Show cropped preview
                cropped_img = img.crop((x1, y1, x2, y2))
                st.image(cropped_img, caption=f"Cropped Preview ({x2-x1}×{y2-y1})", width=300)
            
            else:  # full_image
                st.image(img, caption=f"Original Size: {img.size[0]}×{img.size[1]} pixels", use_column_width=True)
                cropped_img = img
        
        with col2:
            st.subheader("🔧 Processing Options")
            
            # Process buttons
            if st.button("📏 Extract Measurements", type="primary"):
                with st.spinner("🔄 Processing image..."):
                    measurements, raw_text, processed_img = extract_measurements(
                        cropped_img, detection_mode, language, enhance_mode
                    )
                    
                    # Store results in session state
                    st.session_state['measurements'] = measurements
                    st.session_state['raw_text'] = raw_text
                    st.session_state['processed_img'] = processed_img
            
            if st.button("🧠 Smart Auto-Extract", type="secondary"):
                with st.spinner("🧠 Smart extraction in progress..."):
                    measurements, details = smart_measurement_extract(cropped_img, language)
                    
                    # Store results in session state
                    st.session_state['smart_measurements'] = measurements
                    st.session_state['smart_details'] = details
        
        # Display results
        if 'measurements' in st.session_state or 'smart_measurements' in st.session_state:
            st.markdown("---")
            
            # Regular extraction results
            if 'measurements' in st.session_state:
                st.subheader("📏 Extraction Results")
                
                measurements = st.session_state['measurements']
                raw_text = st.session_state['raw_text']
                processed_img = st.session_state['processed_img']
                
                if measurements:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success(f"✅ Found {len(measurements)} measurements:")
                    
                    for i, measurement in enumerate(measurements, 1):
                        st.markdown(f'<div class="measurement-item">• {measurement}</div>', 
                                  unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Copy to clipboard button
                    measurements_text = "\n".join([f"• {m}" for m in measurements])
                    st.text_area("📋 Copy Results:", measurements_text, height=100)
                else:
                    st.warning("No measurements detected")
                
                # Show raw text and processed image
                with st.expander("🔍 View Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Raw OCR Text:", raw_text, height=100)
                    with col2:
                        if processed_img:
                            st.image(processed_img, caption="Processed Image", width=300)
            
            # Smart extraction results
            if 'smart_measurements' in st.session_state:
                st.subheader("🧠 Smart Extraction Results")
                
                measurements = st.session_state['smart_measurements']
                details = st.session_state['smart_details']
                
                if measurements:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success(f"✅ Smart extraction found {len(measurements)} unique measurements:")
                    
                    for i, measurement in enumerate(measurements, 1):
                        st.markdown(f'<div class="measurement-item">• {measurement}</div>', 
                                  unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Copy to clipboard button
                    measurements_text = "\n".join([f"• {m}" for m in measurements])
                    st.text_area("📋 Copy Smart Results:", measurements_text, height=100)
                else:
                    st.warning("No clear measurements detected")
                
                # Show processing details
                with st.expander("📊 Processing Details"):
                    for detail in details:
                        status_icon = {
                            'success': '✅',
                            'text_only': '📝',
                            'error': '❌'
                        }.get(detail['status'], '❓')
                        
                        st.write(f"{status_icon} **{detail['mode']}**: {detail['result']}")
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload an image to get started")
    
    # Tips section
    st.markdown("---")
    st.markdown("""
    <div class="tips-box">
        <h4>💡 Tips for Best Results:</h4>
        <ul>
            <li><strong>Draw Selection:</strong> Use the rectangle tool to precisely select measurement areas</li>
            <li><strong>Paper Labels:</strong> Draw tight boxes around handwritten measurements (like "645m", "155m")</li>
            <li><strong>Cable Text:</strong> Select areas with printed text on black cables/wires</li>
            <li><strong>Smart Extract:</strong> Automatically finds and extracts all measurements in the image</li>
            <li><strong>Selection Modes:</strong> Choose between drawing, slider cropping, or full image processing</li>
            <li><strong>Good Lighting:</strong> Ensure clear contrast between text and background</li>
            <li><strong>Multiple Selections:</strong> You can clear and redraw selections as needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Installation note
    st.markdown("---")
    st.markdown("""
    **📋 Required Installation:**
    ```bash
    pip install streamlit pytesseract pillow opencv-python numpy scipy scikit-image streamlit-drawable-canvas
    
    # For Ubuntu/Debian:
    sudo apt install tesseract-ocr tesseract-ocr-eng
    
    # For Windows: Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
    # For macOS: brew install tesseract
    ```
    """)

if __name__ == "__main__":
    main()
