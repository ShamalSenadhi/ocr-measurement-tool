import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import morphology
import re
import io
import base64

# Configure page
st.set_page_config(
    page_title="📏 OCR Measurement Extractor",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .result-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .tip-box {
        background: rgba(30, 60, 114, 0.1);
        border: 1px solid rgba(30, 60, 114, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
                # Extract the unit from the original match
                full_match = match.group(0).strip()
                measurements.append(full_match)

    # Also look for standalone numbers that might be measurements
    if not measurements:  # Only if no clear measurements found
        standalone_numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d+)(?![a-zA-Z])', text)
        for num in standalone_numbers:
            if float(num) > 1 and float(num) < 10000:  # Reasonable measurement range
                measurements.append(f"{num} (unit unknown)")

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
                    processing_details.append(f"✅ {detection_mode} + {enhance_mode}: {', '.join(measurements)}")
                else:
                    processing_details.append(f"📝 {detection_mode} + {enhance_mode}: '{text}'")
                    
        except Exception as e:
            processing_details.append(f"❌ {detection_mode} + {enhance_mode}: {str(e)}")
            continue
    
    # Remove duplicates
    unique_measurements = list(set(all_measurements))
    
    return unique_measurements, processing_details

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📏 OCR Measurement & Cable Text Extractor</h1>
        <p>Extract handwritten measurements from paper labels AND printed text from cables/wires</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.header("🔧 Configuration")
    
    detection_mode = st.sidebar.selectbox(
        "🎯 Detection Mode",
        ["measurement", "cable_text", "both", "numbers_only"],
        format_func=lambda x: {
            "measurement": "📏 Measurement Labels",
            "cable_text": "🔌 Cable/Wire Text", 
            "both": "🔀 Both Labels & Cable Text",
            "numbers_only": "🔢 Numbers Only"
        }[x]
    )
    
    language = st.sidebar.selectbox(
        "🌐 Language",
        ["eng", "eng+ara", "eng+chi_sim", "eng+fra", "eng+deu"],
        format_func=lambda x: {
            "eng": "English",
            "eng+ara": "English + Arabic",
            "eng+chi_sim": "English + Chinese", 
            "eng+fra": "English + French",
            "eng+deu": "English + German"
        }[x]
    )
    
    enhance_mode = st.sidebar.selectbox(
        "🔧 Enhancement Mode",
        ["auto", "high_contrast", "cable_optimized", "handwriting", "minimal"],
        format_func=lambda x: {
            "auto": "🤖 Auto Enhance",
            "high_contrast": "⚡ High Contrast",
            "cable_optimized": "🔌 Cable Text Optimized",
            "handwriting": "✍️ Handwriting Optimized", 
            "minimal": "🎯 Minimal Processing"
        }[x]
    )

    # Tips section
    with st.sidebar.expander("💡 Tips for Best Results"):
        st.markdown("""
        **Paper Labels:**
        - Select tight crops around handwritten measurements (like "645m", "155m")
        
        **Cable Text:**
        - Crop close to printed text on black cables/wires
        
        **Smart Extract:**
        - Automatically finds and extracts all measurements
        
        **Good Images:**
        - Ensure clear contrast between text and background
        - Good lighting without shadows
        - Sharp focus
        """)

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload images containing measurements or cable text"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Processing buttons
            st.header("🚀 Processing Options")
            
            col1a, col1b = st.columns(2)
            
            with col1a:
                if st.button("📏 Extract Measurements", type="primary"):
                    with st.spinner("Processing image..."):
                        measurements, raw_text, processed_img = extract_measurements(
                            image, detection_mode, language, enhance_mode
                        )
                        
                        st.session_state['measurements'] = measurements
                        st.session_state['raw_text'] = raw_text
                        st.session_state['processed_img'] = processed_img
                        st.session_state['processing_type'] = 'standard'
            
            with col1b:
                if st.button("🧠 Smart Auto-Extract"):
                    with st.spinner("Smart extraction in progress..."):
                        measurements, details = smart_measurement_extract(image, language)
                        
                        st.session_state['measurements'] = measurements
                        st.session_state['processing_details'] = details
                        st.session_state['processing_type'] = 'smart'

    with col2:
        st.header("📊 Results")
        
        # Display results if available
        if 'measurements' in st.session_state:
            measurements = st.session_state['measurements']
            
            if measurements:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success("📏 Extracted Measurements:")
                for measurement in measurements:
                    st.write(f"• {measurement}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Copy button
                measurements_text = "\n".join([f"• {m}" for m in measurements])
                st.code(measurements_text, language=None)
                
            else:
                st.warning("No measurements detected")
            
            # Show processing details for smart extraction
            if st.session_state.get('processing_type') == 'smart' and 'processing_details' in st.session_state:
                st.subheader("🔍 Processing Details")
                for detail in st.session_state['processing_details']:
                    st.text(detail)
            
            # Show raw text and processed image for standard extraction
            elif st.session_state.get('processing_type') == 'standard':
                if 'raw_text' in st.session_state:
                    st.subheader("📝 Raw OCR Text")
                    st.code(st.session_state['raw_text'])
                
                if 'processed_img' in st.session_state and st.session_state['processed_img']:
                    st.subheader("🔧 Processed Image")
                    st.image(st.session_state['processed_img'], caption="Enhanced for OCR")

    # Additional info
    st.markdown("""
    <div class="tip-box">
    <h4>📋 How to Use:</h4>
    <ol>
        <li><strong>Upload Image:</strong> Choose a photo containing measurements or cable text</li>
        <li><strong>Configure Settings:</strong> Select appropriate detection mode and enhancement</li>
        <li><strong>Extract:</strong> Use "Extract Measurements" for targeted extraction or "Smart Auto-Extract" for automatic detection</li>
        <li><strong>Review Results:</strong> Check extracted measurements and copy them as needed</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()