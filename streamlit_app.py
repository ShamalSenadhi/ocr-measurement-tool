
import streamlit as st
import pytesseract
import io
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import morphology
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Measurement & Cable Text OCR",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title("📏 Measurement & Cable Text OCR Extractor")
st.markdown("""
    Extract handwritten measurements from paper labels AND printed text from cables/wires.
    Upload an image below and use the options to refine the extraction process.
""")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Settings")

# --- Image Upload ---
uploaded_file = st.sidebar.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

# --- Detection Mode ---
detection_mode = st.sidebar.selectbox(
    "🎯 Detection Mode",
    ("measurement", "cable_text", "both", "numbers_only"),
    format_func=lambda x: {
        "measurement": "📏 Measurement Labels",
        "cable_text": "🔌 Cable/Wire Text",
        "both": "🔀 Both Labels & Cable Text",
        "numbers_only": "🔢 Numbers Only"
    }[x]
)

# --- Language Selection ---
language = st.sidebar.selectbox(
    "🌐 Language",
    ("eng", "eng+ara", "eng+chi_sim", "eng+fra", "eng+deu"),
    format_func=lambda x: {
        "eng": "English",
        "eng+ara": "English + Arabic",
        "eng+chi_sim": "English + Chinese",
        "eng+fra": "English + French",
        "eng+deu": "English + German",
    }[x]
)

# --- Image Enhancement ---
enhance_mode = st.sidebar.selectbox(
    "🔧 Enhancement",
    ("auto", "high_contrast", "cable_optimized", "handwriting", "minimal"),
    format_func=lambda x: {
        "auto": "🤖 Auto Enhance",
        "high_contrast": "⚡ High Contrast",
        "cable_optimized": "🔌 Cable Text Optimized",
        "handwriting": "✍️ Handwriting Optimized",
        "minimal": "🎯 Minimal Processing"
    }[x]
)

# --- Main Content Area ---
st.header("🖼️ Image Preview and Results")

image_placeholder = st.empty()
result_placeholder = st.empty()
status_placeholder = st.empty()

# --- Image Processing and OCR Functions (from original Colab notebook) ---
def enhance_for_measurements(img, mode='auto'):
    """Enhanced preprocessing for measurement extraction"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if mode == 'cable_optimized':
        enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=50)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(enhanced)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 10)

    elif mode == 'handwriting':
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    elif mode == 'high_contrast':
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6,6))
        enhanced = clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=30)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif mode == 'minimal':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:  # auto
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
    handwriting_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-=()[]{}/"' + "'"
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

    patterns = [
        r'(\d+\.?\d*)\s*m(?:\s|$)',
        r'(\d+\.?\d*)\s*mm(?:\s|$)',
        r'(\d+\.?\d*)\s*cm(?:\s|$)',
        r'(\d+\.?\d*)\s*km(?:\s|$)',
        r'(\d+\.?\d*)\s*ft(?:\s|$)',
        r'(\d+\.?\d*)\s*in(?:\s|$)',
        r'(\d+)\s*-\s*(\d+)',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) == 2:
                measurements.append(f"{match.group(1)}-{match.group(2)}")
            else:
                unit_match = re.search(r'm(?:m)?|cm|km|ft|in', pattern)
                unit = unit_match.group(0) if unit_match else ""
                measurements.append(f"{match.group(1)}{unit}")

    standalone_numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d+)(?![a-zA-Z])', text)
    for num in standalone_numbers:
        if float(num) > 1 and float(num) < 10000:
            measurements.append(f"{num} (unit unknown)")

    return list(set(measurements))

def extract_measurements(img, detection_mode, language, enhance_mode, processing_type, status_callback=None):
    """Extract measurements from images with specialized processing"""
    if status_callback:
        status_callback(f"Processing {processing_type}...")
    else:
        print(f"Processing {processing_type}...")

    try:
        processed_img = enhance_for_measurements(img, enhance_mode)
        config = get_measurement_config(detection_mode, language)
        text = pytesseract.image_to_string(processed_img, config=config)
        cleaned_text = text.strip()

        if len(cleaned_text) < 2:
            text_original = pytesseract.image_to_string(img, config=config)
            if len(text_original.strip()) > len(cleaned_text):
                cleaned_text = text_original.strip()

        measurements = extract_measurement_values(cleaned_text)

        if measurements:
            result_text = f"📏 EXTRACTED MEASUREMENTS:\\n\\n" + "\\n".join(measurements) + f"\\n\\nRAW TEXT: {cleaned_text}"
        else:
            result_text = f"RAW EXTRACTED TEXT:\\n{cleaned_text}\\n\\n(No measurements detected - check the raw text above)"

        if status_callback:
            status_callback(f"✅ {processing_type} completed.")
        else:
            print(f"✅ {processing_type} completed.")

        # Return processed image and result text
        return processed_img, result_text

    except Exception as e:
        error_msg = f"Extraction Error: {str(e)}"
        if status_callback:
            status_callback(f"❌ {error_msg}")
        else:
            print(f"❌ {error_msg}")
        return None, f"Error: {error_msg}"


def smart_measurement_extract(img, language, status_callback=None):
    """Smart extraction that tries multiple approaches and combines results"""
    if status_callback:
        status_callback("🧠 Smart extraction in progress - finding all measurements...")
    else:
        print("🧠 Smart extraction in progress - finding all measurements...")

    all_measurements = []
    processing_details = []

    combinations = [
        ('measurement', 'handwriting'),
        ('measurement', 'cable_optimized'),
        ('cable_text', 'cable_optimized'),
        ('both', 'auto'),
        ('numbers_only', 'high_contrast')
    ]

    for detection_mode, enhance_mode in combinations:
        try:
            processed = enhance_for_measurements(img, enhance_mode)
            config = get_measurement_config(detection_mode, language)
            text = pytesseract.image_to_string(processed, config=config).strip()

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

    unique_measurements = list(set(all_measurements))

    if unique_measurements:
        measurements_text = "\\n".join([f"• {m}" for m in unique_measurements])
    else:
        measurements_text = "No clear measurements detected"

    details_text = "\\n".join(processing_details)

    formatted_results = f"📏 SMART EXTRACTION RESULTS:\\n\\n{measurements_text}\\n\\n📊 DETAILS:\\n{details_text}"

    if status_callback:
        status_callback("🧠 Smart extraction completed.")
    else:
        print("🧠 Smart extraction completed.")

    return None, formatted_results # Return None for preview image in smart mode


# --- Image Processing and OCR Logic ---
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        image_placeholder.image(img, caption="Uploaded Image", use_column_width=True)

        # --- Add Buttons ---
        col1, col2, col3 = st.columns(3)

        with col1:
            # Selection extraction requires frontend canvas interaction which is complex in pure Streamlit
            # We'll disable this button for now or require manual coordinates input (not implemented here)
            extract_button = st.button("📏 Extract Selected (Advanced)", disabled=True)

        with col2:
            full_image_button = st.button("📄 Process Full Image")

        with col3:
            smart_extract_button = st.button("🧠 Smart Auto-Extract")

        # --- Button Actions ---
        if full_image_button:
            processed_preview, result_text = extract_measurements(
                img, detection_mode, language, enhance_mode, 'Full Image', status_placeholder.info
            )
            if processed_preview:
                 st.image(processed_preview, caption="Processed Image Preview", use_column_width=True)
            result_placeholder.text_area("📝 Extraction Results", result_text, height=300)


        if smart_extract_button:
             processed_preview, result_text = smart_measurement_extract(
                 img, language, status_placeholder.info
             )
             # Smart extract doesn't provide a single preview image in this implementation
             result_placeholder.text_area("📝 Extraction Results", result_text, height=300)


    except Exception as e:
        st.error(f"An error occurred: {e}")
        status_placeholder.error("❌ Error loading or processing image.")

# --- Tips Section ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
    #### 💡 Tips:
    - Select tight crops for measurements. (Manual selection not yet supported in this Streamlit version)
    - Good contrast helps OCR accuracy.
    - 'Smart Auto-Extract' tries multiple settings.
""")

# --- How to Run ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
    #### How to Run:
    1. Save this code as a Python file (e.g., `ocr_app.py`).
    2. Make sure you have Tesseract OCR installed on your system.
    3. Open your terminal or command prompt.
    4. Navigate to the directory where you saved the file.
    5. Run the command: `streamlit run ocr_app.py`
    6. Your browser will open with the application.
""")
