import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import pytesseract
import re
import io
import base64
from scipy import ndimage
import pandas as pd

# Configure page
st.set_page_config(
    page_title="📏 Measurement & Cable Text OCR Extractor",
    page_icon="📏",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .measurement-result {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .tips-box {
        background: rgba(30, 60, 114, 0.1);
        border: 1px solid rgba(30, 60, 114, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .selection-box {
        border: 2px solid #ff4444;
        background: rgba(255, 68, 68, 0.2);
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
        'measurement': f'--oem 3 --psm 8 -l {language} -c tesseract_char_whitelist=0123456789.,-+m',
        'cable_text': f'--oem 3 --psm 7 -l {language}',
        'both': f'--oem 3 --psm 6 -l {language}',
        'numbers_only': f'--oem 3 --psm 8 -l {language} -c tesseract_char_whitelist=0123456789.,-+'
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
        if float(num) > 1 and float(num) < 10000:  # Reasonable measurement range
            measurements.append(f"{num} (unit unknown)")

    return list(set(measurements))  # Remove duplicates

def extract_measurements_from_image(img, detection_mode='measurement', language='eng', enhance_mode='auto'):
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
                        processing_details.append(f"✅ {detection_mode} + {enhance_mode}: {', '.join(measurements)}")
                    else:
                        processing_details.append(f"📝 {detection_mode} + {enhance_mode}: '{text}'")

            except Exception as e:
                processing_details.append(f"❌ {detection_mode} + {enhance_mode}: {str(e)}")
                continue

        # Remove duplicates
        unique_measurements = list(set(all_measurements))
        
        return unique_measurements, processing_details

    except Exception as e:
        st.error(f"Smart extraction error: {str(e)}")
        return [], []

def draw_selection_box(img, x1, y1, x2, y2):
    """Draw a selection rectangle on the image"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Add semi-transparent overlay
    overlay = Image.new('RGBA', img_copy.size, (255, 0, 0, 50))
    mask = Image.new('L', img_copy.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([x1, y1, x2, y2], fill=255)
    
    img_copy = Image.composite(overlay, img_copy.convert('RGBA'), mask)
    return img_copy.convert('RGB')

def create_grid_selections(img_width, img_height, grid_size=4):
    """Create predefined grid selections for easy selection"""
    selections = []
    cell_width = img_width // grid_size
    cell_height = img_height // grid_size
    
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            selections.append({
                'name': f'Grid {row+1}-{col+1}',
                'coords': (x1, y1, x2, y2)
            })
    
    return selections

# Main Application
def main():
    st.markdown('<div class="main-header">📏 Measurement & Cable Text OCR Extractor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract handwritten measurements from paper labels AND printed text from cables/wires</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'selection_coords' not in st.session_state:
        st.session_state.selection_coords = None
    if 'selection_mode' not in st.session_state:
        st.session_state.selection_mode = 'manual'

    # Sidebar controls
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing measurements or cable text"
        )
        
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.selection_coords = None  # Reset selection when new image uploaded
        
        # Selection method
        st.subheader("✂️ Selection Method")
        selection_method = st.radio(
            "Choose selection method:",
            ["Manual Coordinates", "Grid Selection", "Preset Areas"],
            help="Different ways to select areas of the image"
        )
        
        # Detection mode
        detection_mode = st.selectbox(
            "🎯 Detection Mode",
            ['measurement', 'cable_text', 'both', 'numbers_only'],
            format_func=lambda x: {
                'measurement': '📏 Measurement Labels',
                'cable_text': '🔌 Cable/Wire Text',
                'both': '🔀 Both Labels & Cable Text',
                'numbers_only': '🔢 Numbers Only'
            }[x]
        )
        
        # Language selection
        language = st.selectbox(
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
        
        # Enhancement mode
        enhance_mode = st.selectbox(
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
    if st.session_state.uploaded_image:
        img = st.session_state.uploaded_image
        img_width, img_height = img.size
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🖼️ Image Selection")
            
            # Selection interface based on chosen method
            if selection_method == "Manual Coordinates":
                st.write("📐 **Manual Coordinate Selection**")
                
                coord_col1, coord_col2 = st.columns(2)
                with coord_col1:
                    x1 = st.number_input("X1 (left)", min_value=0, max_value=img_width, value=0, key="x1")
                    y1 = st.number_input("Y1 (top)", min_value=0, max_value=img_height, value=0, key="y1")
                
                with coord_col2:
                    x2 = st.number_input("X2 (right)", min_value=0, max_value=img_width, value=min(200, img_width), key="x2")
                    y2 = st.number_input("Y2 (bottom)", min_value=0, max_value=img_height, value=min(100, img_height), key="y2")
                
                # Validate coordinates
                if x1 < x2 and y1 < y2:
                    st.session_state.selection_coords = (x1, y1, x2, y2)
                    # Show image with selection
                    selected_img = draw_selection_box(img, x1, y1, x2, y2)
                    st.image(selected_img, caption=f"Selection: {x2-x1}×{y2-y1} pixels", use_column_width=True)
                else:
                    st.warning("⚠️ Invalid coordinates: X2 must be > X1 and Y2 must be > Y1")
            
            elif selection_method == "Grid Selection":
                st.write("🔲 **Grid Selection**")
                
                grid_size = st.selectbox("Grid size:", [2, 3, 4, 6], index=2)
                grid_selections = create_grid_selections(img_width, img_height, grid_size)
                
                selected_grid = st.selectbox(
                    "Select grid cell:",
                    range(len(grid_selections)),
                    format_func=lambda x: grid_selections[x]['name']
                )
                
                if selected_grid is not None:
                    x1, y1, x2, y2 = grid_selections[selected_grid]['coords']
                    st.session_state.selection_coords = (x1, y1, x2, y2)
                    
                    # Show image with grid and selection
                    selected_img = draw_selection_box(img, x1, y1, x2, y2)
                    st.image(selected_img, caption=f"Grid {grid_selections[selected_grid]['name']}: {x2-x1}×{y2-y1} pixels", use_column_width=True)
            
            elif selection_method == "Preset Areas":
                st.write("🎯 **Preset Selection Areas**")
                
                preset_options = {
                    "Top Left Quarter": (0, 0, img_width//2, img_height//2),
                    "Top Right Quarter": (img_width//2, 0, img_width, img_height//2),
                    "Bottom Left Quarter": (0, img_height//2, img_width//2, img_height),
                    "Bottom Right Quarter": (img_width//2, img_height//2, img_width, img_height),
                    "Center Half": (img_width//4, img_height//4, 3*img_width//4, 3*img_height//4),
                    "Top Half": (0, 0, img_width, img_height//2),
                    "Bottom Half": (0, img_height//2, img_width, img_height),
                    "Left Half": (0, 0, img_width//2, img_height),
                    "Right Half": (img_width//2, 0, img_width, img_height)
                }
                
                selected_preset = st.selectbox("Select preset area:", list(preset_options.keys()))
                
                if selected_preset:
                    x1, y1, x2, y2 = preset_options[selected_preset]
                    st.session_state.selection_coords = (x1, y1, x2, y2)
                    
                    # Show image with selection
                    selected_img = draw_selection_box(img, x1, y1, x2, y2)
                    st.image(selected_img, caption=f"{selected_preset}: {x2-x1}×{y2-y1} pixels", use_column_width=True)
            
            # Processing buttons
            st.subheader("🚀 Processing Options")
            
            button_cols = st.columns(4)
            
            with button_cols[0]:
                if st.button("📏 Extract Selection", key="extract_btn"):
                    if st.session_state.selection_coords:
                        x1, y1, x2, y2 = st.session_state.selection_coords
                        
                        # Crop the image
                        cropped_img = img.crop((x1, y1, x2, y2))
                        
                        # Process the cropped image
                        with st.spinner("🔄 Processing selected area..."):
                            measurements, raw_text, processed_img = extract_measurements_from_image(
                                cropped_img, detection_mode, language, enhance_mode
                            )
                            
                            st.session_state.processing_results = {
                                'type': 'selection',
                                'measurements': measurements,
                                'raw_text': raw_text,
                                'processed_img': processed_img,
                                'cropped_img': cropped_img,
                                'coords': (x1, y1, x2, y2)
                            }
                    else:
                        st.warning("⚠️ Please make a selection first")
            
            with button_cols[1]:
                if st.button("📄 Process Full Image", key="full_btn"):
                    with st.spinner("🔄 Processing full image..."):
                        measurements, raw_text, processed_img = extract_measurements_from_image(
                            img, detection_mode, language, enhance_mode
                        )
                        
                        st.session_state.processing_results = {
                            'type': 'full',
                            'measurements': measurements,
                            'raw_text': raw_text,
                            'processed_img': processed_img
                        }
            
            with button_cols[2]:
                if st.button("🧠 Smart Auto-Extract", key="smart_btn"):
                    with st.spinner("🧠 Smart extraction in progress..."):
                        measurements, details = smart_measurement_extract(img, language)
                        
                        st.session_state.processing_results = {
                            'type': 'smart',
                            'measurements': measurements,
                            'details': details
                        }
            
            with button_cols[3]:
                if st.button("🗑️ Clear Results", key="clear_btn"):
                    st.session_state.processing_results = {}
                    st.rerun()
        
        with col2:
            st.subheader("📊 Results")
            
            # Display results
            if st.session_state.processing_results:
                results = st.session_state.processing_results
                
                if results['type'] == 'smart':
                    st.markdown('<div class="measurement-result">', unsafe_allow_html=True)
                    st.write("🧠 **Smart Extraction Results:**")
                    
                    if results['measurements']:
                        for measurement in results['measurements']:
                            st.write(f"• {measurement}")
                    else:
                        st.write("No measurements detected")
                    
                    st.write("\n📊 **Processing Details:**")
                    for detail in results['details']:
                        st.write(f"  {detail}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.markdown('<div class="measurement-result">', unsafe_allow_html=True)
                    st.write(f"📏 **{results['type'].title()} Extraction Results:**")
                    
                    if 'coords' in results:
                        x1, y1, x2, y2 = results['coords']
                        st.write(f"📐 **Selected Area:** {x2-x1}×{y2-y1} pixels at ({x1},{y1})")
                    
                    if results['measurements']:
                        st.write("**Found Measurements:**")
                        for measurement in results['measurements']:
                            st.write(f"• {measurement}")
                    else:
                        st.write("No measurements detected")
                    
                    st.write(f"\n**Raw OCR Text:** '{results['raw_text']}'")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show processed image if available
                    if 'processed_img' in results and results['processed_img']:
                        st.subheader("🔍 Processed Image")
                        st.image(results['processed_img'], caption="Enhanced for OCR", use_column_width=True)
                    
                    # Show cropped image if available
                    if 'cropped_img' in results and results['cropped_img']:
                        st.subheader("✂️ Selected Area")
                        st.image(results['cropped_img'], caption="Selected Region", use_column_width=True)
                
                # Copy results button
                if results.get('measurements'):
                    measurements_text = '\n'.join([f"• {m}" for m in results['measurements']])
                    st.text_area("📋 Copy Results:", measurements_text, height=100)
            
            else:
                st.info("Select an area and choose a processing method to see results here.")
    
    else:
        st.info("👆 Upload an image to get started!")

    # Tips section
    st.markdown('<div class="tips-box">', unsafe_allow_html=True)
    st.subheader("💡 Tips for Best Results:")
    st.write("""
    **Selection Methods:**
    - **Manual Coordinates:** Precise pixel-level selection using X1,Y1,X2,Y2 coordinates
    - **Grid Selection:** Quick selection using predefined grid cells (2×2, 3×3, 4×4, 6×6)
    - **Preset Areas:** Common areas like quarters, halves, and center regions
    
    **Extraction Tips:**
    - **Paper Labels:** Select tight areas around handwritten measurements ("645m", "155m")
    - **Cable Text:** Focus on printed text areas on cables/wires  
    - **Smart Extract:** Tries all methods automatically - good for mixed content
    - **Good Contrast:** Ensure clear difference between text and background
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
