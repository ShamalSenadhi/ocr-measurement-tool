import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔍 Dual Cable Measurement Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240,147,251,0.3);
    }
    
    .results-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .measurement-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .no-measurement-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .comparison-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79,172,254,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize EasyOCR reader
@st.cache_resource
def load_easyocr():
    """Load EasyOCR reader with caching"""
    return easyocr.Reader(['en'], gpu=False)

def enhance_for_measurement(img, method):
    """Apply image enhancement specifically for measurement extraction"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    if method == 'High Contrast':
        # Increase contrast for faded text
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=20)
        
    elif method == 'Cable Optimized':
        # Optimized for text on dark cables
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=40)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6))
        enhanced = clahe.apply(enhanced)
        
    elif method == 'Denoised':
        # Remove noise while preserving text
        enhanced = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
        
    elif method == 'Edge Enhanced':
        # Edge enhancement for better text detection
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
    elif method == 'Handwriting Optimized':
        # Optimized for handwritten text
        enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
    else:  # Original
        enhanced = gray
    
    return Image.fromarray(enhanced)

def extract_length_measurements(text):
    """Extract only length measurements in meters from text"""
    measurements = []
    
    # Patterns to match measurements in meters
    patterns = [
        r'(\d+(?:\.\d+)?)\s*m(?:\s|$|[^a-zA-Z])',     # "645m", "12.5m", "155 m"
        r'(\d+(?:\.\d+)?)\s*meter[s]?',               # "645 meters", "12.5 meter"
        r'(\d+(?:\.\d+)?)\s*mtr[s]?',                 # "645 mtrs", "12.5 mtr"
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            value = match.group(1)
            measurements.append(f"{value}m")
    
    # Convert other units to meters if found
    # mm to meters
    mm_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*mm', text, re.IGNORECASE)
    for match in mm_matches:
        mm_value = float(match.group(1))
        m_value = mm_value / 1000
        if m_value > 0.1:  # Only if result is reasonable
            measurements.append(f"{m_value:.3f}m")
    
    # cm to meters
    cm_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*cm', text, re.IGNORECASE)
    for match in cm_matches:
        cm_value = float(match.group(1))
        m_value = cm_value / 100
        if m_value > 0.01:  # Only if result is reasonable
            measurements.append(f"{m_value:.2f}m")
    
    # Remove duplicates and sort
    measurements = list(set(measurements))
    measurements.sort(key=lambda x: float(x.replace('m', '')))
    
    return measurements

def process_single_image(img, reader, image_name):
    """Process a single image with all enhancement methods"""
    methods = ['Original', 'High Contrast', 'Cable Optimized', 'Denoised', 'Edge Enhanced', 'Handwriting Optimized']
    
    results = {}
    all_measurements = set()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, method in enumerate(methods):
        progress = (idx + 1) / len(methods)
        progress_bar.progress(progress)
        status_text.text(f"📊 Processing {image_name} - {method} ({idx+1}/{len(methods)})")
        
        # Apply enhancement
        enhanced_img = enhance_for_measurement(img, method)
        
        # Convert to numpy array for EasyOCR
        img_array = np.array(enhanced_img)
        
        # Extract text using EasyOCR
        easyocr_results = reader.readtext(img_array)
        
        # Combine all detected text
        all_text = ' '.join([text for _, text, conf in easyocr_results if conf > 0.5])
        
        # Extract measurements
        measurements = extract_length_measurements(all_text)
        all_measurements.update(measurements)
        
        # Store results
        results[method] = {
            'image': enhanced_img,
            'measurements': measurements,
            'raw_text': all_text
        }
    
    progress_bar.empty()
    status_text.empty()
    
    return results, all_measurements

def calculate_length_difference(measurements1, measurements2):
    """Calculate the length difference between two sets of measurements"""
    try:
        # Convert measurements to numeric values (remove 'm' and convert to float)
        def get_numeric_values(measurements):
            return [float(m.replace('m', '')) for m in measurements if m.replace('m', '').replace('.', '').isdigit()]
        
        nums1 = get_numeric_values(measurements1)
        nums2 = get_numeric_values(measurements2)
        
        if not nums1 and not nums2:
            return {
                'analysis_possible': False,
                'reason': 'No valid measurements found in either image'
            }
        elif not nums1:
            return {
                'analysis_possible': False,
                'reason': 'No valid measurements found in Image 1'
            }
        elif not nums2:
            return {
                'analysis_possible': False,
                'reason': 'No valid measurements found in Image 2'
            }
        
        # Get primary measurements (typically the largest/most significant)
        primary1 = max(nums1)  # Get the largest measurement from image 1
        primary2 = max(nums2)  # Get the largest measurement from image 2
        
        # Calculate difference
        difference = primary1 - primary2
        abs_difference = abs(difference)
        
        # Create readable format
        primary1_str = f"{primary1}m"
        primary2_str = f"{primary2}m"
        
        if difference > 0:
            diff_display = f"+{abs_difference}m"
            comparison_text = "Image 1 is longer than Image 2"
        elif difference < 0:
            diff_display = f"-{abs_difference}m"
            comparison_text = "Image 2 is longer than Image 1"
        else:
            diff_display = "0m"
            comparison_text = "Both images have equal length"
        
        # Calculate percentage difference
        percentage_diff = None
        if primary2 != 0:  # Avoid division by zero
            percentage = abs((difference / primary2) * 100)
            if percentage >= 0.1:  # Only show if significant
                percentage_diff = f"{percentage:.1f}%"
        
        return {
            'analysis_possible': True,
            'image1_primary': primary1_str,
            'image2_primary': primary2_str,
            'difference_value': difference,
            'difference_display': diff_display,
            'comparison_text': comparison_text,
            'percentage_difference': percentage_diff
        }
        
    except Exception as e:
        return {
            'analysis_possible': False,
            'reason': f'Error calculating difference: {str(e)}'
        }

def display_results_grid(results, image_name):
    """Display results in a grid format"""
    st.markdown(f"""
    <div class="results-section">
        <h2>🖼️ {image_name} Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 2 rows of 3 columns
    methods = ['Original', 'High Contrast', 'Cable Optimized', 'Denoised', 'Edge Enhanced', 'Handwriting Optimized']
    
    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            method_idx = row * 3 + col_idx
            if method_idx < len(methods):
                method = methods[method_idx]
                result = results[method]
                
                with cols[col_idx]:
                    st.markdown(f"**🎨 {method}**")
                    st.image(result['image'], use_column_width=True)
                    
                    if result['measurements']:
                        measurements_text = ', '.join(result['measurements'])
                        st.markdown(f"""
                        <div class="measurement-box">
                            📏 {measurements_text}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="no-measurement-box">
                            ❌ No measurements detected
                        </div>
                        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Dual Cable Measurement Extractor</h1>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin: 10px 0;">
            🚀 Powered by EasyOCR - Advanced Dual Image Analysis
        </div>
        <p>Upload 2 cable images to extract and compare length measurements with comprehensive analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load EasyOCR
    try:
        reader = load_easyocr()
        st.success("✅ EasyOCR initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing EasyOCR: {str(e)}")
        st.stop()
    
    # Sidebar with features
    with st.sidebar:
        st.markdown("### ✨ Features")
        st.markdown("""
        - 🤖 **EasyOCR Technology**: Neural network-based text recognition
        - 📏 **Smart Unit Conversion**: Automatically converts mm/cm to meters  
        - 🎨 **12 Total Enhancements**: 6 methods per image
        - 🔄 **Comparative Analysis**: Side-by-side measurement comparison
        - 🎯 **Precise Detection**: Focuses only on length measurements
        - 📊 **Comprehensive Results**: Individual and comparative summaries
        """)
        
        st.markdown("### 🎯 Supported Formats")
        st.markdown("""
        - 📐 **Direct meters**: "645m", "12.5 meter", "100 mtr"
        - 📏 **Millimeters**: "1000mm" → "1.000m"
        - 📊 **Centimeters**: "150cm" → "1.50m"
        - ✍️ **Handwritten and printed text** on cables
        """)
    
    # Upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Image 1</h3>
            <p>First cable image for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file1 = st.file_uploader("Choose first cable image", type=['png', 'jpg', 'jpeg'], key="file1")
    
    with col2:
        st.markdown("""
        <div class="upload-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>📁 Upload Image 2</h3>
            <p>Second cable image for comparison</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file2 = st.file_uploader("Choose second cable image", type=['png', 'jpg', 'jpeg'], key="file2")
    
    # Process button
    if uploaded_file1 is not None and uploaded_file2 is not None:
        if st.button("🔍 Analyze Both Images", key="analyze_btn"):
            # Load images
            image1 = Image.open(uploaded_file1)
            image2 = Image.open(uploaded_file2)
            
            st.markdown("---")
            st.markdown("## 🔄 Processing Images...")
            
            # Process both images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🖼️ Processing Image 1")
                results1, measurements1 = process_single_image(image1, reader, "Image 1")
                
            with col2:
                st.markdown("### 🖼️ Processing Image 2") 
                results2, measurements2 = process_single_image(image2, reader, "Image 2")
            
            st.markdown("---")
            
            # Display results
            display_results_grid(results1, "Image 1")
            st.markdown("---")
            display_results_grid(results2, "Image 2")
            
            # Comparison Summary
            length_difference = calculate_length_difference(measurements1, measurements2)
            common_measurements = list(measurements1.intersection(measurements2))
            common_measurements.sort(key=lambda x: float(x.replace('m', '')))
            
            st.markdown("""
            <div class="comparison-summary">
                <h3>📊 Comparative Analysis Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🖼️ Image 1 Summary")
                st.write(f"**Methods tested:** 6")
                st.write(f"**Measurements found:** {len(measurements1)}")
                if measurements1:
                    measurements_str = ', '.join(sorted(measurements1, key=lambda x: float(x.replace('m', ''))))
                    st.write(f"**Best measurements:** {measurements_str}")
                else:
                    st.write("**Best measurements:** None detected")
            
            with col2:
                st.markdown("#### 🖼️ Image 2 Summary")
                st.write(f"**Methods tested:** 6")
                st.write(f"**Measurements found:** {len(measurements2)}")
                if measurements2:
                    measurements_str = ', '.join(sorted(measurements2, key=lambda x: float(x.replace('m', ''))))
                    st.write(f"**Best measurements:** {measurements_str}")
                else:
                    st.write("**Best measurements:** None detected")
            
            # Length Difference Analysis
            st.markdown("#### 📏 Length Difference Analysis")
            if length_difference['analysis_possible']:
                st.write(f"**Image 1 Primary Length:** {length_difference['image1_primary']}")
                st.write(f"**Image 2 Primary Length:** {length_difference['image2_primary']}")
                st.write(f"**Difference:** {length_difference['difference_display']}")
                st.write(f"**Analysis:** {length_difference['comparison_text']}")
                if length_difference['percentage_difference']:
                    st.write(f"**Percentage Difference:** {length_difference['percentage_difference']}")
            else:
                st.warning(f"⚠️ {length_difference['reason']}")
            
            # Overall Comparison
            st.markdown("#### 🔄 Overall Comparison")
            st.write(f"**Total methods tested:** 12 (6 per image)")
            st.write(f"**Combined unique measurements:** {len(measurements1.union(measurements2))}")
            
            if common_measurements:
                st.write(f"**Common measurements:** {', '.join(common_measurements)}")
            else:
                st.write("**Common measurements:** None found")
    
    elif uploaded_file1 is not None or uploaded_file2 is not None:
        st.info("⚠️ Please upload both images for comparative analysis")
    else:
        st.info("📋 Upload both cable images to start dual measurement extraction and comparison")

if __name__ == "__main__":
    main()
