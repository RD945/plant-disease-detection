from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import cv2
from scipy import ndimage
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_model.keras'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
print("Loading model...")
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {os.path.abspath(MODEL_PATH)}")
        # Try alternate path
        alt_path = os.path.join(os.path.dirname(__file__), 'best_model.keras')
        if os.path.exists(alt_path):
            print(f"Found model at: {alt_path}")
            model = load_model(alt_path)
            print("Model loaded successfully from alternate path!")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} or {alt_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best_model.keras' is in the same directory as app.py")
    raise

# Class names (38 plant disease classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pil_to_base64(img):
    """Convert PIL Image to base64 string for web display"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# -------------------------------------------------------------------------
#  [IVP LAB 4 IMPLEMENTATION] - Quality Assessment (Blur Detection)
# -------------------------------------------------------------------------
def assess_quality(img_array):
    """
    Detects if an image is blurry using the Variance of Laplacian method.
    Source: IVP Lab 4 (Image Quality Assessment)
    
    Args:
        img_array: Numpy array of the image (RGB)
        
    Returns:
        dict: Contains boolean 'is_blurry' and the raw 'blur_score'
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate Variance of Laplacian
    # A low variance indicates few edges (blur), high variance indicates sharp edges
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Threshold: < 100 is typically considered blurry
    return {
        'is_blurry': score < 100,
        'blur_score': float(score)
    }

# -------------------------------------------------------------------------
#  [IVP LAB 8 & 5 IMPLEMENTATION] - Illumination Normalization
# -------------------------------------------------------------------------
def remove_shadows_HSV(img_pil):
    """
    Removes shadows and normalizes uneven lighting using HSV color space and CLAHE.
    Source: IVP Lab 8 (Color Models) & Lab 5 (Restoration)
    
    Args:
        img_pil: PIL Image object
        
    Returns:
        PIL Image object (Enhanced)
    """
    img_np = np.array(img_pil)
    
    # 1. Convert RGB to HSV (Lab 8)
    # We separate color (Hue/Saturation) from Intensity (Value)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2. Apply CLAHE to the V-channel only (Lab 5 Logic)
    # Contrast Limited Adaptive Histogram Equalization fixes lighting without shifting colors
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_enhanced = clahe.apply(v)
    
    # 3. Merge channels back
    hsv_enhanced = cv2.merge((h, s, v_enhanced))
    
    # 4. Convert back to RGB
    rgb = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(rgb)

# --- Existing Filters (Kept for backward compatibility) ---

def histogram_equalization(img):
    img_array = np.array(img)
    ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    ycbcr[:, :, 0] = cv2.equalizeHist(ycbcr[:, :, 0])
    enhanced = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(enhanced)

def contrast_stretching_log(img):
    img_array = np.array(img, dtype=np.float32)
    
    # BUG FIX: Prevent division by zero for empty/black images
    max_val = np.max(img_array)
    if max_val == 0:
        return img  # Return original if image is empty
    
    c = 255 / np.log(1 + max_val)
    log_transformed = c * np.log(1 + img_array)
    enhanced = np.clip(log_transformed, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)

def median_filter(img, kernel_size=5):
    img_array = np.array(img)
    filtered = cv2.medianBlur(img_array, kernel_size)
    return Image.fromarray(filtered)

def gaussian_smoothing(img, kernel_size=(5, 5), sigma=1.0):
    img_array = np.array(img)
    smoothed = cv2.GaussianBlur(img_array, kernel_size, sigma)
    return Image.fromarray(smoothed)

def homomorphic_filter(img, d0=30, gamma_h=2.0, gamma_l=0.5, c=1):
    # (Existing implementation kept as is)
    img_array = np.array(img, dtype=np.float32)
    result = np.zeros_like(img_array)
    for i in range(3):
        channel = img_array[:, :, i]
        channel = channel + 1.0
        log_channel = np.log(channel)
        dft = np.fft.fft2(log_channel)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows)
        v = np.arange(cols)
        u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
        d = np.sqrt(u**2 + v**2)
        h = (gamma_h - gamma_l) * (1 - np.exp(-c * (d**2 / d0**2))) + gamma_l
        filtered_dft = dft_shift * h
        idft_shift = np.fft.ifftshift(filtered_dft)
        filtered = np.fft.ifft2(idft_shift)
        filtered = np.real(filtered)
        result[:, :, i] = np.exp(filtered) - 1.0
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype(np.uint8)
    return Image.fromarray(result)

def apply_enhancement(img, enhancement_type):
    """Apply the specified enhancement algorithm to the image."""
    if enhancement_type == 'shadow_removal':
        return remove_shadows_HSV(img)  # NEW: Lab 8 Implementation
    elif enhancement_type == 'histogram_equalization':
        return histogram_equalization(img)
    elif enhancement_type == 'contrast_stretching':
        return contrast_stretching_log(img)
    elif enhancement_type == 'median_filter':
        return median_filter(img)
    elif enhancement_type == 'gaussian_smoothing':
        return gaussian_smoothing(img)
    elif enhancement_type == 'homomorphic_filter':
        return homomorphic_filter(img)
    else:
        return img 

# -------------------------------------------------------------------------
#  [IVP LAB 2 IMPLEMENTATION] - Test-Time Augmentation (TTA)
# -------------------------------------------------------------------------
def predict_with_tta(model, img_array):
    """
    Performs prediction using Test-Time Augmentation (Geometric Transformations).
    Source: IVP Lab 2 (Geometric Transformations)
    
    Instead of one prediction, we predict on 4 variations and average the result.
    """
    # img_array shape is (150, 150, 3), values 0-1 (floats)
    base_img = img_array
    
    batch_images = []
    
    # 1. Original
    batch_images.append(base_img)
    
    # 2. Horizontal Flip (Lab 2)
    batch_images.append(cv2.flip(base_img, 1))
    
    # Setup for rotation
    rows, cols = base_img.shape[:2]
    center = (cols/2, rows/2)
    
    # 3. Rotate +15 degrees (Lab 2)
    M_pos = cv2.getRotationMatrix2D(center, 15, 1)
    # BUG FIX: Added borderMode=cv2.BORDER_REFLECT to fill corners naturally
    batch_images.append(cv2.warpAffine(base_img, M_pos, (cols, rows), borderMode=cv2.BORDER_REFLECT))
    
    # 4. Rotate -15 degrees (Lab 2)
    M_neg = cv2.getRotationMatrix2D(center, -15, 1)
    # BUG FIX: Added borderMode=cv2.BORDER_REFLECT to fill corners naturally
    batch_images.append(cv2.warpAffine(base_img, M_neg, (cols, rows), borderMode=cv2.BORDER_REFLECT))
    
    # Convert list to batch array: shape (4, 150, 150, 3)
    batch_np = np.array(batch_images)
    
    # Predict on the batch
    preds = model.predict(batch_np)
    
    # Average the predictions across the batch
    avg_pred = np.mean(preds, axis=0)
    
    return avg_pred

def generate_preprocessing_advice(baseline_confidence, quality_metrics, selected_enhancement):
    """
    Generate smart preprocessing recommendations based on image quality and baseline confidence
    """
    advice = {
        'should_preprocess': False,
        'recommended_enhancement': 'none',
        'reason': '',
        'confidence_impact': 'neutral'
    }
    
    # High confidence (>80%) - Don't preprocess
    if baseline_confidence > 80:
        advice['should_preprocess'] = False
        advice['reason'] = f'Image already has high confidence ({baseline_confidence:.1f}%). Preprocessing may reduce accuracy.'
        
        if selected_enhancement != 'none':
            advice['confidence_impact'] = 'warning'
            advice['reason'] = f'Baseline confidence is {baseline_confidence:.1f}% without preprocessing. Using preprocessing may alter disease features and reduce accuracy.'
        else:
            advice['confidence_impact'] = 'positive'
            advice['reason'] = f'✓ Excellent confidence ({baseline_confidence:.1f}%). No preprocessing needed.'
        
        return advice
    
    # Medium confidence (50-80%) - Cautious preprocessing
    if baseline_confidence >= 50:
        advice['should_preprocess'] = True
        advice['confidence_impact'] = 'caution'
        
        # Check for specific issues
        if quality_metrics['is_blurry']:
            advice['recommended_enhancement'] = 'none'
            advice['reason'] = f'Image is blurry (score: {quality_metrics["blur_score"]:.1f}). Retake photo instead of preprocessing.'
        else:
            advice['recommended_enhancement'] = 'shadow_removal'
            advice['reason'] = f'Moderate confidence ({baseline_confidence:.1f}%). Try Shadow Removal if image has dark areas, otherwise use no preprocessing.'
        
        return advice
    
    # Low confidence (<50%) - Recommend preprocessing
    advice['should_preprocess'] = True
    advice['confidence_impact'] = 'recommended'
    
    # Check for blur first
    if quality_metrics['is_blurry']:
        advice['recommended_enhancement'] = 'none'
        advice['reason'] = f'❌ Image is too blurry (score: {quality_metrics["blur_score"]:.1f}). Please retake the photo with better focus.'
        return advice
    
    # Recommend based on confidence level
    if baseline_confidence < 30:
        advice['recommended_enhancement'] = 'shadow_removal'
        advice['reason'] = f'Low confidence ({baseline_confidence:.1f}%). Image may have shadows or poor lighting. Try Shadow Removal or retake photo in better lighting.'
    else:
        advice['recommended_enhancement'] = 'shadow_removal'
        advice['reason'] = f'Confidence is {baseline_confidence:.1f}%. Try Shadow Removal if image has visible shadows or dark spots.'
    
    return advice

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Get enhancement option
            enhancement = request.form.get('enhancement', 'none')
            
            # Read image file
            original_img = Image.open(io.BytesIO(file.read()))
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            # Store original image as base64
            original_img_resized = original_img.copy().resize((150, 150))
            original_base64 = pil_to_base64(original_img_resized)
            
            # --- STEP 0: GET BASELINE PREDICTION (without preprocessing) ---
            baseline_img = original_img.copy().resize((150, 150))
            baseline_array = image.img_to_array(baseline_img) / 255.0
            baseline_predictions = predict_with_tta(model, baseline_array)
            baseline_confidence = float(np.max(baseline_predictions) * 100)
            
            # --- STEP 1: QUALITY ASSESSMENT (Lab 4) ---
            # Check if the original image is suitable before processing
            quality_metrics = assess_quality(np.array(original_img))
            # Convert numpy bool to Python bool for JSON serialization
            quality_metrics['is_blurry'] = bool(quality_metrics['is_blurry'])
            
            # --- SMART PREPROCESSING ADVICE ---
            preprocessing_advice = generate_preprocessing_advice(
                baseline_confidence, 
                quality_metrics, 
                enhancement
            )
            
            # --- STEP 2: HANDLE "ALL PREPROCESSING" OPTION ---
            if enhancement == 'all_preprocessing':
                # Apply all 6 preprocessing methods to separate copies and predict each
                all_results = []
                enhancement_methods = [
                    ('none', 'No Enhancement'),
                    ('shadow_removal', 'Shadow Removal'),
                    ('histogram_equalization', 'Histogram Equalization'),
                    ('contrast_stretching', 'Contrast Stretching'),
                    ('median_filter', 'Median Filter'),
                    ('gaussian_smoothing', 'Gaussian Smoothing'),
                    ('homomorphic_filter', 'Homomorphic Filter')
                ]
                
                for method_id, method_name in enhancement_methods:
                    # Apply enhancement
                    if method_id == 'none':
                        processed_img = original_img.copy()
                    else:
                        processed_img = apply_enhancement(original_img.copy(), method_id)
                    
                    # Resize and predict
                    model_input = processed_img.resize((150, 150))
                    img_array = image.img_to_array(model_input) / 255.0
                    predictions = predict_with_tta(model, img_array)
                    
                    # Get prediction results
                    pred_class_idx = np.argmax(predictions)
                    pred_confidence = float(predictions[pred_class_idx] * 100)
                    
                    # Store result
                    all_results.append({
                        'method': method_name,
                        'method_id': method_id,
                        'prediction': CLASS_NAMES[pred_class_idx],
                        'prediction_idx': int(pred_class_idx),
                        'confidence': pred_confidence,
                        'image': pil_to_base64(model_input)
                    })
                
                # Detect prediction inconsistencies
                predictions_set = set(r['prediction'] for r in all_results)
                baseline_prediction = all_results[0]['prediction']  # 'none' is first
                
                # Check if methods disagree significantly
                disagreement_warning = None
                if len(predictions_set) > 3:  # More than 3 different predictions
                    disagreement_warning = {
                        'severity': 'high',
                        'message': f' WARNING: Preprocessing methods produced {len(predictions_set)} different predictions! Image quality is too poor for reliable diagnosis. Consider retaking the photo.',
                        'unique_predictions': len(predictions_set)
                    }
                elif len(predictions_set) > 1:  # Some disagreement
                    wrong_predictions = [r for r in all_results if r['prediction'] != baseline_prediction]
                    if wrong_predictions:
                        disagreement_warning = {
                            'severity': 'medium',
                            'message': f' CAUTION: Some preprocessing methods changed the prediction. The image may have severe degradation that\'s confusing the model.',
                            'wrong_methods': [r['method'] for r in wrong_predictions]
                        }
                
                # Return comparison results
                result = {
                    'success': True,
                    'comparison_mode': True,
                    'all_results': all_results,
                    'disagreement_warning': disagreement_warning,
                    'quality_check': quality_metrics,
                    'baseline_confidence': baseline_confidence,
                    'original_image': original_base64
                }
            else:
                # --- STEP 2: SINGLE ENHANCEMENT (Lab 8/5) ---
                enhanced_img = original_img.copy()
                if enhancement and enhancement != 'none':
                    enhanced_img = apply_enhancement(enhanced_img, enhancement)
                
                # Prepare display version of enhanced image
                enhanced_img_display = enhanced_img.copy().resize((150, 150))
                enhanced_base64 = pil_to_base64(enhanced_img_display)
                
                # --- STEP 3: PREPROCESSING ---
                # Resize and Normalize
                model_input_img = enhanced_img.resize((150, 150))
                img_array = image.img_to_array(model_input_img)
                img_array = img_array / 255.0  # Normalize to 0-1
                
                # --- STEP 4: INFERENCE WITH TTA (Lab 2) ---
                # Pass single image array (150, 150, 3) to TTA function
                predictions_avg = predict_with_tta(model, img_array)
                
                # Get class and confidence from the AVERAGED prediction
                predicted_class_idx = np.argmax(predictions_avg)
                confidence = float(predictions_avg[predicted_class_idx] * 100)
                
                # Get top 3 predictions
                top_3_idx = np.argsort(predictions_avg)[-3:][::-1]
                top_3_predictions = [
                    {
                        'class': CLASS_NAMES[idx],
                        'confidence': float(predictions_avg[idx] * 100)
                    }
                    for idx in top_3_idx
                ]
                
                result = {
                    'success': True,
                    'comparison_mode': False,
                    'prediction': CLASS_NAMES[predicted_class_idx],
                    'confidence': confidence,
                    'baseline_confidence': baseline_confidence,
                    'top_3': top_3_predictions,
                    'enhancement_applied': enhancement,
                    'quality_check': quality_metrics,
                    'inference_method': 'Test-Time Augmentation (4-view Ensemble)',
                    'preprocessing_advice': preprocessing_advice,
                    'original_image': original_base64,
                    'enhanced_image': enhanced_base64
                }
            
            return jsonify(result)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)