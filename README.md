# Plant Disease Detection Flask App

A Flask web application for detecting plant diseases using a CNN model trained on 38 different plant disease classes with 97% accuracy.

This project distinguishes itself by including an advanced **Image Enhancement Engine** and a **Testing/Validation Suite** to demonstrate how preprocessing algorithms can recover accuracy on degraded real-world images (shadows, noise, poor lighting).

## Features

- **Multi-Class Detection**: Detects 38 different plant diseases across 14 plant species.
- **Smart Preprocessing**: Users can apply specific image enhancement algorithms before analysis.
- **Automatic Quality Check**:
  - Analyzes image blur (Laplacian variance).
  - Provides "Smart Recommendations" on which enhancement to use based on confidence scores.
- **Drag-and-Drop Interface**: User-friendly web UI.
- **Advanced Algorithms**: Includes Histogram Equalization, Contrast Stretching, Median Filtering, Gaussian Smoothing, Homomorphic Filtering, and Shadow Removal.
- **Testing Suite**: Includes a dedicated script (`create_degraded_images.py`) to generate test data for validating the model's robustness.

## Plant Species and Diseases Covered

The model is trained to recognize the following:

| Plant | Diseases/Conditions |
|-------|---------------------|
| **Apple** | Scab, Black rot, Cedar apple rust, Healthy |
| **Corn** | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| **Grape** | Black rot, Esca, Leaf blight, Healthy |
| **Potato** | Early blight, Late blight, Healthy |
| **Tomato** | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria, Spider mites, Target Spot, Mosaic virus, Yellow Leaf Curl, Healthy |
| **Others** | Blueberry, Cherry, Orange (Citrus greening), Peach, Pepper, Raspberry, Soybean, Squash, Strawberry |

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file is present**

   * Make sure `best_model.keras` is in the root directory.
   * The model should be the trained CNN model from the notebook.

## Usage

### Starting the Application

1. **Start the Flask server**

   ```bash
   python app.py
   ```

2. **Open your browser**

   * Navigate to `http://localhost:5000`

### Using the Interface

1. **Upload an image**: Supports PNG, JPG, JPEG (Max 16MB).
2. **Review Automatic Recommendation**:

   * **Green**: "Excellent confidence. No preprocessing needed."
   * **Yellow**: "Moderate confidence. Try Shadow Removal if image has dark areas."
   * **Red**: "Low confidence. Preprocessing recommended."
   * **Quality Warning**: "Image is too blurry (score < 100). Please retake."
3. **Choose enhancement**: Select based on the recommendation (e.g., Shadow Removal).
4. **Get predictions**: View the top 3 predictions and confidence scores.

## Image Enhancement Algorithms

The system includes six optional preprocessing algorithms. Here is how they work and when to use them:

### 1. Shadow Removal (HSV+CLAHE) — RECOMMENDED

* **Method**: Converts RGB to HSV color space. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the V (brightness) channel only.
* **Why it works**: Separates color information (disease symptoms) from brightness information (shadows).
* **Best For**: Heavy shadows, uneven lighting, dark spots.

### 2. Histogram Equalization

* **Method**: Converts RGB to YCbCr, applies equalization to the Y (luminance) channel only, then converts back.
* **Why it works**: Spreads out intensity distribution.
* **Best For**: Images with poor overall contrast (washed out or flat).

### 3. Contrast Stretching (Log Transform)

* **Method**: Applies logarithmic transformation `s = c * log(1 + r)` where `r` is pixel intensity.
* **Why it works**: Expands dynamic range of dark regions.
* **Best For**: Very dark or underexposed images.

### 4. Median Filtering

* **Method**: Non-linear 5×5 spatial filter.
* **Why it works**: Replaces each pixel with the median of its neighbors.
* **Best For**: Salt-and-pepper noise.

### 5. Gaussian Smoothing

* **Method**: Convolution with a Gaussian kernel.
* **Why it works**: Removes high-frequency noise.
* **Best For**: Grainy images, sensor noise.

### 6. Homomorphic Filtering

* **Method**: Uses frequency-domain High-Pass Butterworth filter.
* **Why it works**: Separates illumination from reflectance.
* **Best For**: Uneven lighting or vignetting.

## API Endpoints

### `POST /predict`

Accepts an image file and returns predictions with optional enhancement.

**Request:**

* Method: POST
* Content-Type: multipart/form-data
* Body:

  * `file`: image file
  * `enhancement`: optional — one of
    `none`, `histogram_equalization`, `contrast_stretching`,
    `median_filter`, `gaussian_smoothing`, `homomorphic_filter`,
    `shadow_removal`

**Response Example:**

```json
{
  "success": true,
  "prediction": "Tomato___Late_blight",
  "confidence": 0.98,
  "enhancement_applied": "histogram_equalization",
  "top_3": [
    { "class": "Tomato___Late_blight", "confidence": 0.98 },
    { "class": "Tomato___Early_blight", "confidence": 0.015 },
    { "class": "Tomato___Septoria_leaf_spot", "confidence": 0.003 }
  ]
}
```

### `GET /health`

Health check endpoint.

**Response Example:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Testing & Validation Suite

This project includes `create_degraded_images.py` to generate artificially degraded images for validation.

### Generating Test Data

Run:

```bash
python create_degraded_images.py --input_folder ./clean_images --output_folder ./degraded_test
```

**Per-image outputs generated:**

1. Original
2. Shadowed
3. Blurred
4. Gaussian Noise
5. Salt & Pepper Noise
6. Uneven Lighting
7. Dark
8. Combined (stress test)

### Validation Workflow

| Image Degradation | Recommended Enhancement    | Expected Outcome         |
| ----------------- | -------------------------- | ------------------------ |
| Shadow_added      | Shadow Removal (HSV+CLAHE) | +30–40% confidence       |
| Gaussian_noise    | Gaussian Smoothing         | +35–45% confidence       |
| Saltpepper_noise  | Median Filter              | +40–50% confidence       |
| Uneven_light      | Homomorphic Filter         | +25–35% confidence       |
| Darkened          | Contrast Stretching        | +40–50% confidence       |
| Combined_effects  | Shadow Removal             | Large improvement        |
| Motion_blur       | None                       | Triggers Quality Warning |

### Common Testing Mistakes

1. **Don’t preprocess clean images** — accuracy drops.
2. **Don't stack enhancements** — destroys features.
   Use the app’s comparison view instead.

## Project Structure

```
.
├── app.py                      # Main Flask application
├── best_model.keras            # Trained model file
├── create_degraded_images.py   # Script for generating test data
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web interface
├── uploads/                    # Uploaded images folder (auto-created)
└── README.md                   # This file
```

## Deployment

### Local Development

```bash
python app.py
```

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Future Enhancements

* [ ] Batch prediction support
* [ ] More robust image augmentation
* [ ] Disease treatment recommendations
* [ ] Multi-language UI
* [ ] Mobile app version
* [ ] Database integration

## License

This project is provided as-is for educational and research purposes.
