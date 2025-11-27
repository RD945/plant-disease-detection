"""
Image Degradation Script for Testing Preprocessing Effectiveness
-----------------------------------------------------------------
This script takes clean plant disease images and artificially degrades them
by adding shadows, blur, noise, and poor lighting. These degraded images will
benefit from preprocessing techniques like Shadow Removal, CLAHE, and filtering.

Usage:
    python create_degraded_images.py --input_folder ./clean_images --output_folder ./degraded_images

The script creates 5 versions of each image:
1. Original (baseline)
2. With shadows (benefits from Shadow Removal)
3. With blur (benefits from Median/Gaussian filters)
4. With noise (benefits from Median filter)
5. With uneven lighting (benefits from Homomorphic/CLAHE)
"""

import cv2
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path

def add_shadow(img_array):
    """
    Add realistic shadow to simulate outdoor conditions
    This creates images that benefit from Shadow Removal preprocessing
    """
    rows, cols = img_array.shape[:2]
    
    # Create gradient shadow mask
    shadow_mask = np.zeros((rows, cols), dtype=np.float32)
    
    # Create diagonal shadow gradient
    for i in range(rows):
        for j in range(cols):
            # Distance from top-left creates gradient
            distance = np.sqrt((i/rows)**2 + (j/cols)**2)
            shadow_mask[i, j] = 0.3 + 0.7 * distance
    
    # Apply shadow to each channel
    shadowed = img_array.copy().astype(np.float32)
    for c in range(3):
        shadowed[:, :, c] *= shadow_mask
    
    return np.clip(shadowed, 0, 255).astype(np.uint8)

def add_blur(img_array, kernel_size=15):
    """
    Add motion blur to simulate camera shake
    This creates images that benefit from sharpening or quality warnings
    """
    # Create motion blur kernel (horizontal motion)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Apply blur
    blurred = cv2.filter2D(img_array, -1, kernel)
    return blurred

def add_gaussian_noise(img_array, mean=0, sigma=25):
    """
    Add Gaussian noise to simulate sensor noise in low light
    This creates images that benefit from Gaussian Smoothing
    """
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy = img_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img_array, amount=0.02):
    """
    Add salt-and-pepper noise to simulate transmission errors
    This creates images that benefit from Median Filter
    """
    noisy = img_array.copy()
    num_salt = int(amount * img_array.size * 0.5)
    num_pepper = int(amount * img_array.size * 0.5)
    
    # Add salt (white pixels)
    coords = [np.random.randint(0, i-1, num_salt) for i in img_array.shape[:2]]
    noisy[coords[0], coords[1], :] = 255
    
    # Add pepper (black pixels)
    coords = [np.random.randint(0, i-1, num_pepper) for i in img_array.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy

def add_uneven_lighting(img_array):
    """
    Simulate uneven/poor lighting conditions
    This creates images that benefit from Histogram Equalization or Homomorphic Filter
    """
    rows, cols = img_array.shape[:2]
    
    # Create vignette effect (darker at edges)
    center_x, center_y = cols // 2, rows // 2
    
    lighting_mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            max_distance = np.sqrt(center_y**2 + center_x**2)
            lighting_mask[i, j] = 1.0 - 0.6 * (distance / max_distance)
    
    # Apply lighting
    uneven = img_array.copy().astype(np.float32)
    for c in range(3):
        uneven[:, :, c] *= lighting_mask
    
    return np.clip(uneven, 0, 255).astype(np.uint8)

def darken_image(img_array, factor=0.5):
    """
    Make image darker to simulate poor lighting
    Benefits from Contrast Stretching
    """
    darkened = img_array.astype(np.float32) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)

def process_image(input_path, output_folder):
    """
    Process a single image and create all degraded versions with descriptive names
    """
    # Load image
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    
    # Get filename without extension
    base_name = Path(input_path).stem
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define degradations with descriptive names
    degradations = {
        # Single degradations
        'shadow_added': add_shadow(img_array),
        'motion_blur_applied': add_blur(img_array),
        'gaussian_noise_added': add_gaussian_noise(img_array),
        'saltpepper_noise_added': add_salt_pepper_noise(img_array),
        'uneven_light_simulated': add_uneven_lighting(img_array),
        'darkened': darken_image(img_array),
        
        # Combined degradations (multiple effects on same image)
        'shadow_and_noise': add_shadow(add_gaussian_noise(img_array)),
        'shadow_and_dark': add_shadow(darken_image(img_array)),
        'blur_and_noise': add_blur(add_gaussian_noise(img_array)),
        'uneven_light_and_noise': add_uneven_lighting(add_gaussian_noise(img_array)),
        'shadow_blur_noise': add_shadow(add_blur(add_gaussian_noise(img_array))),
        'all_effects_combined': add_shadow(add_gaussian_noise(add_uneven_lighting(darken_image(img_array)))),
        
        # Severe degradations
        'heavy_shadow': add_shadow(add_shadow(img_array)),
        'heavy_blur': add_blur(img_array, kernel_size=25),
        'heavy_noise': add_gaussian_noise(img_array, sigma=50),
        'very_dark': darken_image(img_array, factor=0.3),
    }
    
    # Save all versions with descriptive names
    saved_count = 0
    for degradation_name, degraded_img in degradations.items():
        output_filename = f"{base_name}__{degradation_name}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        Image.fromarray(degraded_img).save(output_path, quality=95)
        print(f"✓ Created: {output_filename}")
        saved_count += 1
    
    return saved_count

def main():
    parser = argparse.ArgumentParser(
        description='Create degraded images for testing preprocessing effectiveness'
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        default='test',
        help='Folder containing clean input images (default: test)'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='test_p',
        help='Folder to save degraded images (default: test_p)'
    )
    parser.add_argument(
        '--image_extensions',
        type=str,
        default='jpg,jpeg,png',
        help='Comma-separated list of image extensions to process'
    )
    
    args = parser.parse_args()
    
    # Get list of extensions
    extensions = [f".{ext.strip()}" for ext in args.image_extensions.split(',')]
    
    # Find all images in input folder
    input_path = Path(args.input_folder)
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {args.input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output folder: {args.output_folder}")
    print("-" * 60)
    
    total_created = 0
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        try:
            count = process_image(str(img_path), args.output_folder)
            total_created += count
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print("-" * 60)
    print(f"\n✓ Successfully created {total_created} degraded images in '{args.output_folder}'!")
    print(f"\nImage naming format: originalname__degradation_type.jpg")
    print(f"\nTest these images with your Flask app:")
    print(f"  • 'shadow_added' → use Shadow Removal (HSV+CLAHE)")
    print(f"  • 'motion_blur_applied' → should trigger quality warning")
    print(f"  • 'gaussian_noise_added' → use Gaussian Smoothing")
    print(f"  • 'saltpepper_noise_added' → use Median Filter")
    print(f"  • 'uneven_light_simulated' → use Homomorphic Filter")
    print(f"  • 'darkened' → use Contrast Stretching")
    print(f"  • 'shadow_and_noise' → use Shadow Removal")
    print(f"  • 'all_effects_combined' → use Shadow Removal (best for multiple issues)")
    print(f"  • 'heavy_*' → severe degradations for stress testing")

if __name__ == "__main__":
    main()
