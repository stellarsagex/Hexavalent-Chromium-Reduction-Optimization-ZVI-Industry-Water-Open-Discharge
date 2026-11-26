#Understanding the Voxels, Pixels and Metrics

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

# 1. Loading the Screenshot
img = cv2.imread("Pore_Screenshot_Step_2.png", cv2.IMREAD_GRAYSCALE)

# 2. Running the exact same Multi-Otsu segmentation from before
thresholds = threshold_multiotsu(img, classes=3)
regions = np.digitize(img, bins=thresholds)
binary_solid = regions >= 1

# 3. Calculating the metrics
pixel_height, pixel_width = binary_solid.shape
total_2d_pixels = binary_solid.size

# Mapping the 400 nm (approximate size) physical constraint to pixel spacing
nm_per_pixel = 400.0 / pixel_width

solid_pixels = np.sum(binary_solid == 1)
void_pixels = np.sum(binary_solid == 0)

solid_fraction = solid_pixels / total_2d_pixels
porosity = void_pixels / total_2d_pixels

# --- Printing the complete structural report ---
print("="*50)
print("2D SCREENSHOT PIXEL METRICS")
print("="*50)
print(f"2D Image Resolution : {pixel_width} x {pixel_height} pixels")
print(f"Total Pixels        : {total_2d_pixels:,} pixels")
print(f"Spatial Resolution  : {nm_per_pixel:.4f} nm per pixel")
print(f"Solid ZVI Pixels    : {solid_pixels:,} ({solid_fraction*100:.2f}%)")
print(f"Void Space Porosity : {void_pixels:,} ({porosity*100:.2f}%)")
print("="*50)
