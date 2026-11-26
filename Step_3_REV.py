# Running Multi Otsu Auto Thresholding Algorithm to refine the TEM Screenshot

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# 1. Loading the crop image in grayscale
img = cv2.imread("Pore_Screenshot_Step_2.png", cv2.IMREAD_GRAYSCALE)

# 2. Applying Multi-Otsu thresholding (finding 2 thresholds to create 3 classes)
thresholds = threshold_multiotsu(img, classes=3)

# 3. Segmenting the image into zones
# Region 0: Dark background void
# Region 1: Intermediate shadows
# Region 2: Bright foreground nanoparticles
regions = np.digitize(img, bins=thresholds)

# Combining Regions 1 and 2 to represent the solid phase (True = Solid ZVI, False = Pore Void)
# This prevents losing the depth of the complex nano-iron clusters!
binary_solid = regions >= 1

# 4. Visualizing the results to check accuracy
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original Micrograph Screenshot")
ax[0].axis('off')

ax[1].imshow(binary_solid, cmap='binary_r') # Black = Void, White = Solid Iron
ax[1].set_title("Multi-Otsu Segmented Solid Phase")
ax[1].axis('off')

plt.tight_layout()
plt.show()
