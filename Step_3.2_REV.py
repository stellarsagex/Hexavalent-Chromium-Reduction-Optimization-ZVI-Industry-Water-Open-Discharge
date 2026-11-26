# Implementing Two Point Correlation bf Fucntion

import cv2
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# 1. Loading and segmenting to ensure identical matrix structure
img = cv2.imread("Pore_Screenshot_Step_2.png", cv2.IMREAD_GRAYSCALE)
thresholds = threshold_multiotsu(img, classes=3)
regions = np.digitize(img, bins=thresholds)
binary_solid = regions >= 1

# Calibrating spatial resolution from previous step output
nm_per_pixel = 0.6349 

print("Calculating Two-Point Correlation Blueprint.")

# 2. Computing the S2(r) spatial statistic on the solid iron phase
s2_profile = ps.metrics.two_point_correlation(im=binary_solid)

# 3. Converting voxel spacing into physical nanometers
physical_distances_nm = s2_profile.distance * nm_per_pixel

# 4. Saving the target spatial fingerprint out as a text file
blueprint_data = np.column_stack((physical_distances_nm, s2_profile.probability))
np.savetxt("Step_3.2_Spatial_Fingerprint.txt", blueprint_data, header="Distance_nm Probability_S2")
print("Success!")

# 5. Plotting the blueprint curve
plt.figure(figsize=(8, 5))
plt.plot(physical_distances_nm, s2_profile.probability, 'r-', linewidth=2.5, label="TEM Micrograph Fingerprint")
plt.title("Two-Point Correlation Blueprint $S_2(r)$ for nZVI Filter")
plt.xlabel("Separation Distance (nm)")
plt.ylabel("Probability $S_2$")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
