#Creating a Convergence Curve
#Finding REV based on porosity needed for the study via PoreSpy library

import numpy as np
import porespy as ps
import scipy.stats as spst
import matplotlib.pyplot as plt

# 1. Defining domain size (using a 200x200x200 voxel cube)
shape = [200, 200, 200]

# 2. Defining varying nZVI size distribution using a scipy.stats object
zvi_distribution = spst.lognorm(s=0.4, scale=np.exp(2.5))

# 3. Generating the varying ZVI geometry
zvi_matrix = ps.generators.polydisperse_spheres(
    shape=shape, 
    porosity=0.4, 
    dist=zvi_distribution,
    r_min=3
)

# 4. Automatically analyzing the REV threshold using the correct updated function
rev_data = ps.metrics.rev_porosity(im=zvi_matrix)

# 5. Plotting the result to find where the geometry stabilizes
# Passing *rev_data directly unpacks the volume and porosity properties
plt.plot(*rev_data, 'b.')
plt.xlabel("Sub-volume Size (voxels)")
plt.ylabel("Measured Porosity")
plt.title("REV Convergence Curve for Varying nZVI")
plt.grid(True)
plt.show()
