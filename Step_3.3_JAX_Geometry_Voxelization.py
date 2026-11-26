# JAX based Geometry Voxelization

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration (Setting to 512 for transport REV)
grid_size = 512  
target_porosity = 0.4452  # From previous step metrics
nm_per_voxel = 0.6349     # From previous step metrics

print(f"Using JAX Device: {jax.devices()[0]}")
print(f"Generating a {grid_size}^3 ultra-realistic 3D nZVI geometry via GPU.")

# 2. Loading the target correlation blueprint curve
blueprint = np.loadtxt("Step_3.2_Spatial_Fingerprint.txt")
dist_nm = blueprint[:, 0]
prob_s2 = blueprint[:, 1]

# Estimating the structural correlation length in voxels (from the 150nm flattening point from the graph of previous step)
correlation_length_voxels = (150.0 / nm_per_voxel)

# 3. Generating a 3D White Noise field on the GPU
key = jax.random.PRNGKey(42)  # Seed for reproducibility
white_noise = jax.random.normal(key, shape=(grid_size, grid_size, grid_size))

# 4. Constructing a 3D Spectral Filter Matrix in parallel
# Establishing a distance matrix in the frequency domain matching the correlation length
coords = jnp.fft.fftfreq(grid_size) * grid_size
X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing='ij')
radius_grid = jnp.sqrt(X**2 + Y**2 + Z**2)

# Applying a Gaussian filter profile in the frequency domain 
# (Equivalent to convolving with the spatial correlation length)
filter_sigma = correlation_length_voxels / (2 * jnp.pi)
spectral_filter = jnp.exp(-(radius_grid**2) / (2 * (filter_sigma**2)))

# 5. Performing GPU-Accelerated 3D FFT Convolutions
noise_fft = jnp.fft.fftn(white_noise)
filtered_fft = noise_fft * spectral_filter
correlated_field = jnp.fft.ifftn(filtered_fft).real

# 6. JAX Percentile Slicing to Match Target Porosity (44.52% void space)
# Flattenning the matrix to find the threshold barrier value
flattened_field = jnp.ravel(correlated_field)
sorted_field = jnp.sort(flattened_field)
threshold_idx = int(target_porosity * sorted_field.size)
threshold_value = sorted_field[threshold_idx]

# Final 3D Binary Mask Matrix: 1 = Pore Fluid, 0 = Solid Iron
# Explicitly forcing it into int8 format to save memory space
zvi_matrix_3d_jax = (correlated_field <= threshold_value).astype(jnp.int8)

# 7. Writing directly to disk as a lightweight raw binary file
# Converting to a numpy array briefly allows standard disk output stream writing
output_name = f"Step_3.3_Geometry_{grid_size}.raw"
np.array(zvi_matrix_3d_jax).tofile(output_name)

# --- Computing and Printing Validation Metrics ---
fluid_voxels = int(jnp.sum(zvi_matrix_3d_jax == 1))
solid_voxels = int(jnp.sum(zvi_matrix_3d_jax == 0))
generated_porosity = fluid_voxels / zvi_matrix_3d_jax.size

print("="*50)
print("JAX 3D GEOMETRY GENERATION REPORT")
print("="*50)
print(f"File Saved As      : {output_name}")
print(f"Matrix Dimensions  : {grid_size} x {grid_size} x {grid_size}")
print(f"Total 3D Voxels    : {zvi_matrix_3d_jax.size:,} voxels")
print(f"Target Porosity    : {target_porosity*100:.2f}%")
print(f"Generated Porosity : {generated_porosity*100:.2f}%")
print(f"Solid Iron Voxels  : {solid_voxels:,} voxels")
print(f"Fluid Void Voxels  : {fluid_voxels:,} voxels")
print("="*50)

# 8. Extracting a 2D slice from the GPU matrix to plot and verify
middle_slice_idx = grid_size // 2
visual_slice = np.array(zvi_matrix_3d_jax[middle_slice_idx, :, :])

plt.figure(figsize=(6, 6))
plt.imshow(visual_slice, cmap='binary_r') # Black = Void, White = Iron
plt.title(f"JAX 2D Voxel Slice ({grid_size}^3 Domain)")
plt.axis('off')
plt.tight_layout()
plt.show()
