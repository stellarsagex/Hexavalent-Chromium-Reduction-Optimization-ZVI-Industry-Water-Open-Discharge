# =========================================================
# File: Step_7_JAX_LBM_Physics.py
# =========================================================
import jax
import jax.numpy as jnp
import numpy as np

# 1. Physical and Domain Settings
grid_size = 512
raw_filename = "Step_3.3_Geometry_512.raw"

print("Loading geometry mask into JAX array memory...")
grid_flat = np.fromfile(raw_filename, dtype=np.int8)
grid_3d = grid_flat.reshape((grid_size, grid_size, grid_size))
geometry_mask = jnp.array(grid_3d)

# True for Fluid Void channels (1), False for Solid Iron walls (0)
fluid_mask = (geometry_mask == 1)
solid_mask = (geometry_mask == 0)

# =========================================================
# 2. DEFINE D3Q19 LATTICE CONSTANTS
# =========================================================
# The 19 directional vectors for a D3Q19 lattice model
cx = jnp.array([0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0])
cy = jnp.array([0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1])
cz = jnp.array([0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1])

# Lattice directional weights (wi) summing up exactly to 1.0
weights = jnp.array([
    1/3, 
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36
])

# Speed of sound squared in lattice units
cs_sq = 1.0 / 3.0

# =========================================================
# 3. INITIALIZE FLUID PROPERTIES & VELOCITY DISTRIBUTIONS
# =========================================================
print("Allocating LBM distribution functions...")

# Total density initially uniform at 1.0 lattice units
rho = jnp.where(fluid_mask, 1.0, 0.0)

# Velocity components (u_x, u_y, u_z) starting completely at rest (0.0)
ux = jnp.zeros_like(rho)
uy = jnp.zeros_like(rho)
uz = jnp.zeros_like(rho)

# Initialize the 19 velocity distribution arrays (f) based on equilibrium
# shape layout: (19, 512, 512, 512) -> perfectly structured for JAX parallel arrays
f = jnp.zeros((19, grid_size, grid_size, grid_size))

for i in range(19):
    # Standard LBM Equilibrium equation initialization at rest
    f = f.at[i].set(weights[i] * rho)

# --- Verification Check ---
allocated_gb = (f.nbytes + rho.nbytes * 4) / (1024**3)
print("="*50)
print("          LBM PHYSICAL PROPERTY INITIALIZATION       ")
print("="*50)
print(f"Distribution Shape    : {f.shape}")
print(f"Fluid Core Density    : {jnp.max(rho)}")
print(f"Lattice Directions    : D3Q19 Model Activated")
print(f"Approx VRAM Allocated : {allocated_gb:.2f} GB")
print("="*50)
print("Physics nodes successfully established.")
