# LBM Validation using JAX-LaB

import jax
import jax.numpy as jnp
import numpy as np

# 1. Configuration matching the 512^3 RAW matrix
grid_size = 512
raw_filename = "Step_3.3_Geometry_512.raw"

print("Reading 512^3 RAW voxel geometry into JAX array memory.")
grid_flat = np.fromfile(raw_filename, dtype=np.int8)
geometry_mask = jnp.array(grid_flat.reshape((grid_size, grid_size, grid_size)))

# Creating explicit Boolean phase maps (1 = Fluid Void space, 0 = Solid Iron)
fluid_mask = (geometry_mask == 1)
solid_mask = (geometry_mask == 0)

# 2. STANDARD D3Q19 LATTICE SPECIFICATIONS
cx = jnp.array([0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0])
cy = jnp.array([0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1])
cz = jnp.array([0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1])
weights = jnp.array([1/3, 1/18,1/18,1/18,1/18,1/18,1/18, 1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36])

# Providing pre-evaluated python integer lists for the shift loops
# This prevents JAX from having to evaluate tracers inside the rolling operation
shifts_x = [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0]
shifts_y = [0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1]
shifts_z = [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1]

# Opposing direction mapping index for no-slip solid bounce-back reflections
noslip_map = jnp.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])

# 3. SETTING BOUNDARY CONDITIONS AND RECENT LITERATURE CONSTANTS
# According to recent papers, keeping lattice velocities (u_inlet) around 0.01 
# prevents numerical instability and keeps the Reynolds number safely in the laminar regime.
u_inlet = 0.01  
tau = 0.6       # Relaxation parameter tracking water's kinematic viscosity
omega = 1.0 / tau

print("Initializing 3D distribution field tensors.")
# Initializing standard equilibrium distribution states across the fluid matrix channels
f = jnp.zeros((19, grid_size, grid_size, grid_size))
for i in range(19):
    f = f.at[i].set(jnp.where(fluid_mask, weights[i] * 1.0, 0.0))

# 4. COMPILED LBM CORE STEP FUNCTION (WITH BOUNDARY CONDITIONS)

@jax.jit
def validation_step(f_in):
    # A. Macro scale tracking
    rho = jnp.sum(f_in, axis=0)
    ux = jnp.sum(f_in * cx[:, None, None, None], axis=0) / (rho + 1e-8)
    uy = jnp.sum(f_in * cy[:, None, None, None], axis=0) / (rho + 1e-8)
    uz = jnp.sum(f_in * cz[:, None, None, None], axis=0) / (rho + 1e-8)

    # B. Applying Boundary Conditions at Inlet (X=0) and Outlet (X=511)
    # Zou-He / Equilibrium style formulation forcing macro values at faces
    ux = ux.at[0, :, :].set(u_inlet)
    ux = jnp.where(fluid_mask, ux, 0.0)
    uy = jnp.where(fluid_mask, uy, 0.0)
    uz = jnp.where(fluid_mask, uz, 0.0)
    rho = jnp.where(fluid_mask, rho, 0.0)

    # C. Collision Operator (BGK Formulation)
    f_post = f_in
    u_sq = ux**2 + uy**2 + uz**2
    for i in range(19):
        cu = cx[i]*ux + cy[i]*uy + cz[i]*uz
        feq = weights[i] * rho * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*u_sq)
        f_post = f_post.at[i].set(f_in[i] + omega * (feq - f_in[i]))

    # D. Streaming Operator (3D Spatial Shifts)
    f_str = jnp.zeros_like(f_post)
    for i in range(19):
        # FIX 2: Replaced int(cx[i]) with the static integer lists shifts_x[i]
        f_str = f_str.at[i].set(
            jnp.roll(
                jnp.roll(
                    jnp.roll(f_post[i], shifts_x[i], axis=0), 
                    shifts_y[i], axis=1
                ), 
                shifts_z[i], axis=2
            )
        )

    # E. Internal Solid Phase Bounce-Back Rule (No-slip walls)
    f_out = f_str
    for i in range(19):
        f_out = f_out.at[i].set(jnp.where(fluid_mask, f_str[i], f_post[noslip_map[i]]))

    return f_out, ux, rho

# 5. INITIAL ITERATION DRY-RUN TO JIT COMPILE
print("Compiling JAX graph engine.")
f, ux, rho = validation_step(f)
print("="*50)
print("3D JAX-LAB BOUNDARY INITIALIZATION COMPLETED")
print("="*50)
print(f"Lattice Boundary Speed : {u_inlet} lu/s")
print(f"Viscosity Relaxation   : {tau} (Omega: {omega:.2f})")
print(f"Validation Tensor Code : Graph compiled successfully.")
print("="*50)
