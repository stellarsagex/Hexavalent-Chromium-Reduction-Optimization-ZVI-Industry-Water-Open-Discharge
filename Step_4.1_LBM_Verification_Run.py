
# LBM Verification
#
# D3Q19 BGK Lattice Boltzmann Solver
# 
# Features:
#   1. Reads a 512^3 RAW geometry file using memory mapping.
#   2. Extracts a smaller 64^3 REV subdomain.
#   3. Implements D3Q19 BGK collision.
#   4. Implements Guo body-force forcing.
#   5. Implements streaming and on-site bounce-back.
#   6. Uses periodic outer boundaries.
#   7. Reports velocity and mass conservation.
#
# Geometry convention:
#   1 = Fluid
#   0 = Solid ZVI
#
# This is a hydrodynamic verification solver.
# It does NOT yet include reactive transport or PHREEQC.
# ============================================================

import os

import time
import numpy as np
import jax
import jax.numpy as jnp


# ============================================================
# 1. CONFIGURATION
# ============================================================

RAW_FILE = "Step_3.3_Geometry_512.raw"

RAW_SIZE = 512

# Starting small. Increasing only after successful verification.
GRID_SIZE = 64

MAX_ITERATIONS = 200

REPORT_INTERVAL = 20

# LBM parameters
TAU = 0.6
OMEGA = 1.0 / TAU

RHO_INITIAL = 1.0

# Uniform body force in the x direction.
# This is a lattice-unit force density.
FORCE_X = 1.0e-6

FORCE_Y = 0.0
FORCE_Z = 0.0

# Numerical precision
DTYPE = jnp.float32

jax.config.update("jax_enable_x64", False)


# ============================================================
# 2. D3Q19 LATTICE DEFINITION
# ============================================================

# Direction vectors
C = np.array([
    [ 0,  0,  0],

    [ 1,  0,  0],
    [-1,  0,  0],

    [ 0,  1,  0],
    [ 0, -1,  0],

    [ 0,  0,  1],
    [ 0,  0, -1],

    [ 1,  1,  0],
    [-1, -1,  0],

    [ 1, -1,  0],
    [-1,  1,  0],

    [ 1,  0,  1],
    [-1,  0, -1],

    [ 1,  0, -1],
    [-1,  0,  1],

    [ 0,  1,  1],
    [ 0, -1, -1],

    [ 0,  1, -1],
    [ 0, -1,  1]
], dtype=np.int32)


# D3Q19 weights
W = np.array([
    1.0 / 3.0,

    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,

    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0
], dtype=np.float32)


# Opposite-direction indices
OPP = np.array([
    0,
    2, 1,
    4, 3,
    6, 5,
    8, 7,
    10, 9,
    12, 11,
    14, 13,
    16, 15,
    18, 17
], dtype=np.int32)


# Converting constants to JAX arrays
C_JAX = jnp.asarray(C, dtype=jnp.float32)
W_JAX = jnp.asarray(W, dtype=jnp.float32)
OPP_JAX = jnp.asarray(OPP, dtype=jnp.int32)

FORCE = jnp.array(
    [FORCE_X, FORCE_Y, FORCE_Z],
    dtype=jnp.float32
)


# ============================================================
# 3. LOADING GEOMETRY
# ============================================================

def load_geometry():

    print("\nReading RAW geometry using memory mapping.")

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Geometry file not found: {RAW_FILE}"
        )

    # Avoiding loading the entire 512^3 file into RAM.
    raw = np.memmap(
        RAW_FILE,
        dtype=np.int8,
        mode="r",
        shape=(RAW_SIZE, RAW_SIZE, RAW_SIZE)
    )

    # Extracting central subdomain.
    start = (RAW_SIZE - GRID_SIZE) // 2
    end = start + GRID_SIZE

    geometry = np.array(
        raw[start:end, start:end, start:end],
        dtype=np.int8
    )

    del raw

    # Convention: 1 = fluid, 0 = solid.
    fluid = geometry == 1

    if not np.any(fluid):
        raise ValueError(
            "Selected subdomain contains no fluid voxels."
        )

    if not np.any(~fluid):
        raise ValueError(
            "Selected subdomain contains no solid voxels."
        )

    porosity = np.mean(fluid)

    print(f"Selected geometry: {GRID_SIZE}^3")
    print(f"Fluid porosity: {porosity:.5f}")
    print(f"Fluid voxels: {np.sum(fluid):,}")
    print(f"Solid voxels: {np.sum(~fluid):,}")

    return jnp.asarray(fluid, dtype=jnp.bool_)


# ============================================================
# 4. INITIALIZING DISTRIBUTION FUNCTIONS
# ============================================================

def initialize_lbm(fluid_mask):

    print("\nInitializing D3Q19 distributions.")

    rho = jnp.where(
        fluid_mask,
        RHO_INITIAL,
        0.0
    ).astype(DTYPE)

    # Initially stationary fluid.
    ux = jnp.zeros_like(rho)
    uy = jnp.zeros_like(rho)
    uz = jnp.zeros_like(rho)

    u_sq = ux**2 + uy**2 + uz**2

    cu = (
        C_JAX[:, 0, None, None, None] * ux[None, :, :, :]
        + C_JAX[:, 1, None, None, None] * uy[None, :, :, :]
        + C_JAX[:, 2, None, None, None] * uz[None, :, :, :]
    )

    feq = (
        W_JAX[:, None, None, None]
        * rho[None, :, :, :]
        * (
            1.0
            + 3.0 * cu
            + 4.5 * cu**2
            - 1.5 * u_sq[None, :, :, :]
        )
    )

    return feq.astype(DTYPE)


# ============================================================
# 5. MACROSCOPIC VARIABLES
# ============================================================

def calculate_macroscopic(f, fluid_mask):

    rho = jnp.sum(f, axis=0)

    rho_safe = jnp.where(
        fluid_mask,
        rho,
        1.0
    )

    # Momentum density
    momentum_x = jnp.sum(
        f * C_JAX[:, 0, None, None, None],
        axis=0
    )

    momentum_y = jnp.sum(
        f * C_JAX[:, 1, None, None, None],
        axis=0
    )

    momentum_z = jnp.sum(
        f * C_JAX[:, 2, None, None, None],
        axis=0
    )

    # Guo forcing correction:
    # rho*u = momentum + 0.5*F
    ux = (momentum_x + 0.5 * FORCE[0]) / rho_safe
    uy = (momentum_y + 0.5 * FORCE[1]) / rho_safe
    uz = (momentum_z + 0.5 * FORCE[2]) / rho_safe

    # Zero velocity in solid voxels.
    ux = jnp.where(fluid_mask, ux, 0.0)
    uy = jnp.where(fluid_mask, uy, 0.0)
    uz = jnp.where(fluid_mask, uz, 0.0)

    rho = jnp.where(fluid_mask, rho, 0.0)

    return rho, ux, uy, uz


# ============================================================
# 6. EQUILIBRIUM DISTRIBUTION
# ============================================================

def equilibrium_distribution(rho, ux, uy, uz):

    u_sq = ux**2 + uy**2 + uz**2

    cu = (
        C_JAX[:, 0, None, None, None] * ux[None, :, :, :]
        + C_JAX[:, 1, None, None, None] * uy[None, :, :, :]
        + C_JAX[:, 2, None, None, None] * uz[None, :, :, :]
    )

    feq = (
        W_JAX[:, None, None, None]
        * rho[None, :, :, :]
        * (
            1.0
            + 3.0 * cu
            + 4.5 * cu**2
            - 1.5 * u_sq[None, :, :, :]
        )
    )

    return feq


# ============================================================
# 7. GUO FORCE TERM
# ============================================================

def guo_force_term(ux, uy, uz):

    u_dot_c = (
        C_JAX[:, 0, None, None, None] * ux[None, :, :, :]
        + C_JAX[:, 1, None, None, None] * uy[None, :, :, :]
        + C_JAX[:, 2, None, None, None] * uz[None, :, :, :]
    )

    c_dot_force = (
        C_JAX[:, 0] * FORCE[0]
        + C_JAX[:, 1] * FORCE[1]
        + C_JAX[:, 2] * FORCE[2]
    )

    u_dot_force = (
        ux * FORCE[0]
        + uy * FORCE[1]
        + uz * FORCE[2]
    )

    term = (
        3.0 * (
            C_JAX[:, 0, None, None, None] * FORCE[0]
            + C_JAX[:, 1, None, None, None] * FORCE[1]
            + C_JAX[:, 2, None, None, None] * FORCE[2]
            - u_dot_force[None, :, :, :]
        )
        + 9.0 * u_dot_c * c_dot_force[:, None, None, None]
    )

    source = (
        W_JAX[:, None, None, None]
        * (1.0 - 0.5 * OMEGA)
        * term
    )

    return source


# ============================================================
# 8. COLLISION + STREAMING + BOUNCE-BACK
# ============================================================

@jax.jit
def lbm_step(f, fluid_mask):

    # A. Macroscopic variables
    rho, ux, uy, uz = calculate_macroscopic(
        f,
        fluid_mask
    )

    # B. Equilibrium
    feq = equilibrium_distribution(
        rho,
        ux,
        uy,
        uz
    )

    # C. Guo forcing
    force_term = guo_force_term(
        ux,
        uy,
        uz
    )

    # D. BGK collision
    f_post = (
        f
        - OMEGA * (f - feq)
        + force_term
    )

    # Removing distributions inside solid nodes.
    f_post = jnp.where(
        fluid_mask[None, :, :, :],
        f_post,
        0.0
    )

    # E. Streaming and on-site bounce-back.
    solid_mask = ~fluid_mask

    streamed_fields = []

    for i in range(19):

        cx = int(C[i, 0])
        cy = int(C[i, 1])
        cz = int(C[i, 2])

        # Periodic streaming.
        streamed = jnp.roll(
            f_post[i],
            shift=(cx, cy, cz),
            axis=(0, 1, 2)
        )

        # Identifying solid upstream neighbour.
        upstream_solid = jnp.roll(
            solid_mask,
            shift=(cx, cy, cz),
            axis=(0, 1, 2)
        )

        # On-site bounce-back:
        # Reversing the population when its upstream
        # neighbour is a solid voxel.
        reflected = f_post[OPP[i]]

        population = jnp.where(
            upstream_solid,
            reflected,
            streamed
        )

        # No fluid distributions inside solid voxels.
        population = jnp.where(
            fluid_mask,
            population,
            0.0
        )

        streamed_fields.append(population)

    f_new = jnp.stack(
        streamed_fields,
        axis=0
    )

    return f_new


# ============================================================
# 9. DIAGNOSTICS
# ============================================================

@jax.jit
def calculate_diagnostics(f, fluid_mask):

    rho, ux, uy, uz = calculate_macroscopic(
        f,
        fluid_mask
    )

    speed = jnp.sqrt(
        ux**2 + uy**2 + uz**2
    )

    fluid_count = jnp.sum(
        fluid_mask.astype(jnp.float32)
    )

    mean_ux = (
        jnp.sum(ux) / fluid_count
    )

    mean_speed = (
        jnp.sum(speed) / fluid_count
    )

    max_speed = jnp.max(
        jnp.where(fluid_mask, speed, 0.0)
    )

    total_mass = jnp.sum(rho)

    return (
        mean_ux,
        mean_speed,
        max_speed,
        total_mass
    )


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("============================================")
    print(" D3Q19 JAX-LBM VERIFICATION")
    print("============================================")

    print(f"JAX version: {jax.__version__}")
    print(f"Execution backend: {jax.default_backend()}")
    print(f"Execution devices: {jax.devices()}")

    # Load geometry
    fluid_mask = load_geometry()

    # Initialize distributions
    f = initialize_lbm(fluid_mask)

    # Compile and execute first step
    print("\nCompiling and executing first LBM step.")

    t0 = time.time()

    f = lbm_step(
        f,
        fluid_mask
    )

    f.block_until_ready()

    print(
        f"First step completed in {time.time() - t0:.2f} seconds."
    )

    # Initial mass
    initial_mass = float(
        jnp.sum(f).block_until_ready()
    )

    print(f"Initial fluid mass: {initial_mass:.6f}")

    # Main loop
    print("\nStarting simulation...")

    simulation_start = time.time()

    for iteration in range(1, MAX_ITERATIONS + 1):

        f = lbm_step(
            f,
            fluid_mask
        )

        if (
            iteration % REPORT_INTERVAL == 0
            or iteration == 1
        ):

            (
                mean_ux,
                mean_speed,
                max_speed,
                total_mass
            ) = calculate_diagnostics(
                f,
                fluid_mask
            )

            # Synchronizing before reporting.
            (
                mean_ux,
                mean_speed,
                max_speed,
                total_mass
            ) = jax.device_get((
                mean_ux,
                mean_speed,
                max_speed,
                total_mass
            ))

            mean_ux = float(mean_ux)
            mean_speed = float(mean_speed)
            max_speed = float(max_speed)
            total_mass = float(total_mass)

            mass_error = abs(
                total_mass - initial_mass
            ) / max(abs(initial_mass), 1e-30)

            print(
                f"Iteration {iteration:04d}/{MAX_ITERATIONS} | "
                f"Mean Ux: {mean_ux:.6e} | "
                f"Mean speed: {mean_speed:.6e} | "
                f"Max speed: {max_speed:.6e} | "
                f"Mass error: {mass_error:.3e}"
            )

            if not np.all(
                np.isfinite([
                    mean_ux,
                    mean_speed,
                    max_speed,
                    total_mass
                ])
            ):
                raise FloatingPointError(
                    "Non-finite numerical values detected."
                )

    # Final synchronization
    f.block_until_ready()

    elapsed = time.time() - simulation_start

    print("\n============================================")
    print(" SIMULATION COMPLETED")
    print("============================================")

    print(f"Iterations: {MAX_ITERATIONS}")
    print(f"Grid: {GRID_SIZE} x {GRID_SIZE} x {GRID_SIZE}")
    print(f"Tau: {TAU}")
    print(f"Omega: {OMEGA}")
    print(f"Body force X: {FORCE_X}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    print("\nVerification complete.")
    print(
        "Note: Numerical stability and mass conservation "
        "must still be assessed before scientific use."
    )


if __name__ == "__main__":
    main()
