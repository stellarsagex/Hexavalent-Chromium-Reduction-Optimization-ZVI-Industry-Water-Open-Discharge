# Extending LBM verification


# ============================================================
# Step_4.2_LBM_Verification.py
# D3Q19 EXTENDED STABILITY TRACKING + VELOCITY FIELD EXPORT
# ============================================================

import os
import time
import numpy as np
import jax
import jax.numpy as jnp

# --- 1. CONFIGURATION ---

RAW_FILE = "Step_3.3_Geometry_512.raw"
RAW_SIZE = 512
GRID_SIZE = 64

MAX_ITERATIONS = 2000
REPORT_INTERVAL = 50

# Convergence threshold: relative change in mean Ux
CONVERGENCE_THRESHOLD = 5.0e-4

# LBM parameters
TAU = 0.6
OMEGA = 1.0 / TAU

RHO_INITIAL = 1.0

FORCE_X = 1.0e-6
FORCE_Y = 0.0
FORCE_Z = 0.0

DTYPE = jnp.float32

jax.config.update("jax_enable_x64", False)


# --- 2. D3Q19 LATTICE CONSTANTS ---

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

W = np.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18,
    1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36
], dtype=np.float32)

OPP = np.array([
    0, 2, 1, 4, 3, 6, 5,
    8, 7, 10, 9, 12, 11,
    14, 13, 16, 15, 18, 17
], dtype=np.int32)

C_JAX = jnp.asarray(C, dtype=jnp.float32)
W_JAX = jnp.asarray(W, dtype=jnp.float32)
FORCE = jnp.array(
    [FORCE_X, FORCE_Y, FORCE_Z],
    dtype=jnp.float32
)


# --- 3. GEOMETRY LOADER ---

def load_geometry():

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Geometry file not found: {RAW_FILE}"
        )

    print("\nLoading 512^3 geometry...")

    raw = np.memmap(
        RAW_FILE,
        dtype=np.int8,
        mode="r",
        shape=(RAW_SIZE, RAW_SIZE, RAW_SIZE)
    )

    start = (RAW_SIZE - GRID_SIZE) // 2
    end = start + GRID_SIZE

    geometry = np.array(
        raw[start:end, start:end, start:end],
        dtype=np.int8
    )

    del raw

    fluid = geometry == 1

    print(f"Subdomain dimensions: {geometry.shape}")
    print(f"Fluid voxels: {np.sum(fluid):,}")
    print(f"Solid voxels: {np.sum(~fluid):,}")

    return jnp.asarray(fluid, dtype=jnp.bool_)


# --- 4. LBM INITIALIZATION ---

def initialize_lbm(fluid_mask):

    rho = jnp.where(
        fluid_mask,
        RHO_INITIAL,
        0.0
    ).astype(DTYPE)

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


# --- 5. MACROSCOPIC VARIABLE CALCULATION ---

def calculate_macroscopic(f, fluid_mask):

    rho = jnp.sum(f, axis=0)

    rho_safe = jnp.where(
        fluid_mask,
        rho,
        1.0
    )

    mx = jnp.sum(
        f * C_JAX[:, 0, None, None, None],
        axis=0
    )

    my = jnp.sum(
        f * C_JAX[:, 1, None, None, None],
        axis=0
    )

    mz = jnp.sum(
        f * C_JAX[:, 2, None, None, None],
        axis=0
    )

    ux = (mx + 0.5 * FORCE[0]) / rho_safe
    uy = (my + 0.5 * FORCE[1]) / rho_safe
    uz = (mz + 0.5 * FORCE[2]) / rho_safe

    return (
        jnp.where(fluid_mask, rho, 0.0),
        jnp.where(fluid_mask, ux, 0.0),
        jnp.where(fluid_mask, uy, 0.0),
        jnp.where(fluid_mask, uz, 0.0)
    )


# --- 6. D3Q19 BGK COLLISION + STREAMING ---

@jax.jit
def lbm_step(f, fluid_mask):

    rho, ux, uy, uz = calculate_macroscopic(
        f,
        fluid_mask
    )

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

    # Guo forcing formulation
    u_dot_force = (
        ux * FORCE[0]
        + uy * FORCE[1]
        + uz * FORCE[2]
    )

    c_dot_force = (
        C_JAX[:, 0] * FORCE[0]
        + C_JAX[:, 1] * FORCE[1]
        + C_JAX[:, 2] * FORCE[2]
    )

    term = (
        3.0 * (
            C_JAX[:, 0, None, None, None] * FORCE[0]
            + C_JAX[:, 1, None, None, None] * FORCE[1]
            + C_JAX[:, 2, None, None, None] * FORCE[2]
            - u_dot_force[None, :, :, :]
        )
        + 9.0 * cu * c_dot_force[:, None, None, None]
    )

    force_term = (
        W_JAX[:, None, None, None]
        * (1.0 - 0.5 * OMEGA)
        * term
    )

    f_post = jnp.where(
        fluid_mask[None, :, :, :],
        f - OMEGA * (f - feq) + force_term,
        0.0
    )

    # Solid-wall halfway bounce-back
    solid_mask = ~fluid_mask

    streamed_fields = []

    for i in range(19):

        cx_val = int(C[i, 0])
        cy_val = int(C[i, 1])
        cz_val = int(C[i, 2])

        streamed = jnp.roll(
            f_post[i],
            shift=(cx_val, cy_val, cz_val),
            axis=(0, 1, 2)
        )

        upstream_solid = jnp.roll(
            solid_mask,
            shift=(cx_val, cy_val, cz_val),
            axis=(0, 1, 2)
        )

        reflected = f_post[OPP[i]]

        population = jnp.where(
            upstream_solid,
            reflected,
            streamed
        )

        streamed_fields.append(
            jnp.where(fluid_mask, population, 0.0)
        )

    return jnp.stack(streamed_fields, axis=0)


# --- 7. HYDRODYNAMIC DIAGNOSTICS ---

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

    mean_ux = jnp.sum(ux) / fluid_count
    mean_speed = jnp.sum(speed) / fluid_count

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


# --- 8. MAIN SIMULATION + VELOCITY EXPORT ---

def main():

    print("============================================")
    print(" D3Q19 EXTENDED STABILITY TRACKING ENGINE")
    print("============================================")

    # Load geometry
    fluid_mask = load_geometry()

    # Initialize distribution functions
    f = initialize_lbm(fluid_mask)

    # JAX compilation warm-up
    f = lbm_step(f, fluid_mask).block_until_ready()

    initial_mass = float(jnp.sum(f))

    print(
        f"\nSimulation Started. TARGET PATH: "
        f"{MAX_ITERATIONS} Steps."
    )

    sim_start = time.time()

    previous_mean_ux = None
    converged = False
    convergence_iteration = None

    # Main LBM iteration loop
    for iteration in range(1, MAX_ITERATIONS + 1):

        f = lbm_step(f, fluid_mask)

        if (
            iteration % REPORT_INTERVAL == 0
            or iteration == 1
        ):

            (
                m_ux,
                m_spd,
                mx_spd,
                t_mass
            ) = calculate_diagnostics(
                f,
                fluid_mask
            )

            m_ux, m_spd, mx_spd, t_mass = jax.device_get(
                (m_ux, m_spd, mx_spd, t_mass)
            )

            mass_error = (
                abs(float(t_mass) - initial_mass)
                / max(abs(initial_mass), 1e-30)
            )

            if (
                previous_mean_ux is not None
                and previous_mean_ux != 0
            ):

                rel_change_ux = (
                    abs(float(m_ux) - previous_mean_ux)
                    / abs(previous_mean_ux)
                )

                rel_change_str = f"{rel_change_ux:.4e}"

            else:

                rel_change_ux = float("inf")
                rel_change_str = "N/A"

            print(
                f"Step {iteration:04d}/{MAX_ITERATIONS} | "
                f"Mean Ux: {float(m_ux):.5e} | "
                f"Rel Change: {rel_change_str} | "
                f"Max Spd: {float(mx_spd):.5e} | "
                f"Mass Err: {mass_error:.2e}"
            )

            # Convergence check
            if rel_change_ux < CONVERGENCE_THRESHOLD:

                converged = True
                convergence_iteration = iteration

                print(
                    f"\nSteady-State Convergence Achieved "
                    f"at iteration {iteration}!"
                )

                print(
                    f"Relative variation: {rel_change_ux:.4e}"
                )

                break

            previous_mean_ux = float(m_ux)

    # Synchronize final LBM state
    f.block_until_ready()

    # --- 9. EXTRACT CONVERGED VELOCITY FIELDS ---

    print("\nExtracting final macroscopic velocity fields...")

    rho, ux, uy, uz = calculate_macroscopic(
        f,
        fluid_mask
    )

    # Transfer JAX arrays to CPU memory
    rho_np, ux_np, uy_np, uz_np, fluid_np = jax.device_get(
        (rho, ux, uy, uz, fluid_mask)
    )

    # Final velocity diagnostics
    speed_np = np.sqrt(
        ux_np**2 + uy_np**2 + uz_np**2
    )

    fluid_count = np.sum(fluid_np)

    mean_ux_final = float(
        np.mean(ux_np[fluid_np])
    )

    mean_speed_final = float(
        np.mean(speed_np[fluid_np])
    )

    max_speed_final = float(
        np.max(speed_np[fluid_np])
    )

    print("\nFINAL VELOCITY FIELD REPORT")
    print("--------------------------------------------")
    print(f"Fluid voxels          : {fluid_count:,}")
    print(f"Mean Ux               : {mean_ux_final:.6e}")
    print(f"Mean speed            : {mean_speed_final:.6e}")
    print(f"Maximum speed         : {max_speed_final:.6e}")
    print(f"Convergence achieved  : {converged}")
    print(f"Final iteration       : {iteration}")

    # --- 10. SAVE VELOCITY FIELDS ---

    output_file = "Step_4.2_Velocity_Field.npz"

    np.savez_compressed(
        output_file,

        ux=np.asarray(ux_np, dtype=np.float32),
        uy=np.asarray(uy_np, dtype=np.float32),
        uz=np.asarray(uz_np, dtype=np.float32),

        rho=np.asarray(rho_np, dtype=np.float32),

        fluid_mask=np.asarray(fluid_np, dtype=bool),

        convergence_achieved=converged,
        convergence_iteration=iteration,

        tau=TAU,
        omega=OMEGA,

        force_x=FORCE_X,
        force_y=FORCE_Y,
        force_z=FORCE_Z,

        grid_size=GRID_SIZE,
        report_interval=REPORT_INTERVAL,
        convergence_threshold=CONVERGENCE_THRESHOLD
    )

    print("\nVELOCITY FIELD EXPORT")
    print("--------------------------------------------")
    print(f"File saved             : {output_file}")
    print(f"Array dimensions       : {ux_np.shape}")
    print(f"File size              : {os.path.getsize(output_file) / 1024:.2f} KB")

    print("\nExecution finished in "
          f"{time.time() - sim_start:.2f} seconds.")

    print("============================================")


if __name__ == "__main__":
    main()
