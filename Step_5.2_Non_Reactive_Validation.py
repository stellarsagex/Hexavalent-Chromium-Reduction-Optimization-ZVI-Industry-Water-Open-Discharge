
# ============================================================
# File: Step_5.2_Tracer_Breakthrough.py
#
# Conservative tracer transport using the velocity field
# exported by Step_4.2 periodic D3Q19 LBM.
#
# PHYSICAL SCALING:
# - Physical voxel spacing: 0.6349 nm
# - LBM relaxation time: read from saved velocity file
# - Water kinematic viscosity: 1.0e-6 m2/s (approximately 25 C)
# - Chloride molecular diffusivity: 2.0e-9 m2/s
# - Physical time step calculated from LBM viscosity scaling
# - Lattice diffusivity calculated from physical diffusivity
#
# TRANSPORT:
# - Does NOT rerun LBM.
# - Uses saved velocity field.
# - X: open solute inlet/outlet.
# - Y,Z: periodic.
# - Solid surfaces: no solute flux.
# - Conservative finite-volume advection-diffusion.
#
# OUTPUTS:
# - Step_5.2_Tracer_BTC.csv
# - Step_5.2_Tracer_BTC.png
# - Step_5.2_Tracer_Final_State.npz
# ============================================================

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

RAW_FILE = "Step_3.3_Geometry_512.raw"
VELOCITY_FILE = "Step_4.2_Velocity_Field.npz"

RAW_SIZE = 512
GRID_SIZE = 64

# ------------------------------------------------------------
# PHYSICAL GEOMETRY
# ------------------------------------------------------------

DX_PHYS = 0.6349e-9       # metres per voxel (0.6349 nm)

# ------------------------------------------------------------
# PHYSICAL FLUID PROPERTIES
# ------------------------------------------------------------

NU_PHYS = 1.0e-6          # m2/s, water at approximately 25 C

D_CL_PHYS = 2.0e-9        # m2/s, assumed chloride diffusivity

# ------------------------------------------------------------
# TRACER
# ------------------------------------------------------------

C_INLET = 2.0             # mmol/L

# ------------------------------------------------------------
# TRANSPORT
# ------------------------------------------------------------

DT_LU = 1.0               # lattice time step

TRANSPORT_STEPS = 61_400_000
CHUNK_STEPS = 1_000

# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

OUTPUT_CSV = "Step_5.2_Tracer_BTC.csv"
OUTPUT_PLOT = "Step_5.2_Tracer_BTC.png"
OUTPUT_STATE = "Step_5.2_Tracer_Final_State.npz"

# Use double precision for improved accumulation accuracy.
jax.config.update("jax_enable_x64", True)


# ============================================================
# 2. LOAD GEOMETRY
# ============================================================

print("=" * 60)
print("STEP 5.2: PHYSICALLY SCALED CONSERVATIVE TRACER TRANSPORT")
print("=" * 60)

print("\nLoading geometry...")

if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(f"Geometry file not found: {RAW_FILE}")

raw = np.memmap(
    RAW_FILE,
    dtype=np.int8,
    mode="r",
    shape=(RAW_SIZE, RAW_SIZE, RAW_SIZE)
)

start = (RAW_SIZE - GRID_SIZE) // 2

geometry = np.array(
    raw[
        start:start + GRID_SIZE,
        start:start + GRID_SIZE,
        start:start + GRID_SIZE
    ],
    dtype=np.int8
)

del raw

fluid_np = geometry == 1
fluid_mask = jnp.asarray(fluid_np)

NX, NY, NZ = fluid_np.shape
porosity = np.mean(fluid_np)

print(f"Grid dimensions: {NX} x {NY} x {NZ}")
print(f"Fluid voxels: {np.sum(fluid_np):,}")
print(f"Porosity: {porosity:.6f}")


# ============================================================
# 3. LOAD EXISTING VELOCITY FIELD
# ============================================================

print("\nLoading saved LBM velocity field...")

if not os.path.exists(VELOCITY_FILE):
    raise FileNotFoundError(
        f"Velocity file not found: {VELOCITY_FILE}"
    )

data = np.load(VELOCITY_FILE)

print("Available NPZ arrays:", data.files)


def find_array(data, candidates, description):

    for name in candidates:
        if name in data.files:
            arr = np.asarray(data[name])
            print(f"Loaded {description}: {name}")
            return arr

    raise KeyError(
        f"Could not find {description}. "
        f"Available arrays: {data.files}"
    )


ux_np = find_array(
    data,
    ["ux", "velocity_x", "u_x", "vx"],
    "X velocity"
)

uy_np = find_array(
    data,
    ["uy", "velocity_y", "u_y", "vy"],
    "Y velocity"
)

uz_np = find_array(
    data,
    ["uz", "velocity_z", "u_z", "vz"],
    "Z velocity"
)

# Read the relaxation time from the original LBM output.
if "tau" in data.files:
    TAU = float(np.asarray(data["tau"]).item())
else:
    TAU = 0.6
    print("WARNING: tau not found in NPZ; using TAU = 0.6.")

if not (TAU > 0.5):
    raise ValueError("LBM relaxation time must be greater than 0.5.")

if not (
    ux_np.shape == fluid_np.shape
    and uy_np.shape == fluid_np.shape
    and uz_np.shape == fluid_np.shape
):
    raise ValueError(
        "Velocity field dimensions do not match geometry."
    )

# Verify saved fluid mask, if available.
for key in ["fluid_mask", "fluid", "mask"]:

    if key in data.files:

        saved_mask = np.asarray(data[key]).astype(bool)

        if saved_mask.shape == fluid_np.shape:

            if not np.array_equal(saved_mask, fluid_np):
                raise ValueError(
                    "Saved velocity mask does not match geometry."
                )

        break

# Remove velocity values inside solid voxels.
ux_np = np.where(fluid_np, ux_np, 0.0)
uy_np = np.where(fluid_np, uy_np, 0.0)
uz_np = np.where(fluid_np, uz_np, 0.0)

ux = jnp.asarray(ux_np, dtype=jnp.float64)
uy = jnp.asarray(uy_np, dtype=jnp.float64)
uz = jnp.asarray(uz_np, dtype=jnp.float64)

data.close()

print("\nVelocity field statistics:")

print(f"Mean Ux (LU): {np.mean(ux_np[fluid_np]):.8e}")
print(f"Mean Uy (LU): {np.mean(uy_np[fluid_np]):.8e}")
print(f"Mean Uz (LU): {np.mean(uz_np[fluid_np]):.8e}")

speed_np = np.sqrt(ux_np**2 + uy_np**2 + uz_np**2)

print(f"Maximum speed (LU): {np.max(speed_np):.8e}")


# ============================================================
# 4. PHYSICAL-TO-LATTICE DIMENSIONALISATION
# ============================================================

print("\n" + "=" * 60)
print("PHYSICAL SCALING")
print("=" * 60)

# D3Q19 BGK kinematic viscosity:
#
# nu_LU = cs^2 * (tau - 0.5)
#
# where cs^2 = 1/3.

CS2 = 1.0 / 3.0

NU_LU = CS2 * (TAU - 0.5)

# Physical time step:
#
# nu_phys = nu_LU * DX_phys^2 / DT_phys
#
# Therefore:
#
# DT_phys = nu_LU * DX_phys^2 / nu_phys

DT_PHYS = NU_LU * DX_PHYS**2 / NU_PHYS

# Physical-to-lattice diffusivity conversion:
#
# D_LU = D_phys * DT_phys / DX_phys^2

D_CL_LU = D_CL_PHYS * DT_PHYS / DX_PHYS**2

# Equivalent simplified relationship:
# D_CL_LU = D_CL_PHYS * NU_LU / NU_PHYS

# Physical velocity conversion:
#
# u_phys = u_LU * DX_phys / DT_phys

VELOCITY_SCALE = DX_PHYS / DT_PHYS

mean_ux_lu = float(np.mean(ux_np[fluid_np]))
mean_speed_lu = float(np.mean(speed_np[fluid_np]))
max_speed_lu = float(np.max(speed_np))

mean_ux_phys = mean_ux_lu * VELOCITY_SCALE
mean_speed_phys = mean_speed_lu * VELOCITY_SCALE
max_speed_phys = max_speed_lu * VELOCITY_SCALE

REV_LENGTH_PHYS = NX * DX_PHYS

TOTAL_PHYSICAL_TIME = TRANSPORT_STEPS * DT_PHYS

# Péclet number using mean X velocity and full REV length.
PE = (
    abs(mean_ux_lu) * NX / D_CL_LU
)

print(f"Physical voxel spacing: {DX_PHYS:.6e} m")
print(f"LBM relaxation time: {TAU:.6f}")
print(f"Lattice kinematic viscosity: {NU_LU:.8e}")
print(f"Physical kinematic viscosity: {NU_PHYS:.8e} m2/s")

print(f"\nPhysical time step: {DT_PHYS:.8e} s")
print(f"Lattice time step: {DT_LU:.6f}")

print(f"\nPhysical chloride diffusivity: {D_CL_PHYS:.8e} m2/s")
print(f"Lattice chloride diffusivity: {D_CL_LU:.8e}")

print(f"\nMean X velocity: {mean_ux_phys:.8e} m/s")
print(f"Mean speed: {mean_speed_phys:.8e} m/s")
print(f"Maximum speed: {max_speed_phys:.8e} m/s")

print(f"\nREV physical length: {REV_LENGTH_PHYS:.8e} m")
print(f"Total simulated physical time: {TOTAL_PHYSICAL_TIME:.8e} s")

print(f"\nEstimated Péclet number: {PE:.6f}")


# ============================================================
# 5. HYDRODYNAMIC / TRANSPORT DIAGNOSTICS
# ============================================================

qin_signed = float(
    np.sum(ux_np[0, :, :] * fluid_np[0, :, :])
)

qout_signed = float(
    np.sum(ux_np[-1, :, :] * fluid_np[-1, :, :])
)

qin_positive = float(
    np.sum(
        np.maximum(ux_np[0, :, :], 0.0)
        * fluid_np[0, :, :]
    )
)

qout_positive = float(
    np.sum(
        np.maximum(ux_np[-1, :, :], 0.0)
        * fluid_np[-1, :, :]
    )
)

print("\nBoundary flow diagnostics (LU):")

print(f"Net inlet flow: {qin_signed:.8e}")
print(f"Net outlet flow: {qout_signed:.8e}")

print(f"Positive inlet flow: {qin_positive:.8e}")
print(f"Positive outlet flow: {qout_positive:.8e}")

if abs(qin_signed) > 1.0e-20:
    print(
        "Net outlet/inlet flow ratio:",
        abs(qout_signed / qin_signed)
    )

if mean_ux_lu <= 0:
    raise ValueError(
        "Mean X velocity is not positive. "
        "Check the inlet-to-outlet direction."
    )

# Explicit stability diagnostics.
cfl_diffusion = 6.0 * D_CL_LU * DT_LU

cfl_advection = DT_LU * (
    np.max(np.abs(ux_np))
    + np.max(np.abs(uy_np))
    + np.max(np.abs(uz_np))
)

combined_stability = cfl_diffusion + cfl_advection

print("\nStability diagnostics:")
print(f"Diffusion indicator: {cfl_diffusion:.8e}")
print(f"Advection indicator: {cfl_advection:.8e}")
print(f"Combined indicator: {combined_stability:.8e}")

if combined_stability > 1.0:
    raise ValueError(
        "Explicit transport time step may be unstable. "
        "Reduce DT_LU or reconsider the numerical scheme."
    )

if PE < 0.1:
    print("Transport is diffusion-dominated for these parameters.")
elif PE > 100:
    print("Transport is advection-dominated for these parameters.")
else:
    print("Transport is in a mixed advection-diffusion regime.")


# ============================================================
# 6. CONSERVATIVE ADVECTION-DIFFUSION SOLVER
# ============================================================

D = jnp.float64(D_CL_LU)
C0 = jnp.float64(C_INLET)
dt = jnp.float64(DT_LU)


@jax.jit
def transport_step(C):

    # --------------------------------------------------------
    # X-DIRECTION FLUXES
    # --------------------------------------------------------

    ux_face = 0.5 * (ux[:-1, :, :] + ux[1:, :, :])

    valid_x = (
        fluid_mask[:-1, :, :]
        & fluid_mask[1:, :, :]
    )

    C_up_x = jnp.where(
        ux_face >= 0.0,
        C[:-1, :, :],
        C[1:, :, :]
    )

    adv_x = ux_face * C_up_x

    diff_x = -D * (
        C[1:, :, :] - C[:-1, :, :]
    )

    internal_flux_x = jnp.where(
        valid_x,
        adv_x + diff_x,
        0.0
    )

    Fx = jnp.zeros(
        (NX + 1, NY, NZ),
        dtype=jnp.float64
    )

    Fx = Fx.at[1:NX, :, :].set(internal_flux_x)

    # INLET
    u_in = ux[0, :, :]

    inlet_adv = u_in * jnp.where(
        u_in >= 0.0,
        C0,
        C[0, :, :]
    )

    inlet_diff = D * (
        C0 - C[0, :, :]
    )

    inlet_flux = jnp.where(
        fluid_mask[0, :, :],
        inlet_adv + inlet_diff,
        0.0
    )

    Fx = Fx.at[0, :, :].set(inlet_flux)

    # OUTLET
    u_out = ux[-1, :, :]

    outlet_adv = u_out * jnp.where(
        u_out >= 0.0,
        C[-1, :, :],
        0.0
    )

    outlet_flux = jnp.where(
        fluid_mask[-1, :, :],
        outlet_adv,
        0.0
    )

    Fx = Fx.at[NX, :, :].set(outlet_flux)

    # --------------------------------------------------------
    # Y-DIRECTION FLUXES (PERIODIC)
    # --------------------------------------------------------

    C_yp = jnp.roll(C, -1, axis=1)
    uy_yp = jnp.roll(uy, -1, axis=1)

    mask_yp = jnp.roll(
        fluid_mask,
        -1,
        axis=1
    )

    uy_face = 0.5 * (uy + uy_yp)

    valid_y = fluid_mask & mask_yp

    C_up_y = jnp.where(
        uy_face >= 0.0,
        C,
        C_yp
    )

    Fy = jnp.where(
        valid_y,
        uy_face * C_up_y - D * (C_yp - C),
        0.0
    )

    # --------------------------------------------------------
    # Z-DIRECTION FLUXES (PERIODIC)
    # --------------------------------------------------------

    C_zp = jnp.roll(C, -1, axis=2)
    uz_zp = jnp.roll(uz, -1, axis=2)

    mask_zp = jnp.roll(
        fluid_mask,
        -1,
        axis=2
    )

    uz_face = 0.5 * (uz + uz_zp)

    valid_z = fluid_mask & mask_zp

    C_up_z = jnp.where(
        uz_face >= 0.0,
        C,
        C_zp
    )

    Fz = jnp.where(
        valid_z,
        uz_face * C_up_z - D * (C_zp - C),
        0.0
    )

    # --------------------------------------------------------
    # CONSERVATIVE FLUX DIVERGENCE
    # --------------------------------------------------------

    div_x = Fx[1:, :, :] - Fx[:-1, :, :]

    div_y = Fy - jnp.roll(Fy, 1, axis=1)

    div_z = Fz - jnp.roll(Fz, 1, axis=2)

    divergence = div_x + div_y + div_z

    C_new = C - dt * divergence

    C_new = jnp.where(
        fluid_mask,
        C_new,
        0.0
    )

    mass_in = dt * jnp.sum(Fx[0, :, :])
    mass_out = dt * jnp.sum(Fx[-1, :, :])

    return C_new, mass_in, mass_out


# ============================================================
# 7. INITIALIZE CONCENTRATION
# ============================================================

print("\nInitializing tracer concentration...")

C_grid = jnp.zeros(
    (NX, NY, NZ),
    dtype=jnp.float64
)

initial_mass = float(jnp.sum(C_grid))

print(f"Initial tracer mass: {initial_mass:.8e}")


# ============================================================
# 8. COMPILED TRANSPORT CHUNKS
# ============================================================

@jax.jit
def advance_chunk(C):

    def body(_, carry):

        C_current, cumulative_in, cumulative_out = carry

        C_next, mass_in, mass_out = transport_step(
            C_current
        )

        return (
            C_next,
            cumulative_in + mass_in,
            cumulative_out + mass_out
        )

    return lax.fori_loop(
        0,
        CHUNK_STEPS,
        body,
        (
            C,
            jnp.float64(0.0),
            jnp.float64(0.0)
        )
    )


# ============================================================
# 9. RUN TRANSPORT SIMULATION
# ============================================================

print("\nStarting conservative tracer simulation...")

print(f"Total transport steps: {TRANSPORT_STEPS:,}")
print(f"Steps per compiled chunk: {CHUNK_STEPS:,}")

num_chunks = TRANSPORT_STEPS // CHUNK_STEPS

if TRANSPORT_STEPS % CHUNK_STEPS != 0:
    raise ValueError(
        "TRANSPORT_STEPS must be divisible by CHUNK_STEPS."
    )

outlet_positive_velocity = jnp.maximum(
    ux[-1, :, :],
    0.0
)

outlet_weight = (
    outlet_positive_velocity
    * fluid_mask[-1, :, :]
)

outlet_flow = jnp.sum(outlet_weight)

if float(outlet_flow) <= 1.0e-20:
    raise ValueError(
        "No positive outlet flow detected. "
        "Check the velocity field."
    )

btc = []
time_steps = []
physical_times = []
mass_balance_errors = []

cumulative_in = 0.0
cumulative_out = 0.0

t0 = time.time()

for chunk in range(num_chunks):

    C_grid, chunk_in, chunk_out = advance_chunk(C_grid)

    cumulative_in += float(chunk_in)
    cumulative_out += float(chunk_out)

    current_step = (chunk + 1) * CHUNK_STEPS

    current_physical_time = current_step * DT_PHYS

    outlet_concentration = (
        jnp.sum(
            outlet_weight * C_grid[-1, :, :]
        ) / outlet_flow
    )

    normalized_outlet = (
        float(outlet_concentration) / C_INLET
    )

    stored_mass = float(jnp.sum(C_grid))

    mass_error = (
        stored_mass
        - initial_mass
        - cumulative_in
        + cumulative_out
    )

    relative_mass_error = (
        abs(mass_error)
        / max(
            abs(cumulative_in),
            abs(stored_mass),
            1.0e-20
        )
    )

    btc.append(normalized_outlet)
    time_steps.append(current_step)
    physical_times.append(current_physical_time)
    mass_balance_errors.append(mass_error)

    if (
        (chunk + 1) % 10 == 0
        or chunk == 0
        or chunk + 1 == num_chunks
    ):

        print(
            f"Step {current_step:08d} | "
            f"Time = {current_physical_time:.3e} s | "
            f"Cout/Cin = {normalized_outlet:.6f} | "
            f"Mass error = {mass_error:.3e} | "
            f"Relative error = {relative_mass_error:.3e}"
        )


elapsed = time.time() - t0

print("\nTransport simulation completed.")
print(f"Execution time: {elapsed:.2f} seconds")


# ============================================================
# 10. FINAL MASS BALANCE
# ============================================================

final_mass = float(jnp.sum(C_grid))

final_mass_error = (
    final_mass
    - initial_mass
    - cumulative_in
    + cumulative_out
)

relative_mass_error = (
    abs(final_mass_error)
    / max(
        abs(cumulative_in),
        abs(final_mass),
        1.0e-20
    )
)

print("\n" + "=" * 60)
print("FINAL TRANSPORT RESULTS")
print("=" * 60)

print(f"Initial tracer mass: {initial_mass:.8e}")
print(f"Final tracer mass: {final_mass:.8e}")

print(f"Cumulative inlet mass:  {cumulative_in:.8e}")
print(f"Cumulative outlet mass: {cumulative_out:.8e}")

print(f"Mass balance residual: {final_mass_error:.8e}")
print(f"Relative mass error: {relative_mass_error:.8e}")

print(f"Final C_out/C_in: {btc[-1]:.6f}")
print(f"Maximum C_out/C_in: {max(btc):.6f}")

print(
    "Mean concentration in fluid: "
    f"{final_mass / np.sum(fluid_np):.8e}"
)

print(f"Final physical simulation time: {physical_times[-1]:.8e} s")


# ============================================================
# 11. EXPORT BREAKTHROUGH DATA
# ============================================================

btc_array = np.column_stack([
    np.asarray(time_steps),
    np.asarray(physical_times),
    np.asarray(btc),
    np.asarray(mass_balance_errors)
])

np.savetxt(
    OUTPUT_CSV,
    btc_array,
    delimiter=",",
    header=(
        "Time_steps_LU,"
        "Physical_time_seconds,"
        "Normalized_outlet_concentration,"
        "Mass_balance_error"
    ),
    comments=""
)

np.savez_compressed(
    OUTPUT_STATE,

    concentration=np.asarray(C_grid),
    fluid_mask=fluid_np,

    ux=ux_np,
    uy=uy_np,
    uz=uz_np,

    final_mass=final_mass,
    cumulative_inlet_mass=cumulative_in,
    cumulative_outlet_mass=cumulative_out,
    mass_balance_error=final_mass_error,
    relative_mass_error=relative_mass_error,

    dx_physical=DX_PHYS,
    dt_physical=DT_PHYS,
    dt_lattice=DT_LU,

    nu_physical=NU_PHYS,
    nu_lattice=NU_LU,

    chloride_diffusivity_physical=D_CL_PHYS,
    chloride_diffusivity_lattice=D_CL_LU,

    relaxation_time=TAU,
    mean_velocity_physical=mean_ux_phys,
    mean_speed_physical=mean_speed_phys,
    maximum_speed_physical=max_speed_phys,

    peclet_number=PE,
    porosity=porosity,
    transport_steps=TRANSPORT_STEPS,
    inlet_concentration=C_INLET
)

print(f"\nSaved BTC data: {OUTPUT_CSV}")
print(f"Saved final state: {OUTPUT_STATE}")


# ============================================================
# 12. PLOT BREAKTHROUGH CURVE
# ============================================================

# Plot physical time in nanoseconds for readability.
physical_time_ns = np.asarray(physical_times) * 1.0e9

plt.figure(figsize=(9, 5))

plt.plot(
    physical_time_ns,
    btc,
    linewidth=2,
    label="Conservative tracer"
)

plt.axhline(
    0.5,
    linestyle="--",
    alpha=0.7,
    label="50% outlet concentration"
)

plt.xlabel("Physical time (ns)")
plt.ylabel("Normalized outlet concentration (C/C0)")

plt.title(
    "Pore-scale Conservative Tracer Breakthrough"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300
)

plt.show()

print(f"Saved breakthrough plot: {OUTPUT_PLOT}")

print("\nSTEP 5.2 FINISHED.")
