# ============================================================
# File: Step_5.3_CrVI_Reactive_Transport.py
#
# Baseline Cr(VI) reactive transport through the existing 64^3 ZVI REV.
#
# Uses:
#   - Step_3.3_Geometry_512.raw
#   - Step_4.2_Velocity_Field.npz
#
# Literature kinetic model:
#   Alowitz & Scherer (2002), pH 7:
#       k_SA = 0.22 L m^-2 h^-1
#   Cr(VI) reduction is first-order in dissolved Cr(VI) and
#   first-order in ZVI surface-area concentration at pH 7.
#
# Local reaction:
#   k_obs,i [s^-1] = k_SA * (A_i / V_voxel) / 3600
#   C_i(t+dt) = C_i(t) * exp(-k_obs,i * dt_phys)
#
# Important scope:
#   - This is a BASELINE Cr(VI)-disappearance model.
#   - Fixed active ZVI surface: no passivation/aging yet.
#   - No competing ions or pH evolution yet.
#   - Cr(VI) disappearance is counted as Cr(VI) -> Cr(III)
#     conversion for electron-transfer bookkeeping, but Cr(III)
#     aqueous speciation/precipitation is NOT modelled here.
#   - PHREEQC coupling comes after this baseline is checked.
#
# Boundary conditions:
#   X: open solute inlet/outlet.
#   Y,Z: periodic.
#   Solid: no solute flux.
#
# Outputs:
#   Step_5.3_CrVI_BTC.csv
#   Step_5.3_CrVI_BTC.png
#   Step_5.3_CrVI_Final_State.npz
#   Step_5.3_Reactive_Surface_Map.npz
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

# Physical voxel spacing: retained from the calibrated geometry.
DX_PHYS = 0.6349e-9                 # m

# Water properties at approximately 25 C.
NU_PHYS = 1.0e-6                    # m^2/s

# CrO4(2-) aqueous diffusivity at approximately 25 C.
# This is a chromate-based baseline approximation for total dissolved
# Cr(VI) at pH 7; replace with a speciation-weighted value if needed.
D_CR6_PHYS = 1.132e-9               # m^2/s

# Inlet concentration inherited from the tracer setup.
C_INLET = 2.0                      # mmol/L

# Baseline chemistry condition.
PH = 7.0

# Alowitz & Scherer (2002), average surface-area-normalized Cr(VI)
# reduction coefficient at pH 7.
K_SA_L_M2_H = 0.22                 # L m^-2 h^-1

# Start with approximately one advective REV crossing.
# The 61.4-million-step tracer run can be repeated by changing this,
# but the estimated reaction over one REV is very small.
TRANSPORT_STEPS = 61_400_000
CHUNK_STEPS = 10_000

OUTPUT_CSV = "Step_5.3_CrVI_BTC.csv"
OUTPUT_PLOT = "Step_5.3_CrVI_BTC.png"
OUTPUT_STATE = "Step_5.3_CrVI_Final_State.npz"
OUTPUT_SURFACE = "Step_5.3_Reactive_Surface_Map.npz"

jax.config.update("jax_enable_x64", True)


# ============================================================
# 2. LOAD 64^3 GEOMETRY
# ============================================================

print("=" * 68)
print("STEP 5.3: BASELINE Cr(VI) REACTIVE TRANSPORT")
print("=" * 68)

if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(f"Geometry file not found: {RAW_FILE}")

print("\nLoading geometry...")
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
solid_np = ~fluid_np

NX, NY, NZ = fluid_np.shape
n_fluid = int(np.sum(fluid_np))
porosity = float(np.mean(fluid_np))

print(f"Grid: {NX} x {NY} x {NZ}")
print(f"Fluid voxels: {n_fluid:,}")
print(f"Porosity: {porosity:.6f}")


# ============================================================
# 3. BUILD REACTIVE IRON SURFACE MAP
# ============================================================

print("\nBuilding reactive iron surface map...")

# X is open, so do NOT wrap solid neighbours from x=0 to x=-1.
# Y and Z are periodic, matching the hydrodynamic/transport REV.

solid_xp = np.roll(solid_np, -1, axis=0)
solid_xp[-1, :, :] = False

solid_xm = np.roll(solid_np, 1, axis=0)
solid_xm[0, :, :] = False

solid_yp = np.roll(solid_np, -1, axis=1)
solid_ym = np.roll(solid_np, 1, axis=1)

solid_zp = np.roll(solid_np, -1, axis=2)
solid_zm = np.roll(solid_np, 1, axis=2)

exposed_faces = (
    (fluid_np & solid_xp).astype(np.int8)
    + (fluid_np & solid_xm).astype(np.int8)
    + (fluid_np & solid_yp).astype(np.int8)
    + (fluid_np & solid_ym).astype(np.int8)
    + (fluid_np & solid_zp).astype(np.int8)
    + (fluid_np & solid_zm).astype(np.int8)
)

reactive_surface_np = fluid_np & (exposed_faces > 0)

voxel_volume_m3 = DX_PHYS ** 3
voxel_volume_L = voxel_volume_m3 * 1000.0
voxel_face_area_m2 = DX_PHYS ** 2

# Area assigned to each adjacent fluid voxel.
surface_area_map_np = exposed_faces.astype(np.float64) * voxel_face_area_m2
surface_area_map_np[~fluid_np] = 0.0

total_surface_area_m2 = float(np.sum(surface_area_map_np))
total_fluid_volume_L = n_fluid * voxel_volume_L
surface_area_concentration_m2_L = (
    total_surface_area_m2 / total_fluid_volume_L
)

n_reactive_nodes = int(np.sum(reactive_surface_np))

print(f"Reactive fluid nodes: {n_reactive_nodes:,}")
print(f"Total exposed Fe surface area: {total_surface_area_m2:.8e} m^2")
print(f"Total fluid volume: {total_fluid_volume_L:.8e} L")
print(
    "Fe surface area / fluid volume: "
    f"{surface_area_concentration_m2_L:.8e} m^2/L"
)

np.savez_compressed(
    OUTPUT_SURFACE,
    fluid_mask=fluid_np,
    reactive_surface_mask=reactive_surface_np,
    exposed_faces=exposed_faces,
    surface_area_map_m2=surface_area_map_np,
    dx_physical_m=DX_PHYS,
    total_surface_area_m2=total_surface_area_m2,
    total_fluid_volume_L=total_fluid_volume_L,
    surface_area_concentration_m2_L=surface_area_concentration_m2_L
)
print(f"Saved surface map: {OUTPUT_SURFACE}")


# ============================================================
# 4. LOAD EXISTING LBM VELOCITY FIELD
# ============================================================

print("\nLoading saved LBM velocity field...")

if not os.path.exists(VELOCITY_FILE):
    raise FileNotFoundError(f"Velocity file not found: {VELOCITY_FILE}")

data = np.load(VELOCITY_FILE)
print("Available arrays:", data.files)


def find_array(npz, candidates, description):
    for name in candidates:
        if name in npz.files:
            arr = np.asarray(npz[name])
            print(f"Loaded {description}: {name}")
            return arr
    raise KeyError(
        f"Could not find {description}. Available arrays: {npz.files}"
    )


ux_np = find_array(data, ["ux", "velocity_x", "u_x", "vx"], "ux")
uy_np = find_array(data, ["uy", "velocity_y", "u_y", "vy"], "uy")
uz_np = find_array(data, ["uz", "velocity_z", "u_z", "vz"], "uz")

if "tau" in data.files:
    TAU = float(np.asarray(data["tau"]).item())
else:
    TAU = 0.6
    print("WARNING: tau missing; using 0.6, as in the original LBM setup.")

data.close()

if not (TAU > 0.5):
    raise ValueError("LBM tau must be greater than 0.5.")

if not (
    ux_np.shape == fluid_np.shape
    and uy_np.shape == fluid_np.shape
    and uz_np.shape == fluid_np.shape
):
    raise ValueError("Velocity arrays do not match the 64^3 geometry.")

ux_np = np.where(fluid_np, ux_np, 0.0)
uy_np = np.where(fluid_np, uy_np, 0.0)
uz_np = np.where(fluid_np, uz_np, 0.0)

speed_np = np.sqrt(ux_np**2 + uy_np**2 + uz_np**2)

mean_ux_lu = float(np.mean(ux_np[fluid_np]))
mean_speed_lu = float(np.mean(speed_np[fluid_np]))
max_speed_lu = float(np.max(speed_np))

print(f"Mean Ux (LU): {mean_ux_lu:.8e}")
print(f"Mean speed (LU): {mean_speed_lu:.8e}")
print(f"Maximum speed (LU): {max_speed_lu:.8e}")


# ============================================================
# 5. PHYSICAL / LATTICE SCALING
# ============================================================

CS2 = 1.0 / 3.0
NU_LU = CS2 * (TAU - 0.5)

DT_PHYS = NU_LU * DX_PHYS**2 / NU_PHYS
D_CR6_LU = D_CR6_PHYS * DT_PHYS / DX_PHYS**2

velocity_scale = DX_PHYS / DT_PHYS
mean_ux_phys = mean_ux_lu * velocity_scale
mean_speed_phys = mean_speed_lu * velocity_scale
max_speed_phys = max_speed_lu * velocity_scale

rev_length_m = NX * DX_PHYS
total_time_s = TRANSPORT_STEPS * DT_PHYS

# Local first-order rate based on local exposed surface area.
# kSA: L m^-2 h^-1; A_i/V_voxel: m^2/L; result: h^-1.
kobs_local_h_np = (
    K_SA_L_M2_H
    * surface_area_map_np
    / voxel_volume_L
)

kobs_local_s_np = kobs_local_h_np / 3600.0
kobs_average_s = (
    K_SA_L_M2_H
    * surface_area_concentration_m2_L
    / 3600.0
)

# First-order reaction over one nominal advective crossing time.
advective_crossing_time_s = (
    rev_length_m / max(abs(mean_ux_phys), 1.0e-30)
)
Da_REV = kobs_average_s * advective_crossing_time_s
predicted_conversion_one_REV = 1.0 - np.exp(-Da_REV)
predicted_conversion_full_run = 1.0 - np.exp(
    -kobs_average_s * total_time_s
)

print("\n" + "=" * 68)
print("PHYSICAL SCALING AND LITERATURE KINETICS")
print("=" * 68)
print(f"pH: {PH:.2f}")
print(f"Literature kSA: {K_SA_L_M2_H:.6g} L m^-2 h^-1")
print(f"Water nu: {NU_PHYS:.6e} m^2/s")
print(f"Chromate D: {D_CR6_PHYS:.6e} m^2/s")
print(f"tau: {TAU:.6f}")
print(f"nu_LU: {NU_LU:.8e}")
print(f"dt_phys: {DT_PHYS:.8e} s")
print(f"D_CrVI_LU: {D_CR6_LU:.8e}")
print(f"Mean physical Ux: {mean_ux_phys:.8e} m/s")
print(f"REV length: {rev_length_m:.8e} m")
print(f"Estimated advective crossing time: {advective_crossing_time_s:.8e} s")
print(f"Mean kobs from geometric surface/volume: {kobs_average_s:.8e} s^-1")
print(f"Approximate REV Damkohler number: {Da_REV:.8e}")
print(
    "Predicted conversion over one REV crossing: "
    f"{100.0 * predicted_conversion_one_REV:.8e} %"
)
print(f"Run duration: {total_time_s:.8e} s")
print(
    "Well-mixed estimate over full run: "
    f"{100.0 * predicted_conversion_full_run:.8e} %"
)

if mean_ux_lu <= 0:
    raise ValueError("Mean X velocity is not positive.")

# Explicit transport stability indicators.
cfl_diffusion = 6.0 * D_CR6_LU
cfl_advection = (
    np.max(np.abs(ux_np))
    + np.max(np.abs(uy_np))
    + np.max(np.abs(uz_np))
)
combined_stability = cfl_diffusion + cfl_advection

print(f"Transport stability indicator: {combined_stability:.8e}")
if combined_stability > 1.0:
    raise ValueError("Explicit transport step may be unstable.")


# ============================================================
# 6. JAX ARRAYS AND LOCAL REACTION FACTOR
# ============================================================

fluid_mask = jnp.asarray(fluid_np)
ux = jnp.asarray(ux_np, dtype=jnp.float64)
uy = jnp.asarray(uy_np, dtype=jnp.float64)
uz = jnp.asarray(uz_np, dtype=jnp.float64)

D = jnp.float64(D_CR6_LU)
C0 = jnp.float64(C_INLET)
dt_lu = jnp.float64(1.0)

kobs_local_s = jnp.asarray(kobs_local_s_np, dtype=jnp.float64)

# Numerically stable local fraction reacted during one lattice time step.
reaction_fraction = -jnp.expm1(-kobs_local_s * DT_PHYS)
reaction_survival = 1.0 - reaction_fraction

voxel_volume_L_jax = jnp.float64(voxel_volume_L)


# ============================================================
# 7. CONSERVATIVE ADVECTION-DIFFUSION + SURFACE REACTION
# ============================================================

@jax.jit
def transport_step(C):
    # ---------------- X faces: open inlet/outlet ----------------
    ux_face = 0.5 * (ux[:-1, :, :] + ux[1:, :, :])
    valid_x = fluid_mask[:-1, :, :] & fluid_mask[1:, :, :]

    C_up_x = jnp.where(ux_face >= 0.0, C[:-1, :, :], C[1:, :, :])
    adv_x = ux_face * C_up_x
    diff_x = -D * (C[1:, :, :] - C[:-1, :, :])

    internal_flux_x = jnp.where(valid_x, adv_x + diff_x, 0.0)

    Fx = jnp.zeros((NX + 1, NY, NZ), dtype=jnp.float64)
    Fx = Fx.at[1:NX, :, :].set(internal_flux_x)

    # Inlet: prescribed concentration for forward flow.
    u_in = ux[0, :, :]
    inlet_adv = u_in * jnp.where(u_in >= 0.0, C0, C[0, :, :])
    inlet_diff = D * (C0 - C[0, :, :])
    inlet_flux = jnp.where(
        fluid_mask[0, :, :],
        inlet_adv + inlet_diff,
        0.0
    )
    Fx = Fx.at[0, :, :].set(inlet_flux)

    # Outlet: advective outflow, zero diffusive gradient.
    u_out = ux[-1, :, :]
    outlet_adv = u_out * jnp.where(u_out >= 0.0, C[-1, :, :], 0.0)
    outlet_flux = jnp.where(
        fluid_mask[-1, :, :],
        outlet_adv,
        0.0
    )
    Fx = Fx.at[NX, :, :].set(outlet_flux)

    # ---------------- Y faces: periodic ----------------
    C_yp = jnp.roll(C, -1, axis=1)
    uy_yp = jnp.roll(uy, -1, axis=1)
    mask_yp = jnp.roll(fluid_mask, -1, axis=1)

    uy_face = 0.5 * (uy + uy_yp)
    valid_y = fluid_mask & mask_yp
    C_up_y = jnp.where(uy_face >= 0.0, C, C_yp)

    Fy = jnp.where(
        valid_y,
        uy_face * C_up_y - D * (C_yp - C),
        0.0
    )

    # ---------------- Z faces: periodic ----------------
    C_zp = jnp.roll(C, -1, axis=2)
    uz_zp = jnp.roll(uz, -1, axis=2)
    mask_zp = jnp.roll(fluid_mask, -1, axis=2)

    uz_face = 0.5 * (uz + uz_zp)
    valid_z = fluid_mask & mask_zp
    C_up_z = jnp.where(uz_face >= 0.0, C, C_zp)

    Fz = jnp.where(
        valid_z,
        uz_face * C_up_z - D * (C_zp - C),
        0.0
    )

    # ---------------- Conservative divergence ----------------
    div_x = Fx[1:, :, :] - Fx[:-1, :, :]
    div_y = Fy - jnp.roll(Fy, 1, axis=1)
    div_z = Fz - jnp.roll(Fz, 1, axis=2)

    C_adv = C - dt_lu * (div_x + div_y + div_z)
    C_adv = jnp.where(fluid_mask, C_adv, 0.0)

    # Convert boundary fluxes to mmol per lattice step.
    mass_in_mmol = dt_lu * jnp.sum(Fx[0, :, :]) * voxel_volume_L_jax
    mass_out_mmol = dt_lu * jnp.sum(Fx[-1, :, :]) * voxel_volume_L_jax

    return C_adv, mass_in_mmol, mass_out_mmol


@jax.jit
def reactive_step(C_adv):
    # Surface reaction only occurs in fluid cells adjacent to solid Fe.
    reacted_concentration = C_adv * reaction_fraction
    C_after = C_adv * reaction_survival
    C_after = jnp.where(fluid_mask, C_after, 0.0)

    reacted_mmol = jnp.sum(reacted_concentration) * voxel_volume_L_jax
    return C_after, reacted_mmol


@jax.jit
def one_step(C):
    C_adv, mass_in, mass_out = transport_step(C)
    C_next, mass_reacted = reactive_step(C_adv)
    return C_next, mass_in, mass_out, mass_reacted


# ============================================================
# 8. COMPILED CHUNK
# ============================================================

@jax.jit
def advance_chunk(C):
    def body(_, carry):
        C_now, m_in, m_out, m_reacted = carry
        C_next, dm_in, dm_out, dm_reacted = one_step(C_now)
        return (
            C_next,
            m_in + dm_in,
            m_out + dm_out,
            m_reacted + dm_reacted
        )

    return lax.fori_loop(
        0,
        CHUNK_STEPS,
        body,
        (
            C,
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0)
        )
    )


# ============================================================
# 9. INITIALIZE AND RUN
# ============================================================

C_grid = jnp.zeros((NX, NY, NZ), dtype=jnp.float64)

initial_mass_mmol = float(jnp.sum(C_grid) * voxel_volume_L)

outlet_weight = (
    jnp.maximum(ux[-1, :, :], 0.0)
    * fluid_mask[-1, :, :]
)
outlet_flow = jnp.sum(outlet_weight)

if float(outlet_flow) <= 1.0e-20:
    raise ValueError("No positive outlet flow detected.")

if TRANSPORT_STEPS % CHUNK_STEPS != 0:
    raise ValueError("TRANSPORT_STEPS must be divisible by CHUNK_STEPS.")

num_chunks = TRANSPORT_STEPS // CHUNK_STEPS

cumulative_in_mmol = 0.0
cumulative_out_mmol = 0.0
cumulative_reacted_mmol = 0.0

steps_record = []
time_record_s = []
btc_record = []
stored_record_mmol = []
reacted_record_mmol = []
balance_record_mmol = []

print("\nStarting Cr(VI) reactive transport...")
print(f"Total steps: {TRANSPORT_STEPS:,}")
print(f"Chunk size: {CHUNK_STEPS:,}")
print(f"Number of chunks: {num_chunks:,}")

t0 = time.time()

for chunk in range(num_chunks):
    (
        C_grid,
        dm_in,
        dm_out,
        dm_reacted
    ) = advance_chunk(C_grid)

    cumulative_in_mmol += float(dm_in)
    cumulative_out_mmol += float(dm_out)
    cumulative_reacted_mmol += float(dm_reacted)

    current_step = (chunk + 1) * CHUNK_STEPS
    current_time_s = current_step * DT_PHYS

    outlet_c = (
        jnp.sum(outlet_weight * C_grid[-1, :, :]) / outlet_flow
    )
    normalized_outlet = float(outlet_c) / C_INLET

    stored_mmol = float(jnp.sum(C_grid) * voxel_volume_L)

    # Initial + inlet = stored + outlet + reacted.
    balance_residual = (
        initial_mass_mmol
        + cumulative_in_mmol
        - stored_mmol
        - cumulative_out_mmol
        - cumulative_reacted_mmol
    )

    steps_record.append(current_step)
    time_record_s.append(current_time_s)
    btc_record.append(normalized_outlet)
    stored_record_mmol.append(stored_mmol)
    reacted_record_mmol.append(cumulative_reacted_mmol)
    balance_record_mmol.append(balance_residual)

    if (
        (chunk + 1) % 50 == 0
        or chunk == 0
        or chunk + 1 == num_chunks
    ):
        denom = max(
            abs(cumulative_in_mmol),
            abs(stored_mmol),
            abs(cumulative_out_mmol + cumulative_reacted_mmol),
            1.0e-30
        )
        rel_error = abs(balance_residual) / denom

        print(
            f"Step {current_step:>12,} | "
            f"t={current_time_s:.3e} s | "
            f"Cout/Cin={normalized_outlet:.6f} | "
            f"reacted={cumulative_reacted_mmol:.4e} mmol | "
            f"mass rel.err={rel_error:.3e}"
        )

elapsed = time.time() - t0


# ============================================================
# 10. FINAL DIAGNOSTICS
# ============================================================

final_stored_mmol = float(jnp.sum(C_grid) * voxel_volume_L)

final_residual_mmol = (
    initial_mass_mmol
    + cumulative_in_mmol
    - final_stored_mmol
    - cumulative_out_mmol
    - cumulative_reacted_mmol
)

balance_scale = max(
    abs(cumulative_in_mmol),
    abs(final_stored_mmol),
    abs(cumulative_out_mmol + cumulative_reacted_mmol),
    1.0e-30
)
relative_balance_error = abs(final_residual_mmol) / balance_scale

total_in_mmol = cumulative_in_mmol
total_removed_mmol = cumulative_reacted_mmol
if abs(total_in_mmol) > 1.0e-30:
    reacted_fraction_of_inlet_mass = total_removed_mmol / total_in_mmol
else:
    reacted_fraction_of_inlet_mass = 0.0

print("\n" + "=" * 68)
print("FINAL Cr(VI) REACTIVE TRANSPORT RESULTS")
print("=" * 68)
print(f"Execution time: {elapsed:.2f} s")
print(f"Physical simulation time: {time_record_s[-1]:.8e} s")
print(f"Initial Cr(VI) mass: {initial_mass_mmol:.8e} mmol")
print(f"Cr(VI) inlet mass: {cumulative_in_mmol:.8e} mmol")
print(f"Cr(VI) outlet mass: {cumulative_out_mmol:.8e} mmol")
print(f"Cr(VI) reacted (Cr(III) formed): {cumulative_reacted_mmol:.8e} mmol")
print(f"Cr(VI) stored in REV: {final_stored_mmol:.8e} mmol")
print(f"Mass-balance residual: {final_residual_mmol:.8e} mmol")
print(f"Relative mass-balance error: {relative_balance_error:.8e}")
print(f"Final Cout/Cin: {btc_record[-1]:.8f}")
print(
    "Reacted / inlet mass: "
    f"{100.0 * reacted_fraction_of_inlet_mass:.8e} %"
)

# Concentration fields are stored in mmol/L.
np.savez_compressed(
    OUTPUT_STATE,
    crvi_concentration_mmol_L=np.asarray(C_grid),
    fluid_mask=fluid_np,
    reactive_surface_mask=reactive_surface_np,
    surface_area_map_m2=surface_area_map_np,
    ux=ux_np,
    uy=uy_np,
    uz=uz_np,
    time_steps=np.asarray(steps_record),
    physical_time_s=np.asarray(time_record_s),
    normalized_outlet_concentration=np.asarray(btc_record),
    cumulative_inlet_mmol=cumulative_in_mmol,
    cumulative_outlet_mmol=cumulative_out_mmol,
    cumulative_reacted_mmol=cumulative_reacted_mmol,
    final_stored_mmol=final_stored_mmol,
    mass_balance_residual_mmol=final_residual_mmol,
    relative_mass_balance_error=relative_balance_error,
    dx_physical_m=DX_PHYS,
    dt_physical_s=DT_PHYS,
    tau=TAU,
    nu_physical_m2_s=NU_PHYS,
    diffusivity_crvi_m2_s=D_CR6_PHYS,
    diffusivity_crvi_lu=D_CR6_LU,
    ksa_L_m2_h=K_SA_L_M2_H,
    pH=PH,
    mean_kobs_s_1=kobs_average_s,
    surface_area_m2=total_surface_area_m2,
    fluid_volume_L=total_fluid_volume_L,
    surface_area_concentration_m2_L=surface_area_concentration_m2_L,
    porosity=porosity,
    inlet_concentration_mmol_L=C_INLET,
    transport_steps=TRANSPORT_STEPS
)

btc_out = np.column_stack([
    np.asarray(steps_record),
    np.asarray(time_record_s),
    np.asarray(btc_record),
    np.asarray(stored_record_mmol),
    np.asarray(reacted_record_mmol),
    np.asarray(balance_record_mmol)
])

np.savetxt(
    OUTPUT_CSV,
    btc_out,
    delimiter=",",
    header=(
        "Time_steps_LU,"
        "Physical_time_s,"
        "Normalized_CrVI_outlet_C_over_C0,"
        "Stored_CrVI_mmol,"
        "Cumulative_CrVI_reacted_mmol,"
        "Mass_balance_residual_mmol"
    ),
    comments=""
)

# Plot the Cr(VI) outlet curve.
plt.figure(figsize=(9, 5))
plt.plot(
    np.asarray(time_record_s) * 1.0e9,
    btc_record,
    linewidth=2,
    label="Reactive Cr(VI)"
)
plt.xlabel("Physical time (ns)")
plt.ylabel("Normalized Cr(VI) outlet concentration (C/C0)")
plt.title("Cr(VI) Breakthrough with Literature-Based ZVI Surface Reaction")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

print(f"\nSaved BTC: {OUTPUT_CSV}")
print(f"Saved final state: {OUTPUT_STATE}")
print(f"Saved plot: {OUTPUT_PLOT}")
print(f"Saved surface map: {OUTPUT_SURFACE}")
print("\nSTEP 5.3 FINISHED.")

print("\nMODEL SCOPE REMINDER:")
print("- pH fixed at 7.0.")
print("- Literature kSA fixed at 0.22 L m^-2 h^-1.")
print("- Active Fe surface assumed constant; no passivation yet.")
print("- No competing ions or Fe corrosion/speciation yet.")
print("- Cr(III) production is tracked by Cr(VI) reaction mass.")
print("- PHREEQC is not called by this script.")
