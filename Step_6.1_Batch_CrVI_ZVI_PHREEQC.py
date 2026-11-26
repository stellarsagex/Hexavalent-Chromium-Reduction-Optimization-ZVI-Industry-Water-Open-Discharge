# ============================================================
# File: Step_6.1_Batch_CrVI_ZVI_PHREEQC.py
#
# PURPOSE
#   Batch-scale baseline model for Cr(VI) reduction by zero-valent
#   iron (ZVI), using the heterogeneous kinetic structure reported
#   by Mitra et al. (2011), as reproduced in Mitra et al. (2014).
#
#   PHREEQC is called at saved time points to calculate aqueous
#   Cr/Fe speciation and saturation indices at the imposed pH.
#
# IMPORTANT MODEL LIMITATIONS
#   1. This is the steel-plate kinetic structure:
#        dC/dt = -k1*A*C^(2/3) - k2*A*C
#        dA/dt = -alpha*k2*A*C
#      It is NOT the additional moving-boundary/product-layer
#      diffusion model used for steel wool in Mitra et al. (2014).
#
#   2. k1, k2 and alpha below are the values reproduced in the
#      2014 paper from Mitra et al. (2011). The original 2011
#      full-text dataset is not available in the current project
#      files. Therefore, a run without measured data is a forward
#      model / implementation check, NOT experimental validation.
#
#   3. The 2014 steel-wool A0=0.13 cm2 is used only as a provisional
#      demonstration value. It is not the steel-plate area from
#      Mitra et al. (2011). Replace A0 with the actual exposed area
#      of the steel used in the experiment being reproduced.
#
#   4. The Mitra stoichiometric reaction is:
#        Fe(0) + Cr(VI) + 7 H+ -> Fe(III) + Cr(III) + ...
#      i.e. 1 mol Fe(0) consumed per mol Cr(VI) reduced.
#      The primary-reduction branch is sent to PHREEQC as aqueous
#      Fe(III) and Cr(III). The passivation branch is tracked as
#      surface-associated Fe(III)-Cr(III) product inventory.
#      This 1:1 surface-product bookkeeping is an explicit
#      stoichiometric assumption, not a resolved solid-phase model.
#
#   5. pH is held at PH_FIXED for this baseline comparison.
#      The paper reports initial pH adjustment but does not provide
#      the acid/base dose needed to close a proton balance. Therefore
#      this script does NOT claim to predict pH drift.
#
#   6. Fe(II)-producing corrosion independent of Cr(VI) reduction
#      is deliberately excluded from this first Mitra baseline.
#      Adding it now would introduce a second electron donor pathway
#      and risk double-counting reduction. It can be added after
#      this baseline is reproduced and separately parameterized.
#
# INPUTS
#   - MINTEQ v4 database (path below)
#   - Optional experimental_data.csv with columns:
#         time_min,CrVI_mg_L
#
# OUTPUTS
#   - Step_6.1_Batch_Kinetics.csv
#   - Step_6.1_PHREEQC_Speciation.csv
#   - Step_6.1_Batch_CrVI.png
#   - Step_6.1_Batch_Active_Area.png
#   - Step_6.1_Batch_Summary.txt
#
# Requirements:
#   pip install numpy scipy pandas matplotlib phreeqpython
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from phreeqpython import PhreeqPython


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PHREEQC_DATABASE = Path(
    r"C:\Users\Saksham Shrestha\Desktop\Chromium\minteq.v4.dat"
)

# Batch experimental conditions (Mitra et al. 2014 method)
BATCH_VOLUME_L = 0.050
INITIAL_CRVI_MG_L = 100.0       # mg/L as elemental Cr(VI)
PH_FIXED = 3.0
TEMPERATURE_C = 31.0
REACTION_TIME_MIN = 120.0

# ZVI/steel properties from Mitra et al. (2014) steel-wool experiment.
# A0 is provisional for demonstration only; see file header.
STEEL_MASS_G = 0.300
FE0_MASS_FRACTION = 0.932
A0_CM2 = 0.13

# Mitra et al. (2011) kinetic constants, reproduced in Mitra et al. (2014).
# Units must remain consistent with C in mg/L and A in cm2.
K1 = 1.04e-5       # mg^(1/3) L^(-1/3) s^(-1) cm^(-2)
K2 = 2.69e-5       # cm^(-2) s^(-1)
ALPHA = 0.45       # cm^2 L mg^(-1)

# Molecular weights
MW_CR_G_MOL = 51.9961
MW_FE_G_MOL = 55.845

# Time discretization for reporting / PHREEQC snapshots
N_OUTPUT = 241

# Optional experimental data file.
# CSV headers must be exactly: time_min,CrVI_mg_L
EXPERIMENTAL_CSV = BASE_DIR / "experimental_data.csv"

# Output paths
KINETICS_CSV = BASE_DIR / "Step_6.1_Batch_Kinetics.csv"
PHREEQC_CSV = BASE_DIR / "Step_6.1_PHREEQC_Speciation.csv"
CRVI_PLOT = BASE_DIR / "Step_6.1_Batch_CrVI.png"
AREA_PLOT = BASE_DIR / "Step_6.1_Batch_Active_Area.png"
SUMMARY_TXT = BASE_DIR / "Step_6.1_Batch_Summary.txt"


# ============================================================
# 2. CHECK INPUTS AND PRINT MODEL DEFINITION
# ============================================================

def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def main():
    print("=" * 72)
    print("STEP 6.1: BATCH Cr(VI)-ZVI KINETICS + PHREEQC SPECIATION")
    print("=" * 72)

    require_file(PHREEQC_DATABASE, "PHREEQC database")

    if BATCH_VOLUME_L <= 0 or INITIAL_CRVI_MG_L <= 0:
        raise ValueError("Batch volume and initial Cr(VI) must be positive.")
    if A0_CM2 <= 0 or STEEL_MASS_G <= 0:
        raise ValueError("Initial reactive area and steel mass must be positive.")

    print("\nBatch conditions")
    print(f"Volume:                 {BATCH_VOLUME_L:.5f} L")
    print(f"Initial Cr(VI):         {INITIAL_CRVI_MG_L:.3f} mg/L as Cr")
    print(f"Initial pH (imposed):   {PH_FIXED:.2f}")
    print(f"Temperature:            {TEMPERATURE_C:.1f} C")
    print(f"Duration:               {REACTION_TIME_MIN:.1f} min")
    print(f"Steel mass:             {STEEL_MASS_G:.4f} g")
    print(f"Fe(0) mass fraction:    {FE0_MASS_FRACTION:.4f}")
    print(f"Initial active area:    {A0_CM2:.5f} cm2 [PROVISIONAL]")

    print("\nKinetic constants")
    print(f"k1:     {K1:.6e}")
    print(f"k2:     {K2:.6e}")
    print(f"alpha:  {ALPHA:.6e}")

    print("\nKinetic equations")
    print("r1 = k1 * A * C^(2/3)        [primary reduction]")
    print("r2 = k2 * A * C              [passivation-associated uptake]")
    print("dC/dt = -(r1 + r2)")
    print("dA/dt = -alpha * r2")

    print("\nNOTE: A0=0.13 cm2 comes from the 2014 steel-wool study.")
    print("It is not the steel-plate area in the original 2011 study.")
    print("Replace A0 before interpreting the predicted rate quantitatively.")


    # ========================================================
    # 3. SOLVE THE BATCH KINETIC MODEL
    # ========================================================

    t_end_s = REACTION_TIME_MIN * 60.0
    t_eval_s = np.linspace(0.0, t_end_s, N_OUTPUT)

    # State vector:
    # y[0] = C: remaining dissolved Cr(VI), mg/L
    # y[1] = A: remaining active iron area, cm2
    # y[2] = Q1: cumulative Cr(VI) uptake through primary route, mg/L
    # y[3] = Q2: cumulative Cr(VI) uptake through passivation route, mg/L

    def rhs(t, y):
        C = max(float(y[0]), 0.0)
        A = max(float(y[1]), 0.0)

        r1 = K1 * A * (C ** (2.0 / 3.0)) if C > 0.0 else 0.0
        r2 = K2 * A * C

        dC = -(r1 + r2)
        dA = -ALPHA * r2

        return [dC, dA, r1, r2]

    y0 = [INITIAL_CRVI_MG_L, A0_CM2, 0.0, 0.0]

    sol = solve_ivp(
        rhs,
        (0.0, t_end_s),
        y0,
        method="LSODA",
        t_eval=t_eval_s,
        rtol=1e-9,
        atol=1e-11,
        max_step=30.0
    )

    if not sol.success:
        raise RuntimeError(f"Kinetic ODE solver failed: {sol.message}")

    C = np.maximum(sol.y[0], 0.0)
    A = np.maximum(sol.y[1], 0.0)
    Q1 = np.maximum(sol.y[2], 0.0)
    Q2 = np.maximum(sol.y[3], 0.0)

    # Numerical mass-balance check in concentration space.
    kinetic_residual = INITIAL_CRVI_MG_L - (C + Q1 + Q2)
    max_kinetic_residual = float(np.max(np.abs(kinetic_residual)))

    time_min = sol.t / 60.0

    # Convert concentration uptake to total mass in the batch.
    primary_cr_mg = Q1 * BATCH_VOLUME_L
    passivation_cr_mg = Q2 * BATCH_VOLUME_L
    total_cr_removed_mg = primary_cr_mg + passivation_cr_mg

    # Reaction stoichiometry from the cited overall reaction:
    # 1 mol Fe(0) consumed per mol Cr(VI) reduced.
    primary_cr_mol = primary_cr_mg / 1000.0 / MW_CR_G_MOL
    passivation_cr_mol = passivation_cr_mg / 1000.0 / MW_CR_G_MOL
    total_cr_reduced_mol = primary_cr_mol + passivation_cr_mol

    fe0_initial_g = STEEL_MASS_G * FE0_MASS_FRACTION
    fe0_initial_mol = fe0_initial_g / MW_FE_G_MOL
    fe0_consumed_mol = total_cr_reduced_mol
    fe0_consumed_g = fe0_consumed_mol * MW_FE_G_MOL
    fe0_remaining_g = fe0_initial_g - fe0_consumed_g

    if np.min(fe0_remaining_g) < -1e-10:
        warnings.warn(
            "Predicted Fe(0) consumption exceeds available Fe(0). "
            "The kinetic run must be constrained by finite iron inventory."
        )

    # Aqueous products from primary route; passivation-route products
    # are tracked separately as surface-associated product inventory.
    aqueous_criii_mol = primary_cr_mol
    aqueous_feiii_mol = primary_cr_mol
    surface_criii_mol = passivation_cr_mol
    surface_feiii_mol = passivation_cr_mol

    kinetics = pd.DataFrame({
        "time_s": sol.t,
        "time_min": time_min,
        "CrVI_mg_L": C,
        "active_area_cm2": A,
        "active_area_fraction": A / A0_CM2,
        "primary_route_uptake_mg_L": Q1,
        "passivation_route_uptake_mg_L": Q2,
        "total_CrVI_removed_mg_L": Q1 + Q2,
        "CrVI_removed_total_mg": total_cr_removed_mg,
        "aqueous_CrIII_mol": aqueous_criii_mol,
        "aqueous_FeIII_mol": aqueous_feiii_mol,
        "surface_CrIII_product_mol": surface_criii_mol,
        "surface_FeIII_product_mol": surface_feiii_mol,
        "Fe0_consumed_mol": fe0_consumed_mol,
        "Fe0_consumed_g": fe0_consumed_g,
        "Fe0_remaining_g": fe0_remaining_g,
        "kinetic_Cr_mass_balance_residual_mg_L": kinetic_residual
    })

    kinetics.to_csv(KINETICS_CSV, index=False)


    # ========================================================
    # 4. PHREEQC SPECIATION AT EACH SAVED TIME POINT
    # ========================================================

    print("\nInitializing PHREEQC with MINTEQ v4...")
    pp = PhreeqPython(database=str(PHREEQC_DATABASE))

    phreeqc_rows = []

    # Potassium is carried over from K2Cr2O7 at approximately 1 mol K
    # per mol Cr atom. Chloride is charge-balanced by PHREEQC because
    # the exact HCl/NaOH dose used to adjust pH is not reported here.
    initial_cr_mol_L = (
        INITIAL_CRVI_MG_L / 1000.0 / MW_CR_G_MOL
    )

    for i, row in kinetics.iterrows():
        cr6_mol_kgw = (
            float(row["CrVI_mg_L"]) / 1000.0 / MW_CR_G_MOL
        )

        # Primary-route products are represented as dissolved totals.
        # Passivation products remain in the surface-associated pool.
        cr3_aq_mol_kgw = float(row["aqueous_CrIII_mol"]) / BATCH_VOLUME_L
        fe3_aq_mol_kgw = float(row["aqueous_FeIII_mol"]) / BATCH_VOLUME_L

        solution_input = {
            "units": "mol/kgw",
            "temp": TEMPERATURE_C,
            "pH": PH_FIXED,
            "pe": 4.0,
            "K": initial_cr_mol_L,
            "Cl": "1 charge",
            "Cr(6)": cr6_mol_kgw,
            "Cr(3)": cr3_aq_mol_kgw,
            "Fe(3)": fe3_aq_mol_kgw
        }

        try:
            aq = pp.add_solution(solution_input)
        except Exception as exc:
            raise RuntimeError(
                "\nPHREEQC could not create a solution with the MINTEQ v4 "
                "oxidation-state inputs Cr(6), Cr(3), and Fe(3).\n"
                "Check the database master-species names and PhreeqPython "
                "version. The kinetic ODE has already been solved, but "
                "chemical speciation was not completed.\n"
                f"Underlying error: {exc}"
            ) from exc

        # Species concentrations in mol/kgw, as returned by PhreeqPython.
        species = aq.species
        cr_species = {
            name: float(value)
            for name, value in species.items()
            if "Cr" in name
        }
        fe_species = {
            name: float(value)
            for name, value in species.items()
            if "Fe" in name
        }

        # Calculate saturation indices for Fe/Cr-bearing phases that
        # are actually present in the loaded database.
        si_values = {}
        try:
            phase_names = list(aq.phases.keys())
        except Exception:
            phase_names = []

        for phase_name in phase_names:
            if ("Cr" in phase_name) or ("Fe" in phase_name):
                try:
                    si_values[phase_name] = float(aq.si(phase_name))
                except Exception:
                    pass

        try:
            total_cr = float(aq.total("Cr", "mol"))
        except Exception:
            total_cr = np.nan

        try:
            total_fe = float(aq.total("Fe", "mol"))
        except Exception:
            total_fe = np.nan

        phreeqc_rows.append({
            "time_min": float(row["time_min"]),
            "pH_imposed": float(aq.pH),
            "CrVI_input_molal": cr6_mol_kgw,
            "CrIII_aqueous_input_molal": cr3_aq_mol_kgw,
            "FeIII_aqueous_input_molal": fe3_aq_mol_kgw,
            "PHREEQC_total_Cr_mol": total_cr,
            "PHREEQC_total_Fe_mol": total_fe,
            "Cr_aqueous_species_molal_json": json.dumps(cr_species),
            "Fe_aqueous_species_molal_json": json.dumps(fe_species),
            "Fe_Cr_phase_SI_json": json.dumps(si_values)
        })

        if i % 40 == 0 or i == len(kinetics) - 1:
            print(
                f"PHREEQC snapshot {i + 1:3d}/{len(kinetics)} | "
                f"t={row['time_min']:7.2f} min | "
                f"Cr(VI)={row['CrVI_mg_L']:10.4f} mg/L | "
                f"pH={aq.pH:.3f}"
            )

    phreeqc_df = pd.DataFrame(phreeqc_rows)
    phreeqc_df.to_csv(PHREEQC_CSV, index=False)


    # ========================================================
    # 5. OPTIONAL COMPARISON WITH EXPERIMENTAL DATA
    # ========================================================

    experimental_metrics = None
    if EXPERIMENTAL_CSV.exists():
        exp = pd.read_csv(EXPERIMENTAL_CSV)

        required_columns = {"time_min", "CrVI_mg_L"}
        if not required_columns.issubset(exp.columns):
            raise ValueError(
                "experimental_data.csv must contain columns: "
                "time_min,CrVI_mg_L"
            )

        exp = exp.dropna(subset=["time_min", "CrVI_mg_L"]).copy()
        predicted_at_exp = np.interp(
            exp["time_min"].to_numpy(),
            time_min,
            C
        )

        residuals = predicted_at_exp - exp["CrVI_mg_L"].to_numpy()
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum(
            (exp["CrVI_mg_L"].to_numpy()
             - np.mean(exp["CrVI_mg_L"].to_numpy())) ** 2
        ))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        experimental_metrics = {
            "n_points": int(len(exp)),
            "RMSE_mg_L": rmse,
            "R2": r2
        }

        print("\nExperimental comparison")
        print(f"Points: {len(exp)}")
        print(f"RMSE:   {rmse:.6g} mg/L")
        print(f"R2:     {r2:.6g}")
    else:
        print("\nNo experimental_data.csv found.")
        print("No experimental validation metrics have been calculated.")


    # ========================================================
    # 6. PLOTS
    # ========================================================

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(time_min, C, linewidth=2.2, label="Mitra kinetic model")

    if EXPERIMENTAL_CSV.exists():
        plt.scatter(
            exp["time_min"],
            exp["CrVI_mg_L"],
            s=28,
            marker="o",
            label="Experimental data"
        )

    plt.xlabel("Time (min)")
    plt.ylabel("Cr(VI) concentration (mg/L as Cr)")
    plt.title("Batch Cr(VI) Reduction by ZVI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CRVI_PLOT, dpi=300)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        time_min,
        A / A0_CM2,
        linewidth=2.2,
        label="Remaining active area"
    )
    plt.xlabel("Time (min)")
    plt.ylabel("A(t) / A0 (-)")
    plt.title("Loss of Active Iron Surface by Passivation")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(AREA_PLOT, dpi=300)
    plt.close()


    # ========================================================
    # 7. SUMMARY AND MASS-BALANCE REPORT
    # ========================================================

    initial_cr_mass_mg = INITIAL_CRVI_MG_L * BATCH_VOLUME_L
    final_crvi_mass_mg = C[-1] * BATCH_VOLUME_L
    removed_cr_mass_mg = total_cr_removed_mg[-1]

    cr_balance_error_mg = (
        initial_cr_mass_mg
        - final_crvi_mass_mg
        - removed_cr_mass_mg
    )

    final_removal_pct = (
        100.0 * (INITIAL_CRVI_MG_L - C[-1]) / INITIAL_CRVI_MG_L
    )

    summary_lines = [
        "STEP 6.1 BATCH Cr(VI)-ZVI + PHREEQC SUMMARY",
        "=" * 60,
        f"Initial Cr(VI): {INITIAL_CRVI_MG_L:.8g} mg/L as Cr",
        f"Initial pH imposed: {PH_FIXED:.5g}",
        f"Temperature: {TEMPERATURE_C:.5g} C",
        f"Batch volume: {BATCH_VOLUME_L:.8g} L",
        f"Reaction time: {REACTION_TIME_MIN:.8g} min",
        "",
        "KINETICS",
        f"Initial active area: {A0_CM2:.8g} cm2 [provisional wool value]",
        f"Final active area: {A[-1]:.8g} cm2",
        f"Final active-area fraction: {A[-1] / A0_CM2:.8g}",
        f"Final Cr(VI): {C[-1]:.8g} mg/L",
        f"Cr(VI) removal: {final_removal_pct:.8g} %",
        f"Primary-route uptake: {Q1[-1]:.8g} mg/L",
        f"Passivation-route uptake: {Q2[-1]:.8g} mg/L",
        f"Maximum kinetic concentration-balance residual: "
        f"{max_kinetic_residual:.8e} mg/L",
        "",
        "IRON INVENTORY",
        f"Initial Fe(0): {fe0_initial_g:.8g} g",
        f"Fe(0) consumed: {fe0_consumed_g[-1]:.8g} g",
        f"Fe(0) remaining: {fe0_remaining_g[-1]:.8g} g",
        "",
        "BATCH CHROMIUM MASS BALANCE",
        f"Initial Cr(VI) mass: {initial_cr_mass_mg:.8g} mg",
        f"Final dissolved Cr(VI) mass: {final_crvi_mass_mg:.8g} mg",
        f"Total Cr(VI) removed: {removed_cr_mass_mg:.8g} mg",
        f"Mass-balance residual: {cr_balance_error_mg:.8e} mg",
        "",
        "PHREEQC",
        f"PHREEQC snapshots: {len(phreeqc_df)}",
        "pH is imposed/fixed; this run does not predict pH drift.",
        "PHREEQC computes aqueous speciation and saturation indices.",
        "No solid phase is allowed to precipitate in this baseline run.",
        "Positive saturation index indicates thermodynamic supersaturation,",
        "not proof that precipitation occurred during the experiment.",
        "",
        "VALIDATION STATUS",
        "A numerical mass balance is not experimental validation.",
        "A run without measured concentration-time data is a forward run only.",
        "The 2014 steel-wool area is not the 2011 steel-plate area.",
        "Use the original experiment's area or fit only with independent",
        "experimental concentration-time data before claiming validation."
    ]

    if experimental_metrics is not None:
        summary_lines.extend([
            "",
            "EXPERIMENTAL COMPARISON",
            f"Number of points: {experimental_metrics['n_points']}",
            f"RMSE: {experimental_metrics['RMSE_mg_L']:.8g} mg/L",
            f"R2: {experimental_metrics['R2']:.8g}"
        ])

    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FINAL BATCH RESULTS")
    print("=" * 72)
    print(f"Final Cr(VI):          {C[-1]:.6f} mg/L")
    print(f"Cr(VI) removal:        {final_removal_pct:.6f} %")
    print(f"Final active area:     {A[-1]:.6e} cm2")
    print(f"Fe(0) consumed:        {fe0_consumed_g[-1]:.6e} g")
    print(f"Cr mass balance error: {cr_balance_error_mg:.6e} mg")
    print("\nSaved:")
    print(KINETICS_CSV.name)
    print(PHREEQC_CSV.name)
    print(CRVI_PLOT.name)
    print(AREA_PLOT.name)
    print(SUMMARY_TXT.name)

    print("\nIMPORTANT:")
    print("This is not yet a validated model unless experimental data are")
    print("provided and the correct geometry/area basis is used.")
    print("Do not transfer these provisional batch parameters to the voxel")
    print("model until the batch kinetics are calibrated and checked.")


if __name__ == "__main__":
    main()
