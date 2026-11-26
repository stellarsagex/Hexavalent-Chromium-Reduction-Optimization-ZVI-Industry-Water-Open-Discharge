#Checking the PHREEQC database before running the validation model
# PHREEQC Check

# MINTEQ V4 DATABASE & CHROMIUM VALIDATION

from pathlib import Path
from phreeqpython import PhreeqPython

print("=" * 65)
print(" MINTEQ V4 — CHROMIUM CHEMISTRY VALIDATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOCATING DATABASE
# ------------------------------------------------------------

project_dir = Path(__file__).resolve().parent
db_path = project_dir / "minteq.v4.dat"

print("\n[1] DATABASE CHECK")
print("Database path:", db_path)
print("Database exists:", db_path.is_file())

if not db_path.is_file():
    raise FileNotFoundError(
        f"Cannot find MINTEQ database: {db_path}"
    )

print("Database size:", db_path.stat().st_size, "bytes")

# ------------------------------------------------------------
# 2. INITIALIZING PHREEQC
# ------------------------------------------------------------

print("\n[2] INITIALIZING PHREEQC")

pp = PhreeqPython(
    database="minteq.v4.dat",
    database_directory=project_dir
)

# ------------------------------------------------------------
# 3. VERIFYING DATABASE LOADING
# ------------------------------------------------------------

print("\n[3] DATABASE LOADING STATUS")

error_count = pp.ip.phc_database_error_count

print("Database errors:", error_count)

if error_count != 0:
    print(pp.ip.get_error_string())
    raise RuntimeError("MINTEQ database failed to load.")

print("MINTEQ V4 DATABASE LOADED SUCCESSFULLY.")

# ------------------------------------------------------------
# 4. DEFINING INITIAL WATER CHEMISTRY
# ------------------------------------------------------------

print("\n[4] INITIAL WATER CHEMISTRY")

solution_input = """
SOLUTION 1
    temp 25.0
    pH 7.0
    pe 4.0
    units mmol/L

    Cl       2.0
    S(6)     1.0
    C(4)     2.5
    Ca       1.5
    Mg       0.5
    P        0.05
    Cr(6)    0.1
    Cr(3)    0.0

SAVE SOLUTION 1
END
"""

# ------------------------------------------------------------
# 5. EXECUTING SOLUTION
# ------------------------------------------------------------

print("\n[5] RUNNING MULTI-ION SOLUTION")

try:
    pp.ip.run_string(solution_input)
    print("Solution calculation completed.")

except Exception as e:
    print("SOLUTION CALCULATION FAILED:")
    print(e)
    raise

# ------------------------------------------------------------
# 6. RETRIEVING SOLUTION
# ------------------------------------------------------------

print("\n[6] RETRIEVING SOLUTION")

solution = pp.get_solution(1)

print("Solution retrieved successfully.")


# ------------------------------------------------------------
# 7. CHEMICAL OUTPUT EXTRACTION
# ------------------------------------------------------------

print("\n[7] CHEMICAL OUTPUTS")

print(f"pH                  : {solution.pH:.5f}")
print(f"Ionic strength      : {solution.I:.7f} mol/L")

# Element totals are returned in mol.
# Will LAter Convert to mmol for reporting.

print("\nTOTAL COMPONENT AMOUNTS:")
print("-" * 55)

for component, amount_mol in solution.elements.items():
    print(
        f"{component:15s}: {amount_mol * 1000:.8f} mmol"
    )

# Total chromium across all oxidation states
print("\nTOTAL ELEMENTAL CHROMIUM:")

try:
    total_cr = solution.total_element("Cr", units="mmol")
    print(f"Total Cr: {total_cr:.8f} mmol")
except Exception as e:
    print("Chromium total unavailable:", e)

# Listing individual aqueous species and their amounts
print("\nCHROMIUM AQUEOUS SPECIES:")
print("-" * 55)

for species, amount in solution.species.items():
    if "Cr" in species:
        print(f"{species:20s}: {amount:.10f} mmol")

# Displaying master-species mapping
print("\nCHROMIUM MASTER-SPECIES MAPPING:")
print("-" * 55)

for master, species_list in solution.masters_species.items():
    if "Cr" in master:
        print(f"{master}: {species_list}")

print("\n" + "=" * 55)

# ------------------------------------------------------------
# 8. FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print(" MINTEQ V4 VERIFICATION REPORT")
print("=" * 65)

print("Database loading      : SUCCESS")
print("Solution execution    : SUCCESS")
print("Solution retrieval    : SUCCESS")

print("\nNOTE:")
print("This test verifies database compatibility and solution")
print("processing, not kinetic Cr(VI) reduction by zero-valent iron.")

print("=" * 65)
