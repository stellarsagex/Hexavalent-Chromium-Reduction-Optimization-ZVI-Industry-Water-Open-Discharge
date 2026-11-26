
import numpy as np
from scipy.ndimage import label, generate_binary_structure

# Load the original geometry
geometry = np.fromfile(
    "Step_3.3_Geometry_512.raw",
    dtype=np.int8
).reshape((512, 512, 512))

# Extract the same centered 64³ sample
start = (512 - 64) // 2
geometry = geometry[
    start:start+64,
    start:start+64,
    start:start+64
]

fluid = geometry == 1

print("=== GEOMETRY BOUNDARY DIAGNOSTIC ===")
print("Total fluid voxels:", np.sum(fluid))
print("Fluid porosity:", np.mean(fluid))

# Check fluid fractions on each face
for axis, name in enumerate(["X", "Y", "Z"]):
    inlet = np.take(fluid, 0, axis=axis)
    outlet = np.take(fluid, -1, axis=axis)

    print(f"\n{name}-direction:")
    print("  First face fluid fraction:", np.mean(inlet))
    print("  Opposite face fluid fraction:", np.mean(outlet))

# Identify connected fluid regions using 6-neighbour connectivity
structure = generate_binary_structure(3, 1)
labels, count = label(fluid, structure=structure)

print("\nConnected fluid components:", count)

# Check whether any component connects both x-faces
inlet_labels = set(np.unique(labels[0, :, :])) - {0}
outlet_labels = set(np.unique(labels[-1, :, :])) - {0}

spanning = inlet_labels.intersection(outlet_labels)

print("Components touching inlet:", len(inlet_labels))
print("Components touching outlet:", len(outlet_labels))
print("Components spanning x-direction:", len(spanning))
print("Through-flow connectivity:", bool(spanning))
