# 3D Rendering

import numpy as np
import pyvista as pv

# 1. Forcing PyVista to run off-screen at first
pv.OFF_SCREEN = True

# 2. Loading the raw 512^3 binary file
grid_size = 512
raw_filename = "Step_3.3_Geometry_512.raw"

print("Reading raw geometry file.")
grid_flat = np.fromfile(raw_filename, dtype=np.int8)
grid_3d = grid_flat.reshape((grid_size, grid_size, grid_size))

# 3. Extracting a corner subset box
sub_size = 120  # Slightly smaller to speed up processing
start_idx = 100 
sub_grid = grid_3d[start_idx:start_idx+sub_size, 
                   start_idx:start_idx+sub_size, 
                   start_idx:start_idx+sub_size]

print("Processing geometry into a 3D structural mesh.")
grid = pv.ImageData()
grid.dimensions = np.array(sub_grid.shape) + 1
grid.spacing = (0.6349, 0.6349, 0.6349) 

grid.cell_data["ZVI_Phase"] = sub_grid.flatten(order="F")

# Explicitly setting algorithm
solid_mesh = grid.threshold([0, 0], scalars="ZVI_Phase")
surface_mesh = solid_mesh.extract_surface(algorithm='dataset_surface').clean()

# 4. Initializing Plotter (off_screen=True variable added here)
print("Generating publication image in background.")
plotter = pv.Plotter(window_size=[1200, 1200], off_screen=True)

# Turning off shadows to avoid the specific shader calculation bug
plotter.add_mesh(
    surface_mesh,
    color="#cfd8dc",        
    ambient=0.4,            # Boosted ambient light to keep pores bright without shadows
    diffuse=0.6, 
    specular=0.1,             
    smooth_shading=True
)

# Labeling grids and dimensions
plotter.add_axes(xlabel='X (nm)', ylabel='Y (nm)', zlabel='Z (nm)', color='black')
plotter.show_grid(grid=False, location='outer', ticks='both', font_size=12, color='#555555')
plotter.background_color = "white"

plotter.camera_position = [
    (250, 250, 250),  
    (40, 40, 40),     
    (0, 0, 1)         
]

# 5. Rendering to save file
plotter.show(screenshot="Step_3.4_3D_Structure.png")
print("Success!")
