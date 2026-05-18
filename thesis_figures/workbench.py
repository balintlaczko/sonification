# %%
# imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from skimage import data
from skimage.transform import resize
import pydicom
import nibabel
from matplotlib.collections import PolyCollection
import os

# %%
# 1. Load a standard public domain benchmark image
# "The Astronaut" is Eileen Collins, standard in sci-kit image
original_img = data.astronaut() 

# 2. "Pixelate" it by resizing to a tiny grid (e.g., 8x8)
# This makes the "pixels" large enough to hold readable numbers
grid_size = (8, 8) 
tiny_img = resize(original_img, grid_size, anti_aliasing=True, preserve_range=True)
tiny_img = tiny_img.astype(int) # Convert to integers (0-255)

# Setup plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

def save_channel_grid(matrix, cmap, filename):
    plt.figure(figsize=(5, 5))
    # annot=True prints the numbers
    # cbar=False hides the color bar
    ax = sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, cbar=False, 
                     square=True, linewidths=1, linecolor='lightgray',
                     xticklabels=False, yticklabels=False)
    
    # Force rasterization on the colored background only
    for mesh in ax.collections:
        mesh.set_rasterized(True)
    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()

# --- Generate Subfigure A Data (Grayscale) ---
# Convert RGB to Grayscale using standard luminosity formula
gray_matrix = (tiny_img[:,:,0] * 0.299 + tiny_img[:,:,1] * 0.587 + tiny_img[:,:,2] * 0.114).astype(int)
save_channel_grid(gray_matrix, 'Greys', 'A_grayscale_matrix.svg')

# --- Generate Subfigure B Data (RGB Channels) ---
# Red Channel (Index 0)
save_channel_grid(tiny_img[:,:,0], 'Reds', 'B_red_channel.svg')

# Green Channel (Index 1)
save_channel_grid(tiny_img[:,:,1], 'Greens', 'B_green_channel.svg')

# Blue Channel (Index 2)
save_channel_grid(tiny_img[:,:,2], 'Blues', 'B_blue_channel.svg')

print(f"Generated 4 SVGs based on the Astronaut image resized to {grid_size}.")

# %%
# generate the dicom volumetric visualizations

# --- 1. Data Loading Helper ---
def load_dicom_volume(path_to_folder):
    """
    Loads a DICOM series from a folder into a 3D numpy array.
    Assumes standard 'one file per slice' structure.
    """
    files = []
    for fname in os.listdir(path_to_folder):
        if fname.endswith('.dcm'):
            files.append(pydicom.dcmread(os.path.join(path_to_folder, fname)))
    
    # Sort slices by Instance Number (standard Z-ordering)
    # If this fails, try sorting by SliceLocation
    try:
        files.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        print("Warning: 'InstanceNumber' not found, sorting by filename.")
        files.sort(key=lambda x: x.filename)

    # Stack them into a 3D array (Depth, Height, Width)
    volume = np.stack([s.pixel_array for s in files])
    return volume

# --- 2. The "Stacked Slices" Visualizer ---
def plot_volumetric_stack(volume, ax, num_slices_to_show=5, offset_step=0.05):
    """
    Draws a 3D volume as a stack of 2D images with fake depth.
    
    Args:
        volume: 3D numpy array (Depth, H, W) - can be 16-bit unnormalized
        ax: Matplotlib axis
        num_slices_to_show: How many slices to extract for the stack
        offset_step: How far apart to space the slices visually
    """
    # Select slices evenly spaced across the volume
    total_slices = volume.shape[0]
    indices = np.linspace(0, total_slices - 1, num_slices_to_show, dtype=int)
    
    # Convert to float and normalize to 0-1 range
    # Handles 16-bit (0-65535) or any unnormalized data
    volume = volume.astype(np.float32)
    vol_min = np.min(volume)
    vol_max = np.max(volume)
    
    # Avoid division by zero if volume is constant
    if vol_max - vol_min > 0:
        vol_norm = (volume - vol_min) / (vol_max - vol_min)
    else:
        vol_norm = np.zeros_like(volume)
    
    # Loop from back (index 0 in our visual) to front
    for i, slice_idx in enumerate(indices):
        slice_img = vol_norm[slice_idx]
        
        # Calculate offset (simulating depth)
        current_offset = (num_slices_to_show - 1 - i) * offset_step
        
        # We use imshow with the 'extent' parameter to shift the image
        extent = [current_offset, 1 + current_offset, current_offset, 1 + current_offset]
        
        # Add a border (rectangle) to define the slice edge
        rect = plt.Rectangle((current_offset, current_offset), 1, 1, 
                             transform=ax.transData, 
                             facecolor='none', edgecolor='blue', linewidth=1, zorder=i)
        ax.add_patch(rect)
        
        # Plot the image with vmin/vmax to ensure proper display range
        ax.imshow(slice_img, cmap='gray', extent=extent, zorder=i, alpha=0.95,
                  vmin=0, vmax=1)

    # Clean up axis
    ax.set_xlim(0, 1 + (num_slices_to_show * offset_step))
    ax.set_ylim(0, 1 + (num_slices_to_show * offset_step))
    ax.axis('off')

# --- 3. Generate Subfigure C (Single Stack) ---
def generate_subfigure_c(dicom_path):
    try:
        vol = load_dicom_volume(dicom_path)
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        print("Creating dummy noise volume for demonstration...")
        vol = np.random.rand(20, 128, 128) # Dummy data if path fails

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_volumetric_stack(vol, ax, num_slices_to_show=6, offset_step=0.1)
    plt.title("3D Volumetric Tensor (Space)", fontname="Times New Roman", fontsize=14)
    plt.tight_layout()
    plt.savefig('subfigure_C_dicom.svg')
    print("Saved subfigure_C_dicom.svg")
    plt.close()

# --- 4. Generate Subfigure D (Time Series) ---
def generate_subfigure_d(dicom_path):
    try:
        vol = load_dicom_volume(dicom_path)
    except:
        vol = np.random.rand(20, 128, 128)

    # Setup a figure with 3 columns (Time point 1, 2, 3)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    time_points = ["Time Point 1", "Time Point 2", "Time Point 3"]
    
    for idx, ax in enumerate(axes):
        # In a real scenario, you would load 'vol_t1', 'vol_t2', etc.
        # Here we just reuse 'vol' but hypothetically rotated or modified
        # to show it's dynamic. Let's just plot the same one for schema consistency.
        plot_volumetric_stack(vol, ax, num_slices_to_show=5, offset_step=0.08)
        
        # Add labels below
        ax.text(0.5, -0.1, time_points[idx], 
                transform=ax.transAxes, ha='center', 
                fontname="Times New Roman", fontsize=12)
        
        # Add arrows between plots (except the last one)
        if idx < 2:
            # Draw arrow in figure coordinates roughly between axes
            # This is tricky in code, easier to add in Inkscape.
            # But we can try adding a simple text arrow to the right of the plot
            ax.text(1.2, 0.5, "→", transform=ax.transAxes, fontsize=20, va='center')

    plt.suptitle("4D Image Tensor (Time-Series of Volumes)", 
                 fontname="Times New Roman", fontsize=16)
    plt.tight_layout()
    plt.savefig('subfigure_D_dicom.svg')
    print("Saved subfigure_D_dicom.svg")
    plt.close()

# %%
# --- RUN IT ---
# Replace this with your actual folder path
# e.g., path_to_dicom = "C:/Users/Data/BrainMRI/"
path_to_dicom = "./your_dicom_folder_here" 

# If you don't have data right now, the code handles the error 
# and generates a noise block so you can see the style.
generate_subfigure_c(path_to_dicom)
generate_subfigure_d(path_to_dicom)

# %%
# alternative with nifti

# %%
# load a nifti file
nifti_path = "/Users/balintl/Desktop/image_datasets/HCP-Development_3g_9Kxk/S1200_AverageT1w_restore.nii.gz"
nifti_img = nibabel.load(nifti_path)
nifti_data = nifti_img.get_fdata()
nifti_data.shape

# %%
fig, ax = plt.subplots(figsize=(4, 4))
plot_volumetric_stack(nifti_data, ax, num_slices_to_show=6, offset_step=0.1)
plt.title("3D Volumetric Tensor (Space)", fontname="Times New Roman", fontsize=14)
plt.tight_layout()
plt.savefig('subfigure_C_nifti.svg')
print("Saved subfigure_C_nifti.svg")
plt.close()

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# Define the data based on the conceptualization
data = {
    'Technique': [
        'Earcons', 
        'Auditory Icons', 
        'Audification', 
        'Standard PMSon', 
        'Cat. PMSon (Dataset)', 
        'Cat. PMSon (Interp)', 
        'Model-Based (MBS)'
    ],
    'Continuity (X)': [0, 1, 10, 8, 3, 6, 9],  # Discrete (0) <-> Continuous (10)
    'Semiotics (Y)':  [1, 9, 10, 2, 6, 4, 8],  # Symbolic (0) <-> Analogic (10)
    'Abstraction (Z)': [2, 2, 0, 3, 4, 5, 9]   # Direct (0) <-> Model (10)
}

df = pd.DataFrame(data)

# Create the plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
# Using different colors for clarity
scatter = ax.scatter(
    df['Continuity (X)'], 
    df['Semiotics (Y)'], 
    df['Abstraction (Z)'], 
    c=df['Abstraction (Z)'], 
    cmap='viridis', 
    s=150, 
    edgecolors='k',
    depthshade=False
)

# Labels and Titles
ax.set_xlabel('Continuity (Discrete $\leftrightarrow$ Continuous)', fontsize=12, labelpad=10)
ax.set_ylabel('Semiotics (Symbolic $\leftrightarrow$ Analogic)', fontsize=12, labelpad=10)
ax.set_zlabel('Mapping Order (Direct $\leftrightarrow$ Model)', fontsize=12, labelpad=10)
ax.set_title('Sonification Design Space', fontsize=16)

# Annotate each point
for i, txt in enumerate(df['Technique']):
    # Add a small offset to the text so it doesn't overlap the dot
    x_offset = 0
    y_offset = 0
    z_offset = 0.5
    
    # Custom offsets for better readability based on position
    if txt == 'Audification':
        x_offset = -1
        z_offset = 0.5
    elif txt == 'Earcons':
        z_offset = 0.8
    elif 'Cat.' in txt:
        z_offset = 0.6
        
    ax.text(
        df['Continuity (X)'][i] + x_offset, 
        df['Semiotics (Y)'][i] + y_offset, 
        df['Abstraction (Z)'][i] + z_offset, 
        txt, 
        fontsize=10,
        horizontalalignment='center'
    )

# Set axis limits to give some breathing room
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_zlim(-1, 11)

# Rotate the view
# elev: elevation angle (up/down) in degrees
# azim: azimuthal angle (left/right) in degrees
ax.view_init(elev=20, azim=50)

# Add grid lines for better depth perception
ax.grid(True)

# Save the plot
plt.tight_layout()
filename = 'sonification_design_space_3d.png'
plt.savefig(filename, dpi=300)
print(f"File saved as {filename}")
# %%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# --- GLOBAL SETTINGS FOR PUBLICATION QUALITY ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
except:
    print("LaTeX not found, using standard fonts.")
    plt.rcParams.update({
        "font.family": "Times New Roman",
    })

# Data
data = {
    'Technique': [
        'Earcons', 
        'Auditory Icons', 
        'Audification', 
        'Standard PMSon', 
        'Cat. PMSon (Dataset)', 
        'Cat. PMSon (Interp)', 
        'Model-Based (MBS)'
    ],
    'Continuity (X)': [1, 1, 10, 8, 3, 6, 9],
    'Semiotics (Y)':  [1, 9, 10, 2, 6, 4, 8],
    'Abstraction (Z)': [2, 2, 1, 3, 4, 5, 9]
}

df = pd.DataFrame(data)

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Axis limits
x_min, x_max = 0, 11
y_min, y_max = 0, 11
z_min, z_max = 0, 11
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Get colormap and normalize values
cmap = plt.cm.tab10
norm = plt.Normalize(vmin=df['Abstraction (Z)'].min(), vmax=df['Abstraction (Z)'].max())

# Colors
colors = df['Abstraction (Z)']
sc = ax.scatter(df['Continuity (X)'], df['Semiotics (Y)'], df['Abstraction (Z)'], 
                c=colors, cmap='tab10', s=100, depthshade=False, edgecolors='k', zorder=10)

# Add Drop Lines and Shadows
for i in range(len(df)):
    x, y, z = df['Continuity (X)'][i], df['Semiotics (Y)'][i], df['Abstraction (Z)'][i]
    
    # Get the color for this point from the colormap
    point_color = cmap(norm(z))
    
    # 1. Drop lines to the "Floor" (XY plane at Z_min)
    ax.plot([x, x], [y, y], [z_min, z], color=point_color, linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Drop lines to the "Back Wall" (XZ plane at Y_max)
    ax.plot([x, x], [y, y_max], [z, z], color=point_color, linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 3. Drop lines to the "Left Wall" (YZ plane at X_min)
    ax.plot([x_min, x], [y, y], [z, z], color=point_color, linestyle='--', linewidth=0.8, alpha=0.5)

    # 4. Draw "Shadows" (Projections) on the planes
    # Floor Shadow
    ax.scatter(x, y, z_min, color=point_color, alpha=0.3, s=20, marker='o')
    # Back Wall Shadow
    ax.scatter(x, y_max, z, color=point_color, alpha=0.3, s=20, marker='o')
    # Left Wall Shadow
    ax.scatter(x_min, y, z, color=point_color, alpha=0.3, s=20, marker='o')

    # 5. Label with Name AND Coordinates
    label_text = f"{df['Technique'][i]}\n({x}, {y}, {z})"
    
    # Dynamic offset to avoid overlapping the line
    z_offset = 0.6 if z < 10 else -1.5
    
    ax.text(x, y, z + z_offset, label_text, fontsize=9, ha='center', fontweight='bold')

# Labels
ax.set_xlabel('\nContinuity\n(Discrete $\leftrightarrow$ Continuous)', fontsize=11, linespacing=3.2)
ax.set_ylabel('\nSemiotics\n(Symbolic $\leftrightarrow$ Analogic)', fontsize=11, linespacing=3.2)
ax.set_zlabel('\nAbstraction\n(Direct $\leftrightarrow$ Model)', fontsize=11, linespacing=3.2)

# Title
ax.set_title('Sonification Design Space', fontsize=14)

# Adjust viewing angle for better perspective
ax.view_init(elev=20, azim=-75)

plt.tight_layout()
filename = 'sonification_design_space_3d_lines.png'
plt.savefig(filename, dpi=300)
print(f"Saved {filename}")

# %%
# the design space of sonification figure for thesis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# --- GLOBAL SETTINGS FOR PUBLICATION QUALITY ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 18,
        "font.size": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
except:
    print("LaTeX not found, using standard fonts.")
    plt.rcParams.update({
        "font.family": "Times New Roman",
    })

# Data
data = {
    'Technique': [
        'Earcons', 
        'Auditory Icons', 
        'Audification', 
        'Standard PMSon', 
        'Cat. PMSon\n(Dataset)', 
        'Cat. PMSon\n(Interp)', 
        'Model-Based'
    ],
    'Continuity (X)': [1, 1, 10, 8, 3, 6, 9],
    'Semiotics (Y)':  [1, 9, 10, 2, 6, 4, 8],
    'Abstraction (Z)': [2, 2, 1, 3, 4, 5, 9],
    # # Per-technique "spread" radii (rx, ry, rz) — how fuzzy each category is
    # 'Spread': [
    #     (1.5, 1.5, 1.5),   # Earcons: fairly tight
    #     (1.5, 2.0, 1.5),   # Auditory Icons: broader on semiotics
    #     (2.0, 1.5, 1.0),   # Audification: broad on continuity, tight on abstraction
    #     (2.0, 2.0, 2.0),   # Standard PMSon: medium spread
    #     (2.0, 2.0, 2.0),   # Cat. PMSon (Dataset)
    #     (2.0, 2.0, 2.0),   # Cat. PMSon (Interp)
    #     (2.0, 2.0, 2.5),   # Model-Based: broad on abstraction
    # ]
}

df = pd.DataFrame(data)

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Axis limits
x_min, x_max = 0, 11
y_min, y_max = 0, 11
z_min, z_max = 0, 11
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Function to map coordinates to RGB (10% to 90% range)
def coords_to_rgb(x, y, z, data_min=1, data_max=10, rgb_min=0.1, rgb_max=0.85):
    """Map X->R, Y->G, Z->B with values in 10%-90% range"""
    def normalize(val):
        # Normalize to 0-1, then scale to rgb_min-rgb_max
        norm = (val - data_min) / (data_max - data_min)
        return rgb_min + norm * (rgb_max - rgb_min)
    
    r = normalize(x)
    g = normalize(y)
    b = normalize(z)
    return (r, g, b)

# Generate colors for each point
point_colors = [coords_to_rgb(df['Continuity (X)'][i], 
                               df['Semiotics (Y)'][i], 
                               df['Abstraction (Z)'][i]) 
                for i in range(len(df))]


# # --- Draw translucent ellipsoid "clouds" for each technique ---
# def draw_ellipsoid(ax, center, radii, color, alpha=0.08, resolution=20):
#     """Draw a translucent ellipsoid surface."""
#     cx, cy, cz = center
#     rx, ry, rz = radii
    
#     u = np.linspace(0, 2 * np.pi, resolution)
#     v = np.linspace(0, np.pi, resolution)
    
#     x = cx + rx * np.outer(np.cos(u), np.sin(v))
#     y = cy + ry * np.outer(np.sin(u), np.sin(v))
#     z = cz + rz * np.outer(np.ones_like(u), np.cos(v))
    
#     ax.plot_surface(x, y, z, color=color, alpha=alpha, 
#                     rstride=1, cstride=1, linewidth=0, 
#                     antialiased=True, shade=False, zorder=1)

# # Draw clouds FIRST (behind everything)
# for i in range(len(df)):
#     center = (df['Continuity (X)'][i], df['Semiotics (Y)'][i], df['Abstraction (Z)'][i])
#     radii = df['Spread'][i]
#     draw_ellipsoid(ax, center, radii, color=point_colors[i], alpha=0.10, resolution=20)


# Scatter plot with RGB colors
sc = ax.scatter(df['Continuity (X)'], df['Semiotics (Y)'], df['Abstraction (Z)'], 
                c=point_colors, s=100, depthshade=False, edgecolors='k', zorder=10)

# Add Drop Lines and Shadows
for i in range(len(df)):
    x, y, z = df['Continuity (X)'][i], df['Semiotics (Y)'][i], df['Abstraction (Z)'][i]
    
    # Get the RGB color for this point
    point_color = point_colors[i]
    
    # 1. Drop lines to the "Floor" (XY plane at Z_min)
    ax.plot([x, x], [y, y], [z_min, z], color=point_color, linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Drop lines to the "Back Wall" (XZ plane at Y_max)
    ax.plot([x, x], [y, y_max], [z, z], color=point_color, linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 3. Drop lines to the "Left Wall" (YZ plane at X_min)
    ax.plot([x_min, x], [y, y], [z, z], color=point_color, linestyle='--', linewidth=0.8, alpha=0.5)

    # 4. Draw "Shadows" (Projections) on the planes
    # Floor Shadow
    ax.scatter(x, y, z_min, color=point_color, alpha=0.3, s=20, marker='o')
    # Back Wall Shadow
    ax.scatter(x, y_max, z, color=point_color, alpha=0.3, s=20, marker='o')
    # Left Wall Shadow
    ax.scatter(x_min, y, z, color=point_color, alpha=0.3, s=20, marker='o')

    # 5. Label with Name AND Coordinates
    # label_text = f"{df['Technique'][i]}\n({x}, {y}, {z})"
    label_text = f"{df['Technique'][i]}"
    
    # Dynamic offset to avoid overlapping the line
    z_offset = 0.6 if z < 10 else -1.5
    
    ax.text(x, y, z + z_offset, label_text, fontsize=13, ha='center')

# Labels
ax.set_xlabel('\nContinuity\n(Discrete $\leftrightarrow$ Continuous)', linespacing=2, labelpad=10)
ax.set_ylabel('\nSemiotics\n(Symbolic $\leftrightarrow$ Analogic)', linespacing=2, labelpad=10)
ax.set_zlabel('\nAbstraction\n(Direct $\leftrightarrow$ Model)', linespacing=2, labelpad=10)

# Title
# ax.set_title('Sonification Design Space: 3D Projection\n(Color = Position: X→R, Y→G, Z→B)', fontsize=14)

# Adjust viewing angle for better perspective
ax.view_init(elev=20, azim=-80)

plt.tight_layout()
filename = 'sonification_design_space_3d_rgb.pdf'
plt.savefig(filename, dpi=300, format='pdf')
print(f"Saved {filename}")

# %%
# McAdams timbre space figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# --- SETUP ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
# Use a gray grid for that academic look
plt.rcParams['grid.color'] = '#d3d3d3' 
plt.rcParams['grid.alpha'] = 0.5

# --- GLOBAL SETTINGS FOR PUBLICATION QUALITY ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 18,
        "font.size": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
except:
    print("LaTeX not found, using standard fonts.")
    plt.rcParams.update({
        "font.family": "Times New Roman",
    })

# --- DATA: McAdams (1995) 3D Timbre Space ---
# Dim 1: Log Rise Time (Negative = Slower Attack, Positive = Faster Attack)
# Dim 2: Spectral Centroid (Negative = Darker, Positive = Brighter)
# Dim 3: Spectral Flux (Variation over time)

data_mcadams = {
    'hrn':   [-3.3, 1.3,  -1.5], # French Horn
    'tpt':   [-2.6,  -1.9,  0.4], # Trumpet
    'tbn':   [-2.4, 1.7, - 1.2], # Trombone
    'hrp':   [3.0, 1.7, -0.4], # Harp
    '"tpr"':   [-0.1, -2.7, 0.1], # "Trumpar"
    '"ols"':   [3.0, 1.7, 0.7], # "Oboleste"
    'vbn':   [3.8, 1.8, 1.3], # Vibraphone
    '"sno"':   [-1.4, -0.9, 1.6], # "Striano"
    'hcd':   [3.6, -2.8, 0.5], # Harpsichord
    'ehn':   [-1.9, -1.5, -1.9], # English Horn
    'bsn':   [-2.4, -1.8, -2.0], # Bassoon
    'cnt':   [-2.4, 1.9, 0.5], # Clarinet
    '"vbn"':   [0.7, 2.3, -1.6], # "Vibrone"
    '"obc"':   [2.5, -2.3, -2.7], # "Obochord"
    'gtr':   [2.9, 0.2, 2.4], # Guitar
    'stg':   [-2.4, -1.4, 1.4], # String
    'pno':   [1.3, 1.3, 0.2], # Piano
    '"gnt"':   [-1.8, 1.2, 2.0], # "Guitarnet"
}

names = list(data_mcadams.keys())
coords = np.array(list(data_mcadams.values()))

df = pd.DataFrame(data_mcadams).T
df.columns = ['Dim. 1', 'Dim. 2', 'Dim. 3']
df.reset_index(inplace=True)

df.head()

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Axis limits
x_min, x_max = -3, 3
y_min, y_max = -3, 3
z_min, z_max = -4, 4
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Function to map coordinates to RGB (10% to 90% range)
def coords_to_rgb(x, y, z, data_min=-4, data_max=4, rgb_min=0.1, rgb_max=0.85):
    """Map X->R, Y->G, Z->B with values in 10%-90% range"""
    def normalize(val):
        # Normalize to 0-1, then scale to rgb_min-rgb_max
        norm = (val - data_min) / (data_max - data_min)
        return rgb_min + norm * (rgb_max - rgb_min)
    
    r = normalize(x)
    g = normalize(y)
    b = normalize(z)
    return (r, g, b)

# Generate colors for each point
point_colors = [coords_to_rgb(df['Dim. 1'][i], 
                               df['Dim. 2'][i], 
                               df['Dim. 3'][i]) 
                for i in range(len(df))]

# Scatter plot with RGB colors
sc = ax.scatter(df['Dim. 2'], df['Dim. 3'], df['Dim. 1'], 
                c=point_colors, s=100, depthshade=False, edgecolors='k', zorder=10)

# Add Drop Lines and Shadows
for i in range(len(df)):
    x, y, z = df['Dim. 2'][i], df['Dim. 3'][i], df['Dim. 1'][i]
    
    # Get the RGB color for this point
    point_color = point_colors[i]
    
    # 1. Drop lines to the "Floor" (XY plane at Z_min)
    ax.plot([x, x], [y, y], [z_min, z], color=point_color, linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Drop lines to the "Back Wall" (XZ plane at Y_max)
    ax.plot([x, x], [y, y_max], [z, z], color=point_color, linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 3. Drop lines to the "Left Wall" (YZ plane at X_min)
    ax.plot([x_min, x], [y, y], [z, z], color=point_color, linestyle='--', linewidth=0.8, alpha=0.5)

    # 4. Draw "Shadows" (Projections) on the planes
    # Floor Shadow
    ax.scatter(x, y, z_min, color=point_color, alpha=0.3, s=20, marker='o')
    # Back Wall Shadow
    ax.scatter(x, y_max, z, color=point_color, alpha=0.3, s=20, marker='o')
    # Left Wall Shadow
    ax.scatter(x_min, y, z, color=point_color, alpha=0.3, s=20, marker='o')

    # 5. Label with Name AND Coordinates
    label_text = f"{df['index'][i]}"
    
    # Dynamic offset to avoid overlapping the line
    z_offset = 0.25 if z < 10 else -1.5
    
    ax.text(x, y, z + z_offset, label_text, ha='center')#, fontweight='bold')

# Labels
ax.set_xlabel('\nDimension 2\n(spectral centroid)', linespacing=2, labelpad=15)
ax.set_ylabel('\nDimension 3\n(spectral flux)', linespacing=2, labelpad=15)
ax.set_zlabel('\nDimension 1\n(rise time)', linespacing=2, labelpad=10)

# Title
# ax.set_title('Sonification Design Space: 3D Projection\n(Color = Position: X→R, Y→G, Z→B)', fontsize=14)

# Adjust viewing angle for better perspective
ax.view_init(elev=18, azim=-50)

plt.tight_layout()
filename = 'mcadams_timbre_space.pdf'
plt.savefig(filename, dpi=300, format='pdf')
print(f"Saved {filename}")

# %%
# ---- The Control Loop figure ----

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_control_loop():
    # Setup the plot
    fig, ax = plt.figure(figsize=(8, 8)), plt.gca()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    # --- Configuration ---
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5, color='#444444')
    
    # Coordinates for the 4 nodes (Top, Right, Bottom, Left)
    # We use a diamond layout: (0,1), (1,0), (0,-1), (-1,0)
    pos = {
        'top': (0, 1.1),
        'right': (1.2, 0),
        'bottom': (0, -1.1),
        'left': (-1.2, 0)
    }

    # --- 1. Draw Boxes ---
    
    # Top: Inter/Action
    ax.text(*pos['top'], "Inter/Action", ha='center', va='center', fontsize=12, fontweight='bold', bbox=box_props)

    # Left: Energy Transfer
    ax.text(*pos['left'], "Energy Transfer\n/\nMapping", ha='center', va='center', fontsize=12, bbox=box_props)

    # Right: Perception
    ax.text(*pos['right'], "Perception", ha='center', va='center', fontsize=12, bbox=box_props)

    # Bottom: Complex Label
    # We split this into a wider box with two sections
    bottom_text = (
        "Change in Light/Sound\n"
        "---------------------------\n"
        "Software (Audio/Video)\n"
        "⬇\n"
        "Hardware (Speaker/Screen)"
    )
    ax.text(*pos['bottom'], bottom_text, ha='center', va='center', fontsize=10, bbox=box_props)

    # --- 2. Draw Arrows (The Cycle) ---
    # Top -> Left
    ax.annotate("", xy=pos['left'], xytext=pos['top'], arrowprops=arrow_props)
    # Left -> Bottom
    ax.annotate("", xy=pos['bottom'], xytext=pos['left'], arrowprops=arrow_props)
    # Bottom -> Right
    ax.annotate("", xy=pos['right'], xytext=pos['bottom'], arrowprops=arrow_props)
    # Right -> Top
    ax.annotate("", xy=pos['top'], xytext=pos['right'], arrowprops=arrow_props)

    # --- 3. The "Learning" Bridge ---
    # Line connecting Left and Right
    ax.plot([pos['left'][0]+0.3, pos['right'][0]-0.3], [0, 0], color='red', linestyle='--', linewidth=1.5, zorder=0)
    
    # Label for Learning
    ax.text(0, 0.05, "Learning", ha='center', va='bottom', fontsize=11, color='red', fontweight='bold', backgroundcolor='white')
    
    plt.title("Figure 1: The Control Loop", y=0.02, fontsize=10, style='italic')
    plt.tight_layout()
    plt.show()

draw_control_loop()
# %%

# --- The Spiral of Embodiment ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# --- GLOBAL SETTINGS FOR PUBLICATION QUALITY ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
except:
    print("LaTeX not found, using standard fonts.")
    plt.rcParams.update({
        "font.family": "Times New Roman",
    })

def draw_figure_1():
    """
    Generates Figure 1: The Control Loop (2D)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis('off')

    # Properties
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5, color='#444444')
    
    # Coordinates (Diamond layout)
    pos = {
        'top': (0, 1.2),     # Action
        'right': (1.3, 0),   # Perception
        'bottom': (0, -1.2), # Change
        'left': (-1.3, 0)    # Transfer
    }

    # --- 1. Draw Text Boxes with Shorthand Labels ---
    
    # Top: Inter/Action [A]
    ax.text(pos['top'][0], pos['top'][1], r"\textbf{[A] Inter/Action}", 
            ha='center', va='center', fontsize=12, bbox=box_props, zorder=10)

    # Left: Energy Transfer [T]
    ax.text(pos['left'][0], pos['left'][1], r"\textbf{[T] Energy Transfer}" + "\n" + r"\textbf{/ Mapping}", 
            ha='center', va='center', fontsize=12, bbox=box_props, zorder=10)

    # Right: Perception [P]
    ax.text(pos['right'][0], pos['right'][1], r"\textbf{[P] Perception}", 
            ha='center', va='center', fontsize=12, bbox=box_props, zorder=10)

    # Bottom: Change [C]
    bottom_text = (
        r"\textbf{[C] Change}" + "\n" +
        r"in Light/Sound" + "\n" +
        r"-------------------" + "\n" +
        r"Software (AV)" + "\n" +
        r"$\downarrow$" + "\n" +
        r"Hardware (I/O)"
    )
    ax.text(pos['bottom'][0], pos['bottom'][1], bottom_text, 
            ha='center', va='center', fontsize=10, bbox=box_props, zorder=10)

    # --- 2. Draw Arrows (The Cycle) ---
    ax.annotate("", xy=pos['left'], xytext=pos['top'], arrowprops=arrow_props)
    ax.annotate("", xy=pos['bottom'], xytext=pos['left'], arrowprops=arrow_props)
    ax.annotate("", xy=pos['right'], xytext=pos['bottom'], arrowprops=arrow_props)
    ax.annotate("", xy=pos['top'], xytext=pos['right'], arrowprops=arrow_props)

    # --- 3. The "Learning" Bridge ---
    # Connect Left [T] and Right [P]
    # We draw the line slightly shorter so it doesn't overlap the text
    ax.plot([pos['left'][0]+0.4, pos['right'][0]-0.4], [0, 0], 
            color='red', linestyle='--', linewidth=2.0, zorder=5)
    
    ax.text(0, 0.1, r"\textbf{Learning}", ha='center', va='bottom', fontsize=11, color='red', fontweight='bold', backgroundcolor='white')
    
    # Save
    plt.tight_layout()
    plt.savefig("figure_1_control_loop.pdf", bbox_inches='tight')
    print("Figure 1 saved as 'figure_1_control_loop.pdf'")
    plt.close()

def get_spiral_point(t, x_max, t_max, r_max, r_min):
    """Calculates x, y, z for a given parameter t."""
    frac = t / t_max
    x = frac * x_max
    r = r_max - (frac * (r_max - r_min))
    y = r * np.cos(t)
    z = r * np.sin(t)
    return x, y, z

def get_label_alpha(t, t_max, alpha_start=1.0, alpha_end=0.0):
    """Calculate alpha (opacity) for labels that fades along the spiral."""
    frac = t / t_max
    return alpha_start - frac * (alpha_start - alpha_end)

def draw_figure_2():
    """
    Generates Figure 2: The Spiral of Embodiment (3D)
    Using exact coordinates for labels and connecting lines.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')

    # --- Parameters ---
    loops = 12                # Number of full turns
    x_max = 40               # Stretched length (Time/Skill axis)
    points_per_loop = 100
    total_points = loops * points_per_loop
    t_max = loops * 2 * np.pi
    r_max = 4.0
    r_min = 0.5

    # --- 1. Draw the Spiral Trace ---
    # Start from the first A node (at π/2) instead of the first P node (at 0)
    t_start = np.pi / 2
    t_vals = np.linspace(t_start, t_max, total_points)
    x_trace = []
    y_trace = []
    z_trace = []
    for t in t_vals:
        px, py, pz = get_spiral_point(t, x_max, t_max, r_max, r_min)
        x_trace.append(px)
        y_trace.append(py)
        z_trace.append(pz)
    
    ax.plot(x_trace, y_trace, z_trace, color='black', linewidth=1.5, alpha=0.9, label='Sensorimotor Loop')

    # --- 2. Place Labels and Draw Rungs (The Ladder) ---
    
    # We use a high alpha (1.0) for the box background to hide the line start/end points
    box_props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=1.0)
    text_kwargs = dict(ha='center', va='center', fontsize=8, bbox=box_props, zorder=100)

    # Lists to store the coordinates of T and P boxes for the stems
    T_coords = []
    P_coords = []

    # We iterate through loops to identify key points: A (top), T (left), C (bottom), P (right/end)
    
    for i in range(loops):
        # Define exact angles for this loop
        t_A       = i * 2 * np.pi + np.pi/2
        t_T       = i * 2 * np.pi + np.pi    # Transfer
        t_C       = i * 2 * np.pi + 3*np.pi/2
        t_P_end   = (i + 1) * 2 * np.pi      # Perception (next cycle start)

        # Calculate Coordinates
        
        # P (Perception) - Right (We draw P at the END of the cycle to connect T -> P_end)
        xp, yp, zp = get_spiral_point(t_P_end, x_max, t_max, r_max, r_min)
        if t_P_end <= t_max + 0.001:
            alpha_p = get_label_alpha(t_P_end, t_max)
            box_props_p = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=alpha_p)
            ax.text(xp, yp, zp, r"\textbf{P}", ha='center', va='center', fontsize=10, 
                    bbox=box_props_p, zorder=100, alpha=alpha_p)
            P_coords.append((xp, yp, zp))

        # T (Transfer) - Left
        xt, yt, zt = get_spiral_point(t_T, x_max, t_max, r_max, r_min)
        if t_T <= t_max:
            alpha_t = get_label_alpha(t_T, t_max)
            box_props_t = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=alpha_t)
            ax.text(xt, yt, zt, r"\textbf{T}", ha='center', va='center', fontsize=10, 
                    bbox=box_props_t, zorder=100, alpha=alpha_t)
            T_coords.append((xt, yt, zt))

            # --- DRAW LADDER RUNG ---
            # Connects [T] of this loop to [P] of this loop (the one at the end)
            # This represents the "Learning" bridge: Transfer -> Perception (skipping Change)
            # The line goes exactly from coordinate (xt, yt, zt) to (xp, yp, zp)
            ax.plot([xt, xp], [yt, yp], [zt, zp], 
                    color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=50)

        # A (Action) - Top
        xa, ya, za = get_spiral_point(t_A, x_max, t_max, r_max, r_min)
        if t_A <= t_max:
            alpha_a = get_label_alpha(t_A, t_max)
            box_props_a = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=alpha_a)
            ax.text(xa, ya, za, r"\textbf{A}", ha='center', va='center', fontsize=10, 
                    bbox=box_props_a, zorder=100, alpha=alpha_a)

        # C (Change) - Bottom
        xc, yc, zc = get_spiral_point(t_C, x_max, t_max, r_max, r_min)
        if t_C <= t_max:
            alpha_c = get_label_alpha(t_C, t_max)
            box_props_c = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=alpha_c)
            ax.text(xc, yc, zc, r"\textbf{C}", ha='center', va='center', fontsize=10, 
                    bbox=box_props_c, zorder=100, alpha=alpha_c)
            
    # --- 3. Draw Ladder Stems ---
    # Connect all P nodes
    if P_coords:
        px_list, py_list, pz_list = zip(*P_coords)
        ax.plot(px_list, py_list, pz_list, color='red', linestyle='-', linewidth=2.0, alpha=0.8, zorder=40)

    # Connect all T nodes
    if T_coords:
        tx_list, ty_list, tz_list = zip(*T_coords)
        ax.plot(tx_list, ty_list, tz_list, color='red', linestyle='-', linewidth=2.0, alpha=0.8, zorder=40)

    # --- 3b. Add "Ladder of Integration" label with arrow ---
    # Point to the middle of the ladder (around the midpoint of T coords)
    if T_coords and len(T_coords) > 2:
        mid_idx = len(T_coords) // 2 # Point to early-mid section of ladder
        ladder_target = T_coords[mid_idx]
        # Text position offset from the ladder
        ladder_label_x = ladder_target[0] - 5
        ladder_label_y = ladder_target[1] - 4
        ladder_label_z = ladder_target[2] - 2
        ax.text(ladder_label_x, ladder_label_y, ladder_label_z, 
                r"The Ladder of Integration", fontsize=12, color='red', 
                ha='center', va='center', zorder=110)
        # Draw a simple line as arrow (3D arrows are tricky)
        ax.plot([ladder_label_x + 2.2, ladder_target[0] + 1], 
                [ladder_label_y + 2.5, ladder_target[1]], 
                [ladder_label_z - 0.15, ladder_target[2]], 
                color='red', linewidth=1.0, alpha=0.8, zorder=105)


    # --- 4. The "Embodied Knowledge" Volume (Transparent Cone) ---
    X_mesh = np.linspace(0, x_max, 50)
    Theta_mesh = np.linspace(0, 2*np.pi, 40)
    X_grid, Theta_grid = np.meshgrid(X_mesh, Theta_mesh)
    
    R_grid = r_max - (X_grid / x_max) * (r_max - r_min)
    Y_grid = R_grid * np.cos(Theta_grid)
    Z_grid = R_grid * np.sin(Theta_grid)
    
    # Plot transparent cone surface
    ax.plot_surface(X_grid, Y_grid, Z_grid, color='cyan', alpha=0.08, rstride=2, cstride=2, linewidth=0, antialiased=False)

    # --- 5. The Outer Cylinder (Wireframe for reference) ---
    Xc_grid, Thetac_grid = np.meshgrid(np.linspace(0, x_max, 10), np.linspace(0, 2*np.pi, 20))
    Yc_grid = r_max * np.cos(Thetac_grid)
    Zc_grid = r_max * np.sin(Thetac_grid)
    ax.plot_wireframe(Xc_grid, Yc_grid, Zc_grid, color='gray', alpha=0.1, linewidth=0.5)

    # --- 6. Embodied Knowledge: Annular rings between cone and cylinder ---
    # Draw filled annular discs at intervals to show the space between cone and cylinder
    num_rings = 40
    ring_positions = np.linspace(0, x_max, num_rings + 1)[1:]  # Skip position 0
    theta_ring = np.linspace(0, 2*np.pi, 60)
    
    embodied_patch = None  # For legend
    for x_pos in ring_positions:
        # Calculate cone radius at this x position
        r_cone = r_max - (x_pos / x_max) * (r_max - r_min)
        
        # Create annular ring from r_cone to r_max
        r_ring = np.linspace(r_cone, r_max, 10)
        R_ring, Theta_ring = np.meshgrid(r_ring, theta_ring)
        
        Y_ring = R_ring * np.cos(Theta_ring)
        Z_ring = R_ring * np.sin(Theta_ring)
        X_ring = np.full_like(Y_ring, x_pos)
        
        # Plot the annular disc with green color and hatching effect via alpha
        surf = ax.plot_surface(X_ring, Y_ring, Z_ring, color='purple', alpha=0.05, 
                               rstride=1, cstride=1, linewidth=0, antialiased=True, zorder=5)
        if embodied_patch is None:
            embodied_patch = surf
    
    # Create a proxy artist for the legend
    from matplotlib.patches import Patch
    legend_patch = Patch(facecolor='purple', alpha=0.4, label='Embodied Knowledge')
    # Legend position: (x, y) where (1, 1) is top-right, (1, 0) is bottom-right
    legend_x = 1.0
    legend_y = 0.77  # Adjust this to move legend up/down (0=bottom, 1=top)
    ax.legend(handles=[legend_patch], loc='upper right', bbox_to_anchor=(legend_x, legend_y), framealpha=0.9)

    # --- Labels & Aesthetics ---
    ax.set_xlabel('Skill Acquisition (Time)')
    ax.set_ylabel('Transfer $\leftrightarrow$ Perception', labelpad=-8)
    ax.set_zlabel('Change $\leftrightarrow$ Action', labelpad=-8)

    ax.set_xlim(0, x_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)

    # Force the aspect ratio to stretch horizontally
    ax.set_box_aspect([3, 1, 1])  # This makes x-axis much longer

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    ax.view_init(elev=20, azim=-60)

    plt.subplots_adjust(left=0, right=0.9, bottom=0, top=1)
    
    plt.savefig("figure_2_spiral_embodiment.pdf", dpi=300, format='pdf', pad_inches=0) #, bbox_inches='tight')
    print("Figure 2 saved as 'figure_2_spiral_embodiment.pdf'")

# draw_figure_1()
draw_figure_2()

# %%
# fractal of spirals

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- PUBLICATION SETTINGS ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
    })
except:
    pass

def draw_spiral_branch(ax, origin, direction, length, radius_start, radius_end, level, max_levels):
    """
    Recursive function to draw a spiral and spawn children at its end.
    
    origin: (x,y,z) start point
    direction: (x,y,z) vector pointing in the growth direction
    length: length of this spiral segment
    radius_start: radius at the start of the spiral
    radius_end: radius at the wide open mouth
    """
    
    # 1. Coordinate System Construction
    # We need to build a local coordinate system (u, v, w) where w is the direction
    w = direction / np.linalg.norm(direction)
    
    # Arbitrary vector to find u (orthogonal to w)
    if np.abs(w[2]) < 0.9:
        arb = np.array([0, 0, 1])
    else:
        arb = np.array([0, 1, 0])
        
    u = np.cross(arb, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    
    # 2. Generate the Spiral Points
    loops = 6
    points = 300
    t = np.linspace(0, loops * 2 * np.pi, points)
    
    # Linearly interpolate radius (cone shape)
    # Parametric variable s goes from 0 to 1 along the length
    s = np.linspace(0, 1, points)
    current_radius = radius_start + s * (radius_end - radius_start)
    
    # Spiral geometry in local coords
    # x_local is along the direction w
    local_w = s * length
    local_u = current_radius * np.cos(t)
    local_v = current_radius * np.sin(t)
    
    # Transform to Global Coords
    # P = origin + (local_u * u) + (local_v * v) + (local_w * w)
    xs = origin[0] + local_u * u[0] + local_v * v[0] + local_w * w[0]
    ys = origin[1] + local_u * u[1] + local_v * v[1] + local_w * w[1]
    zs = origin[2] + local_u * u[2] + local_v * v[2] + local_w * w[2]
    
    # 3. Plot this branch
    # Color gets lighter as we go deeper (more abstract)
    color_val = level / max_levels
    color = cm.magma(0.2 + 0.6 * color_val) # Dark to Light
    
    alpha = 0.9
    lw = 1.5 - (level * 0.3)
    ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=lw)
    
    # 4. Recursion: Spawn Children
    if level < max_levels:
        # Number of children branches
        num_children = 4
        
        # Calculate the center of the "mouth" (end of current spiral)
        end_center = origin + w * length

        # --- TWEAKABLE PARAMETERS ---
        flare_factor = 0.3      # How much children flare outward (0 = straight, 1 = very flared)
        inset_factor = 0.8      # How close to center (0 = on rim, 1 = at center of mouth)
        push_into_mouth = 0.2   # How far to push children back into parent's mouth (0 = at mouth, 1 = full length back)

        for i in range(num_children):
            # Angle around the mouth circle
            angle_offset = (i / num_children) * 2 * np.pi
            
            # Position on the rim (adjusted by inset_factor)
            effective_radius = radius_end * (1.0 - inset_factor)  # Shrinks toward center
            rim_u = effective_radius * np.cos(angle_offset)
            rim_v = effective_radius * np.sin(angle_offset)
            
            # Start from mouth center, then offset radially AND push back into mouth
            child_origin = end_center + (rim_u * u) + (rim_v * v) - (w * length * push_into_mouth)
            
            # Direction for the child
            # It should continue forward (w) but flare out slightly
            if effective_radius > 0.001:  # Avoid division by zero
                radial_vector = (rim_u * u) + (rim_v * v)
                radial_vector = radial_vector / np.linalg.norm(radial_vector)
                child_direction = w + (radial_vector * flare_factor)
            else:
                child_direction = w  # If at center, just go straight
            child_direction = child_direction / np.linalg.norm(child_direction)
            
            # Parameters for next level
            scale_factor = 0.5
            new_length = length * scale_factor
            new_r_start = radius_end * 0.1 # Start small (tip)
            new_r_end = radius_end * scale_factor # Grow to proportional mouth
            
            draw_spiral_branch(ax, child_origin, child_direction, 
                               new_length, new_r_start, new_r_end, 
                               level + 1, max_levels)

def generate_spiral_fractal():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setup
    origin = np.array([0, 0, 0])
    direction = np.array([1, 0, 0]) # Moving along X axis
    length = 80
    r_start = 0.1 # Tip of the foundational skill
    r_end = 10.0   # Mouth of the foundational skill
    max_levels = 5 # How deep the fractal goes
    
    draw_spiral_branch(ax, origin, direction, length, r_start, r_end, 0, max_levels)
    
    # Camera
    # ax.view_init(elev=25, azim=30)
    ax.view_init(elev=0, azim=0)
    ax.view_init(elev=20, azim=10)
    ax.set_axis_off() # Remove all axes/grid/text
    
    # Make aspect ratio equal-ish
    # Matplotlib 3D aspect ratio is tricky, this is a hack
    ax.set_box_aspect([1, 1, 1]) 
    
    plt.tight_layout()
    plt.savefig("figure_3_spiral_tree.png", bbox_inches='tight', dpi=500)
    plt.show()

generate_spiral_fractal()

# %%
# figure to illustrate latent dim shuffling

import numpy as np
import matplotlib.pyplot as plt

def permute_dims(x):
    """
    Permutes dimensions independently across the batch.
    
    Args:
        x: numpy array of shape (batch, z_dim)
    Returns:
        y: numpy array of shape (batch, z_dim)
    """
    b, z_dim = x.shape
    y = np.zeros_like(x)
    
    # Iterate over dimensions and shuffle each column independently
    for i in range(z_dim):
        # np.random.permutation returns a shuffled copy or range
        idx = np.random.permutation(b)
        y[:, i] = x[idx, i]
        
    return y

def generate_data(n_samples=1000):
    """Generates the three specific distributions."""
    
    # Case 1: Isotropic Gaussian (Standard Normal)
    # No correlation between dimensions.
    data_gaussian = np.random.randn(n_samples, 2)
    
    # Case 2: Slightly Correlated
    # We create a covariance matrix with some off-diagonal energy.
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # 0.8 correlation
    data_correlated = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Case 3: Perfectly Correlated
    # y = x. We sample x from normal, and set y = x.
    x_vals = np.random.randn(n_samples, 1)
    data_perfect = np.hstack([x_vals, x_vals])
    
    return data_gaussian, data_correlated, data_perfect

def plot_distributions():
    # Setup
    np.random.seed(42) # For reproducibility
    n_samples = 1000
    
    # Generate Data
    d1, d2, d3 = generate_data(n_samples)
    
    # Apply Permutation (FactorVAE shuffling)
    d1_shuffled = permute_dims(d1)
    d2_shuffled = permute_dims(d2)
    d3_shuffled = permute_dims(d3)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Plot params
    scatter_kwargs = {'alpha': 0.4, 's': 10, 'edgecolor': 'none'}
    titles = ["(1) Independent Gaussian", "(2) Slightly Correlated", "(3) Perfectly Correlated"]
    
    # Row 1: Originals
    data_list = [d1, d2, d3]
    for i, ax in enumerate(axes[0]):
        ax.scatter(data_list[i][:, 0], data_list[i][:, 1], c='tab:blue', **scatter_kwargs)
        ax.set_title(f"Original: {titles[i]}", fontsize=14)
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Row 2: Permuted
    shuffled_list = [d1_shuffled, d2_shuffled, d3_shuffled]
    for i, ax in enumerate(axes[1]):
        ax.scatter(shuffled_list[i][:, 0], shuffled_list[i][:, 1], c='tab:orange', **scatter_kwargs)
        
        # Add subtitle explaining the effect
        if i == 0:
            sub = "Result: No Difference\n(Product of marginals = Joint)"
        elif i == 1:
            sub = "Result: Visible Difference\n(Correlation removed)"
        else:
            sub = "Result: HUGE Difference\n(Line becomes Blob)"
            
        ax.set_title(f"Permuted\n{sub}", fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Global labels
    fig.suptitle("Effect of Independent Dimension Shuffling (FactorVAE Permutation)", fontsize=16, weight='bold')
    
    plt.show()


plot_distributions()
# %%
# figure for overlaying kld scale and lr

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Define a formatter function
def thousands_formatter(x, pos):
    """Format large numbers as 'Xk' for thousands"""
    if x >= 1000:
        return f'{int(x/1000)}k'
    return f'{int(x)}'

# Setup plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# 1. Load data
try:
    df_kld = pd.read_csv("kld_scale.csv")
    df_lr = pd.read_csv("lr.csv")
except FileNotFoundError:
    raise FileNotFoundError("Please ensure 'kld_scale.csv' and 'lr.csv' are present in the working directory.")

# 2. Setup colors for clear distinction
color_kld = '#D62728'  # Red
color_lr = '#1F77B4'    # Blue

# 3. Create the figure and the first axis (Left: Loss)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Beta
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('$\\beta$', color=color_kld, fontsize=12)
ax1.plot(df_kld['trainer/global_step'], df_kld['imv_v3.2 - vae_kld_scale'], color=color_kld, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_kld, colors=color_kld) # colors= sets tick mark color
ax1.grid(True, alpha=0.3)

# 4. Create the second axis (Right: Learning Rate)
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

# Plot Learning Rate
ax2.set_ylabel('Learning Rate', color=color_lr, fontsize=12)
ax2.plot(df_lr['trainer/global_step'], df_lr['imv_v3.2 - lr_vae'], color=color_lr, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_lr, colors=color_lr) # Matches tick color to LR curve
# ax2.spines['right'].set_color(color_lr)      # Optional: color the right spine blue
# ax2.spines['left'].set_color(color_kld)     # Optional: color the left spine red

# Apply the formatter to x-axis
ax1.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

plt.tight_layout()
plt.savefig("beta_vs_lr.pdf", bbox_inches='tight')
# %%
