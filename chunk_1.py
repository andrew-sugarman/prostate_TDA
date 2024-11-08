import os
import tifffile as tiff
import numpy as np

# Load the stack of TIFF images into a 3D NumPy array
# Define the directory containing the TIFF files
input_dir = r'C:\Users\g7712_razer2\cheng_lab_data\prostate_reconstruction_b8\JRW006\jrw006_section1_reslice'

# Get a sorted list of all TIFF files in the directory
tiff_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')])

# Load each TIFF file and stack them into a 3D NumPy array
#tiff_stack = np.array([tiff.imread(file) for file in tiff_files])
tiff_stack = tiff.imread(tiff_files)

#
print(f'Loaded {len(tiff_stack)} slices into a 3D NumPy array with shape {tiff_stack.shape}')

# Define the size of each chunk and the overlap
# chunk_size = (128, 128, 128)  # (z, y, x) dimensions of each chunk
chunk_size = (256, 256, 256)  # (z, y, x) dimensions of each chunk
overlap_size = (16, 16, 16)  # (z, y, x) overlap in each dimension

# Get the shape of the 3D array
z_size, y_size, x_size = tiff_stack.shape

# List to store chunks
chunks = []

# Loop through the 3D array and extract chunks with overlap
for z in range(0, z_size, chunk_size[0] - overlap_size[0]):
    for y in range(0, y_size, chunk_size[1] - overlap_size[1]):
        for x in range(0, x_size, chunk_size[2] - overlap_size[2]):
            # Calculate the start and end indices for each chunk, making sure not to exceed the boundaries
            z_start = z
            z_end = min(z + chunk_size[0], z_size)
            
            y_start = y
            y_end = min(y + chunk_size[1], y_size)
            
            x_start = x
            x_end = min(x + chunk_size[2], x_size)

            # Extract the chunk
            chunk = tiff_stack[z_start:z_end, y_start:y_end, x_start:x_end]
            chunks.append(chunk)

#
stitched_volume = np.zeros(tiff_stack.shape, dtype=tiff_stack.dtype)

## stitch chunkies together
# for chunk, (z_start, z_end, y_start, y_end, x_start, x_end) in zip(chunks, indices_list):
#     # Overwrite the existing data in the stitched volume with the non-overlapping part of each chunk
#     stitched_volume[z_start+overlap_size[0]//2:z_end-overlap_size[0]//2,
#                     y_start+overlap_size[1]//2:y_end-overlap_size[1]//2,
#                     x_start+overlap_size[2]//2:x_end-overlap_size[2]//2] = chunk[
#                     overlap_size[0]//2:chunk.shape[0]-overlap_size[0]//2,
#                     overlap_size[1]//2:chunk.shape[1]-overlap_size[1]//2,
#                     overlap_size[2]//2:chunk.shape[2]-overlap_size[2]//2]

# Define the directory where the chunks will be saved
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Iterate over chunks and save each one as a separate TIFF file
for i, chunk in enumerate(chunks):
    output_path = os.path.join(output_dir, f'chunk_{i}.tif')
    tiff.imwrite(output_path, chunk)
    print(f'Saved chunk {i} to {output_path}')