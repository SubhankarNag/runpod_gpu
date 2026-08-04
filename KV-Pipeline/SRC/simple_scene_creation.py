"""
From Scene Creation GUI, we usually get a shape which is not compatible with TIGRE
We run this to convert it, So Pipeline runs without any issue
"""
import numpy as np

# Create an empty volume matching the exact voxel bounds defined in geometry
actual_fov = np.load("../Simulated_Scene/Scene1_1p3mm.npy")

# Change the dimensions, so it would be compatible with Pipeline
actual_fov = actual_fov.transpose((2, 0, 1))
print(actual_fov.shape)

# Export to disk for simulation ingestion
np.save("../Simulated_Scene/Scene1_1p3mm_converted.npy", actual_fov)