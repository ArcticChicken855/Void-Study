import numpy as np

def inverse_distance_mesh(distances, alpha):
    """
    Generates mesh sizes based on the inverse distance function.
    
    Parameters:
    - distances: List or array of distances from the boundary.
    - h0: Maximum mesh size (initial mesh spacing).
    - alpha: Scaling factor that controls how quickly the mesh size decreases near the boundary.
    
    Returns:
    - List of mesh sizes corresponding to the input distances.
    """
    # Calculate mesh sizes using the inverse distance formula
    mesh_sizes = [float(1/ (1 + alpha * d)) for d in distances]
    return mesh_sizes

alpha = 1.5
N = 5
distances = np.linspace(1, 0, N)
print(inverse_distance_mesh(distances, alpha))
