import numpy as np

def detect_reflection_symmetry(paths_XYs):
    """
    Detects reflection symmetries in closed shapes.
    Returns a list of tuples containing the type of symmetry and the corresponding polyline.
    """
    symmetries = []
    for path in paths_XYs:
        for polyline in path:
            if has_reflection_symmetry(polyline):
                symmetries.append(('reflection', polyline))
    
    return symmetries

def has_reflection_symmetry(polyline):
    """
    Checks if the polyline has reflection symmetry by comparing the polyline to its mirrored version.
    """
    centroid = np.mean(polyline, axis=0)
    mirrored = 2 * centroid - polyline  # Mirror across the centroid
    return np.allclose(polyline, mirrored[::-1], atol=1e-2)  # Check if the polyline is close to its mirror
