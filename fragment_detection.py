import numpy as np
from scipy.spatial import distance
from curve2d_detect.regularization import detect_regular_shapes
from curve2d_detect.symmetry import detect_reflection_symmetry

def merge_polylines(polylines, threshold=5):
    """
    Merges polylines that are close enough to be part of the same shape.
    
    Args:
        polylines (list): List of numpy arrays representing polylines.
        threshold (float): Distance threshold for merging polylines.
        
    Returns:
        list: Merged polylines.
    """
    merged_polylines = []
    while polylines:
        polyline = polylines.pop(0)
        merged = False
        for i, other in enumerate(polylines):
            if (distance.euclidean(polyline[-1], other[0]) < threshold or
                distance.euclidean(polyline[0], other[-1]) < threshold):
                merged_polylines.append(np.vstack([polyline, other]))
                polylines.pop(i)
                merged = True
                break
        if not merged:
            merged_polylines.append(polyline)
    return merged_polylines

def detect_shapes_in_fragments(paths_XYs, merge_threshold=5):
    """
    Handles fragmented shapes by merging and detecting regular shapes and symmetries.
    
    Args:
        paths_XYs (list): List of paths, where each path is a list of polylines.
        merge_threshold (float): Distance threshold for merging polylines.
        
    Returns:
        list: Detected regular shapes and symmetries.
    """
    all_polylines = [polyline for path in paths_XYs for polyline in path]
    merged_polylines = merge_polylines(all_polylines, threshold=merge_threshold)
    regular_shapes = []
    
    for polyline in merged_polylines:
        shapes = detect_regular_shapes([polyline])
        if shapes:
            regular_shapes.extend(shapes)
        
        symmetries = detect_reflection_symmetry([polyline])
        if symmetries:
            regular_shapes.extend(symmetries)
    
    return regular_shapes
