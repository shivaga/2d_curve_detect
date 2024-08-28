import numpy as np
import cv2


# Step 2: Convert arrays to contours for processing
def array_to_contour(array):
    contour = np.array(array, dtype=np.int32).reshape((-1, 1, 2))
    return contour
def detect_shapes(polyline, epsilon_factor=0.05):
    """
    Detects the type of shape based on the provided polyline.
    
    Args:
        polyline (np.ndarray): The polyline representing the contour.
        epsilon_factor (float): Factor for approximating the polyline's irregularity.
    
    Returns:
        str: Type of the shape detected.
    """
    # Approximate the contour
    epsilon = epsilon_factor * cv2.arcLength(polyline, True)
    approx = cv2.approxPolyDP(polyline, epsilon, True)

    # Detect shape based on the number of vertices
    if len(approx) == 2:
        shape_type = "Line"
    elif len(approx) == 3:
        shape_type = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:  # Square
            shape_type = "Square"
        else:  # Rectangle
            shape_type = "Rectangle"
    elif len(approx) > 4:
        # Calculate circularity
        area = cv2.contourArea(polyline)
        (x, y), radius = cv2.minEnclosingCircle(polyline)
        circularity = (4 * np.pi * area) / (cv2.arcLength(polyline, True) ** 2)

        if circularity > 0.7:  # Circle or Ellipse
            shape_type = "Circle or Ellipse"
        elif len(approx) == 10:
            shape_type = "Star"
        else:
            shape_type = "Regular Polygon"
    else:
        shape_type = "Unknown"

    return shape_type


def detect_regular_shapes(polylines):
    """
    Detects regular shapes from a list of polylines.
    
    Args:
        polylines (list): List of numpy arrays representing polylines.
    
    Returns:
        list: Detected regular shapes with their types.
    """
    detected_shapes = []
    
    for polyline in polylines:
        contour=array_to_contour(polyline)
        shape_type = detect_shapes(contour)
        detected_shapes.append((contour, shape_type))
    
    return detected_shapes
