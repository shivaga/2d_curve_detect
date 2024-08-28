import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import numpy as np

def plot_regularized_shapes(regular_shapes, title="Regularized Shapes"):
    """
    Visualizes the regularized shapes using Matplotlib.
    """
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    
    for shape_type, polyline in regular_shapes:
        if shape_type == 'line':
            ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=2)
        elif shape_type == 'circle':
            centroid = np.mean(polyline, axis=0)
            radius = np.linalg.norm(polyline[0] - centroid)
            circle = plt.Circle(centroid, radius, color='r', fill=False)
            ax.add_patch(circle)
        elif shape_type == 'ellipse':
            centroid = np.mean(polyline, axis=0)
            cov = np.cov(polyline.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle, edgecolor='g', fill=False)
            ax.add_patch(ellipse)
        elif shape_type == 'rectangle':
            polygon = Polygon(polyline, closed=True, edgecolor='m', fill=False)
            ax.add_patch(polygon)
        elif shape_type == 'polygon':
            polygon = Polygon(polyline, closed=True, edgecolor='y', fill=False)
            ax.add_patch(polygon)
        elif shape_type == 'star':
            polygon = Polygon(polyline, closed=True, edgecolor='k', fill=False)
            ax.add_patch(polygon)
    
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

def save_as_svg(regular_shapes, file_name="output.svg"):
    """
    Saves the regularized shapes as an SVG file.
    """
    from svgwrite import Drawing
    dwg = Drawing(file_name, profile='tiny')
    
    for shape_type, polyline in regular_shapes:
        if shape_type == 'line':
            dwg.add(dwg.polyline(points=polyline.tolist(), stroke='blue', fill='none'))
        elif shape_type == 'circle':
            centroid = np.mean(polyline, axis=0)
            radius = np.linalg.norm(polyline[0] - centroid)
            dwg.add(dwg.circle(center=centroid.tolist(), r=radius, stroke='red', fill='none'))
        elif shape_type == 'ellipse':
            centroid = np.mean(polyline, axis=0)
            cov = np.cov(polyline.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            dwg.add(dwg.ellipse(center=centroid.tolist(), r=(width/2, height/2), rotate=angle, stroke='green', fill='none'))
        elif shape_type == 'rectangle':
            dwg.add(dwg.polygon(points=polyline.tolist(), stroke='magenta', fill='none'))
        elif shape_type == 'polygon':
            dwg.add(dwg.polygon(points=polyline.tolist(), stroke='yellow', fill='none'))
        elif shape_type == 'star':
            dwg.add(dwg.polygon(points=polyline.tolist(), stroke='black', fill='none'))
    
    dwg.save()
