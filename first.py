import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from matplotlib.cm import get_cmap
from shapely.geometry import LineString, Point, Polygon
from rtree import index
from scipy.spatial.distance import euclidean
import copy
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import math

def distance(p1, p2):
  """Calculates the Euclidean distance between two points."""
  return np.sqrt(np.sum((p1 - p2) ** 2))


def is_line(polyline, threshold=0.1):
    # Ensure there are at least two points
    if len(polyline) < 2:
        return False, None
    
    polyline = np.array(polyline)  # Convert list to numpy array
    
    # Calculate differences in x and y coordinates
    dx = polyline[1:, 0] - polyline[0, 0]
    dy = polyline[1:, 1] - polyline[0, 1]
    
    # Calculate the slopes, avoiding division by zero
    slopes = dy / (dx + (dx == 0))
    
    # Calculate the variance of the slopes
    variance = np.var(slopes)
    
    # Check if the variance is below the threshold
    is_straight = variance < threshold
    
    # Return whether it's a straight line and the endpoints
    return is_straight, [polyline[0], polyline[-1]]


def is_circle(points, variance_threshold=10):
    points = np.array(points)
    
    # Calculate the mean of all points to estimate the center
    center = np.mean(points, axis=0)
    
    # Calculate the distance of each point from the center
    distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
    
    # Calculate the variance of the distances
    distance_variance = np.var(distances)
    # print(distance_variance)
    
    # Check if the variance is below the specified threshold
    return distance_variance < variance_threshold,[center[0],center[1],np.mean(distances)]


def find_closest_point(target_point, points):
    #Find the point in 'points' closest to 'target_point'

    points = np.array(points) # Convert points to a NumPy array for efficient computation
    target_point = np.array(target_point)  # Convert target_point to a NumPy array
    
    # Compute the Euclidean distance from the target_point to each point in points
    distances = np.linalg.norm(points - target_point, axis=1)
    
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    # Return the point with the minimum distance
    return points[closest_index]

def get_rect_corners(points):
    #Find representative corner points from a dataset based on distance to target corners.
    points = np.array(points)
    
    # Find the minimum and maximum x-coordinates
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])

    # Find the minimum and maximum y-coordinates
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    target_corners = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]
    
    # Find the closest points to each target corner
    corner_points = []
    for corner in target_corners:
        closest_point = find_closest_point(corner, points)
        if not any(np.array_equal(closest_point, p) for p in corner_points):
            corner_points.append(closest_point)
    corner_points=np.array(corner_points)
    return corner_points



def is_rectangle(polyline, length_tolerance=0.1):
    #Checks if a polyline is a rectangle within given tolerances.
   
    corners=get_rect_corners(polyline)
    
    # Check for four unique corners
    if len(corners) != 4:
        return False,None

    # Check for opposite sides (parallel and equal length)
    def side_length(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    side_lengths = [side_length(corners[i], corners[(i + 1) % 4]) for i in range(4)]

    # Opposite sides should be equal
    if not (abs(side_lengths[0] - side_lengths[2]) <= length_tolerance * side_lengths[0] and
            abs(side_lengths[1] - side_lengths[3]) <= length_tolerance * side_lengths[1]):
        return False,None
    
    # Check for right angles between adjacent sides
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp the value to the range [-1, 1] to avoid invalid values for arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta) * 180 / np.pi
    

    for i in range(4):
        v1 = corners[i] - corners[(i - 1+4)%4]
        v2 = corners[(i + 1) % 4] - corners[i]
        angle = angle_between(v1, v2)
        # print(angle)
        if not abs(angle - 90) <= 10:  # Assuming angle_tolerance of 10 degrees
            return False,None
    
    return True,corners

def calculate_angle(v1, v2):
    #Calculate the angle between two vectors in degrees
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Avoid division by zero and numerical errors
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    cos_theta = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def identify_corners(points, angle_threshold):
    #Identify corner points from a set of points based on angle change
    # print(points)
    num_points = len(points)
    corners = []
    
    for i in range(num_points):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % num_points]
        
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        angle = calculate_angle(v1, v2)
        # print(angle)
        
        if angle >= angle_threshold and angle <= (160):
            if not any(np.array_equal(p2, c) for c in corners):
                corners.append(p2)
    # corners=np.array(corners)
    # print(corners)
    # plt.scatter(corners[:,0],corners[:,1],c='r')
    
    if len(corners):
        return True, corners
    else:
        return False, []
    
def is_star(points):
    angle_threshold=30
    return identify_corners(points,angle_threshold)
    


def regularize(polyline):
  # Attempts to identify the shape of a polyline
  isline,features=is_line(polyline)
  if isline:
    return "Line",features
  # Not a line
  iscircle,features=is_circle(polyline)
  if iscircle:
    return "Circle",features
  # Not a circle
  isrect,corners=is_rectangle(polyline)
  if isrect:
    # print("Rectangle")
    return "Rectangle",corners
  # Not a Rectangle
  isstar,points=is_star(polyline)
  if(isstar):
     return "Star",points
  # Not a Star
  return "Unrecognized Shape",None



def find_all_closest_polylines(polyline, polylines, threshold):
    closest_polylines = []
    start_end_combinations = []

    for i, candidate in enumerate(polylines):
        if np.array_equal(polyline, candidate):
            continue
        d_start_start = euclidean(polyline[0], candidate[0])
        d_start_end = euclidean(polyline[0], candidate[-1])
        d_end_start = euclidean(polyline[-1], candidate[0])
        d_end_end = euclidean(polyline[-1], candidate[-1])

        d_min = min(d_start_start, d_start_end, d_end_start, d_end_end)
        if d_min <= threshold:
            closest_polylines.append((candidate, i))
            if d_min == d_start_start:
                start_end_combinations.append(('start', 'start'))
            elif d_min == d_start_end:
                start_end_combinations.append(('start', 'end'))
            elif d_min == d_end_start:
                start_end_combinations.append(('end', 'start'))
            else:
                start_end_combinations.append(('end', 'end'))

    return closest_polylines, start_end_combinations

def connect_polylines_backtracking(current_curve, polylines, distance_threshold, visited):
    # Plot the current curve
    plt.figure()
    current_curve_array = np.array(current_curve)
    plt.plot(current_curve_array[:, 0], current_curve_array[:, 1], 'o-')
    plt.title("Current Curve")
    plt.show()
    if not polylines:
        return [current_curve]
    
    # last_point = current_curve[-1]
    closest_polylines, start_end_combinations = find_all_closest_polylines(np.array(current_curve), polylines, distance_threshold)
    if not closest_polylines:
        return [current_curve]
    
    all_curves = []
    
    for (closest_polyline, closest_index), start_end in zip(closest_polylines, start_end_combinations):
        if closest_index in visited:
            continue
        
        new_curve = copy.deepcopy(current_curve)
        
        if start_end == ('start', 'start'):
            new_curve = closest_polyline[::-1].tolist() + new_curve
        elif start_end == ('start', 'end'):
            new_curve = closest_polyline.tolist() + new_curve
        elif start_end == ('end', 'start'):
            new_curve = new_curve + closest_polyline.tolist()
        elif start_end == ('end', 'end'):
            new_curve = new_curve + closest_polyline[::-1].tolist()
        
        new_visited = copy.deepcopy(visited)
        new_visited.add(closest_index)
        remaining_polylines = [poly for i, poly in enumerate(polylines) if i != closest_index]
        
        curves = connect_polylines_backtracking(new_curve, remaining_polylines, distance_threshold, new_visited)
        all_curves.extend(curves)
    
    return all_curves

def find_all_connected_curves(polylines, distance_threshold):
    all_curves = []
    
    for i, polyline in enumerate(polylines):
        initial_curve = polyline.tolist()
        visited = {i}
        remaining_polylines = [poly for j, poly in enumerate(polylines) if j != i]
        
        curves = connect_polylines_backtracking(initial_curve, remaining_polylines, distance_threshold, visited)
        # print(curves)
        all_curves.extend(curves)
    
    return all_curves

def visualize_curves(curves):
    plt.figure()
    cmap = get_cmap('tab10')  # Choose a colormap with a range of colors

    for idx, curve in enumerate(curves):
        color = cmap(idx % cmap.N)  # Get a color from the colormap
        curve_array = np.array(curve)
        plt.plot(curve_array[:, 0], curve_array[:, 1], 'o-', color=color, label=f'Curve {idx + 1}')
        plt.title('All Curves')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
    plt.show()




def read_csv(csv_path):
  np_path_XYs = np.genfromtxt(csv_path , delimiter=',') 
  path_XYs = []
  for i in np.unique(np_path_XYs[:, 0]):
    npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:] 
    XYs = []
    for j in np.unique(npXYs[:, 0]):
        XY = npXYs[npXYs[:, 0] == j][:, 1:]
        XYs.append(XY) 
    path_XYs.append(XYs)
  return path_XYs




def plot_shapes(paths_XYs):
  fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
  colours = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange',
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
#   group=np.array(paths_XYs)
  for i,group in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        c1 = colours[(len(colours)-i-1) % len(colours)] 
        shape,features = regularize(group)
        print(shape)
        group=np.array(group)
        if(shape=="Star"):
            new_points = np.array(features)
            ax.plot(new_points[:,0],new_points[:,1],c=c,linewidth=2,label="Not a star")
        elif shape == "Straight Line":
            start, end = features
            ax.plot([start[0], end[0]], [start[1], end[1]], c, linewidth=2, label='Not a Line')
            ax.plot(start[0], start[1], 'go', label='Start')
            ax.plot(end[0], end[1], 'ro', label='End')
        elif shape == "Circle":
            center_x, center_y, radius = features
            circle = plt.Circle((center_x, center_y), radius, color=c, fill=False, linewidth=2)
            ax.add_artist(circle)
            ax.plot(center_x, center_y, 'go', label='Center')
            ax.annotate(f'Radius: {radius:.2f}', (center_x, center_y + radius), color='r')
        elif shape == "Rectangle":
            corner1,_,corner3,_ = features
            min_x=corner1[0]
            min_y=corner1[1]
            max_x=corner3[0]
            max_y=corner3[1]
            rectangle = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, edgecolor=c, fill=False, linewidth=2)
            ax.add_artist(rectangle)
            ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'go', label='Corners')
        elif shape=="Unrecognized Shape":
            ax.plot(group[:,0],group[:,1],c=c1,linewidth=2,label="Not known")
  ax.plot(group[:,0],group[:,1],c='r',linewidth=2,label="Not known")
  ax.legend()
  ax.set_aspect('equal')
  plt.show()

def calculate_angle_new(p1, p2, p3):
    """Calculate the angle between the line segments (p1p2) and (p2p3)."""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Avoid division by zero and numerical errors
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    cos_theta = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def identify_corner_points_new(points, angle_threshold=30):
    """Identify corner points based on the angle threshold."""
    corners = [0]  # Start with the first point index
    for i in range(1, len(points) - 1):
        angle = calculate_angle_new(points[i-1], points[i], points[i+1])
        if angle>=30 and angle<=160:
            corners.append(i)
    corners.append(len(points) - 1)  # End with the last point index
    return corners


def segment_points(points, angle_threshold=30):
    """Segment the points into sets based on identified corner points."""
    points=np.array(points)
    corners = identify_corner_points_new(points, angle_threshold)
    sets = []
    for i in range(len(corners) - 1):
        sets.append(points[corners[i]:corners[i+1]+1])
    return sets

def main(csv_path):
    path_XYs = read_csv(csv_path)
    # print(len(path_XYs))
    distance_threshold =1  # Example threshold
    all_polylines = [item for sublist in path_XYs for item in sublist]
    # print(len(all_polylines[8]))
    new_polylines=[]
    for polyline in all_polylines:
        new_polyline=segment_points(polyline)
        new_polylines.extend(new_polyline)
    # print(len(new_polylines[2:]))
    new_new_polylines=[]
    for new_polyline in new_polylines:
        if(len(new_polyline)>=10):
            new_new_polylines.extend([new_polyline])
    visualize_curves(new_new_polylines)
    # print(len(new_new_polylines))
    # all_curves = find_all_connected_curves(new_new_polylines, distance_threshold)
    # visualize_curves(all_curves)
    # plot_shapes(all_curves)

# Example usage
csv_file = "./problems/isolated.csv"  # Replace with your actual CSV file path
main(csv_file)



