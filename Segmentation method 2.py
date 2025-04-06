# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from skimage.morphology import skeletonize
from skimage.draw import polygon

# Load the grayscale image
image_path = 'architectural_plan6.jpg'  # Replace with your image path
image = io.imread(image_path, as_gray=True)

# Step 1: Threshold the image to isolate darker pixels
threshold = 0.55  # Only keep pixels darker than this value
dark_pixels_mask = image <= threshold

# Step 2: Detect contours (borders) around the cloud
contours = measure.find_contours(dark_pixels_mask, level=0.5)  # Detect contours at the boundary of the mask

# Step 3: Plot the original image with contours
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray', alpha=0.5)  # Show the original image in the background

# Plot each contour
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, label='Border')

plt.title('Border of Dark Pixel Clouds')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')
plt.gca().invert_yaxis()
plt.show()


# Step 3.5: Connect separate contours if their endpoints are close

# Parameters for connection threshold
connection_threshold = 20  # Maximum distance to connect contours

# Extract the endpoints of each contour
endpoints = []
for contour in contours:
    if len(contour) > 1:
        endpoints.append((contour[0], contour[-1]))  # First and last points of the contour

# Create a new list to hold the connected contours
connected_contours = [contour for contour in contours if len(contour) > 0]  # Ensure no empty arrays initially

# Iterate through endpoints to find and connect close pairs
for i, (start1, end1) in enumerate(endpoints):
    for j, (start2, end2) in enumerate(endpoints):
        if i >= j or len(connected_contours[j]) == 0:  # Avoid self-comparison, duplicates, or empty contours
            continue
        
        # Check distances between endpoints
        distances = [
            np.linalg.norm(start1 - start2),
            np.linalg.norm(start1 - end2),
            np.linalg.norm(end1 - start2),
            np.linalg.norm(end1 - end2),
        ]
        
        # If the minimum distance is below the threshold, connect the contours
        if min(distances) < connection_threshold:
            # Choose the pair of endpoints to connect
            if len(connected_contours[i]) > 0 and len(connected_contours[j]) > 0:
                idx1, idx2 = divmod(np.argmin(distances), 2)
                if idx1 == 0:
                    connected_contours[i] = np.vstack([connected_contours[j], connected_contours[i]])
                else:
                    connected_contours[i] = np.vstack([connected_contours[i], connected_contours[j]])
            
            # Clear the merged contour to avoid duplicate processing
            connected_contours[j] = np.array([])

# Filter out empty entries after merging
connected_contours = [contour for contour in connected_contours if len(contour) > 0]

# Create a dictionary for connected shapes
connected_shapes_dict = {idx: contour for idx, contour in enumerate(connected_contours)}

# Plot the connected contours after Step 3.5

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray', alpha=0.5)  # Show the original image in the background

# Generate new unique colors for the connected shapes
colors = plt.cm.jet(np.linspace(0, 1, len(connected_contours)))

for idx, contour in connected_shapes_dict.items():
    plt.plot(contour[:, 1], contour[:, 0], color=colors[idx], linewidth=2, label=f'Connected Shape {idx + 1}')

plt.title('Connected Solid Shapes After Merging')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

# At this point, `connected_shapes_dict` contains the connected shapes with unique labels.





# Step 4: Skeletonization for each contour
# Create a binary mask from the contours
binary_shape = np.zeros_like(dark_pixels_mask, dtype=bool)
for contour in contours:
    rr, cc = polygon(contour[:, 0], contour[:, 1], binary_shape.shape)
    binary_shape[rr, cc] = True

# Perform skeletonization
skeleton = skeletonize(binary_shape)

# Step 5: Plot the final skeleton on a larger canvas
plt.figure(figsize=(12, 12))
plt.imshow(image, cmap='gray', alpha=0.5)  # Show the original image in the background
plt.plot(skeleton.nonzero()[1], skeleton.nonzero()[0], 'r.', markersize=1, label='Skeleton')
plt.title('Skeleton of Dark Pixel Clouds (Enhanced Canvas)')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

# Step 6: Plot only the skeleton on a separate canvas
plt.figure(figsize=(12, 12))
plt.plot(skeleton.nonzero()[1], skeleton.nonzero()[0], 'k.', markersize=1, label='Skeleton')
plt.title('Skeleton Only (Separate Canvas)')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')

plt.legend()
plt.show()



from matplotlib.path import Path

# Step 7: Identify which closed shape each skeleton point belongs to

# Extract skeleton points
skeleton_points = np.column_stack(np.where(skeleton))

# Create a dictionary to map skeleton points to shapes
skeleton_point_labels = {}

for idx, contour in connected_shapes_dict.items():
    # Create a Path object for the current contour
    path = Path(contour[:, [1, 0]])  # Use (x, y) format

    for i, point in enumerate(skeleton_points):
        if path.contains_point((point[1], point[0])):  # Check if the point is inside the contour
            skeleton_point_labels[i] = idx  # Map the skeleton point index to the shape ID

# Step 8: Visualize labeled skeleton points
plt.figure(figsize=(12, 12))
plt.imshow(image, cmap='gray', alpha=0.5)  # Show the original image in the background

# Generate unique colors for the labels
label_colors = plt.cm.tab20(np.linspace(0, 1, len(connected_shapes_dict)))

for i, point in enumerate(skeleton_points):
    if i in skeleton_point_labels:
        shape_id = skeleton_point_labels[i]
        plt.plot(point[1], point[0], '.', color=label_colors[shape_id], label=f'Shape {shape_id}' if shape_id == 0 else "")

plt.title('Skeleton Points Labeled by Shape')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

# Step 9: Print the results
print(f"Total skeleton points: {len(skeleton_points)}")
print(f"Labeled skeleton points: {len(skeleton_point_labels)}")




from scipy.spatial.distance import cdist

# Step 10: Fit polylines to the skeleton points grouped by shapes

# Parameters for polyline fitting
max_iterations = 100  # Maximum iterations for optimization
max_distance = 20  # Maximum distance for connecting points
polylines = {}  # Dictionary to store the fitted polylines for each shape

for shape_id in range(len(connected_shapes_dict)):
    # Extract skeleton points belonging to the current shape
    shape_points = np.array([
        skeleton_points[i] for i in range(len(skeleton_points))
        if skeleton_point_labels.get(i) == shape_id
    ])
    
    if len(shape_points) < 2:  # Skip if not enough points to fit a polyline
        continue
    
    # Initialize the polyline with the first point
    polyline = [shape_points[0]]
    remaining_points = shape_points[1:]
    
    # Iteratively construct the polyline
    for _ in range(max_iterations):
        if len(remaining_points) == 0:
            break
        
        # Find the closest point to the last point in the polyline
        last_point = polyline[-1].reshape(1, -1)
        distances = cdist(last_point, remaining_points)
        min_idx = np.argmin(distances)
        
        # Check if the distance is within the maximum threshold
        if distances[0, min_idx] <= max_distance:
            polyline.append(remaining_points[min_idx])
            remaining_points = np.delete(remaining_points, min_idx, axis=0)
        else:
            break
    
    polylines[shape_id] = np.array(polyline)

# Step 11.1: Visualize the fitted polylines in context
plt.figure(figsize=(12, 12))
plt.imshow(image, cmap='gray', alpha=0.5)  # Show the original image in the background

# Generate unique colors for the fitted polylines
polyline_colors = plt.cm.tab20(np.linspace(0, 1, len(polylines)))

for shape_id, polyline in polylines.items():
    plt.plot(polyline[:, 1], polyline[:, 0], '-', color=polyline_colors[shape_id], linewidth=2, label=f'Shape {shape_id}')

plt.title('Fitted Polylines for Shapes')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

# Step 11.2: Plot only the fitted polylines on a separate enlarged canvas
plt.figure(figsize=(20, 20))

for shape_id, polyline in polylines.items():
    plt.plot(polyline[:, 1], polyline[:, 0], '-', color=polyline_colors[shape_id], linewidth=2, label=f'Shape {shape_id}')

plt.title('Fitted Polylines Only (Enlarged Canvas)')
plt.xlabel('X (Column)')
plt.ylabel('Y (Row)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

import math

# Step 12: Compute alpha_angles for each polyline
def compute_alpha_angles(polyline):
    """
    Computes the angles (alpha_angles) between consecutive segments of a polyline.
    """
    alpha_angles = []
    for i in range(1, len(polyline) - 1):
        # Define vectors from the current vertex
        vector_prev = polyline[i] - polyline[i - 1]
        vector_next = polyline[i + 1] - polyline[i]

        # Compute the dot product and magnitudes of the vectors
        dot_product = np.dot(vector_prev, vector_next)
        magnitude_prev = np.linalg.norm(vector_prev)
        magnitude_next = np.linalg.norm(vector_next)

        # Avoid division by zero
        if magnitude_prev == 0 or magnitude_next == 0:
            alpha_angles.append(0)  # Default to 0 if vectors are degenerate
            continue

        # Compute the angle in radians and convert to degrees
        cos_theta = dot_product / (magnitude_prev * magnitude_next)
        alpha_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Ensure numerical stability
        alpha_angles.append(np.degrees(alpha_angle))
    return alpha_angles

# Dictionary to store alpha_angles for each shape
alpha_angles_dict = {}

# Compute alpha_angles for each polyline
for shape_id, polyline in polylines.items():
    alpha_angles = compute_alpha_angles(polyline)
    alpha_angles_dict[shape_id] = alpha_angles

# Step 12.1: Visualize the alpha_angles as a histogram for each shape
plt.figure(figsize=(15, 10))
for shape_id, alpha_angles in alpha_angles_dict.items():
    plt.hist(alpha_angles, bins=20, alpha=0.5, label=f'Shape {shape_id}')

plt.title('Distribution of Alpha Angles for Each Shape')
plt.xlabel('Alpha Angle (degrees)')
plt.ylabel('Frequency')
plt.legend()
plt.show()





from scipy.spatial.distance import euclidean

def douglas_peucker(polyline, epsilon):
    """
    Simplifies a polyline using the Douglas-Peucker algorithm.
    Args:
        polyline (np.ndarray): Polyline represented as a Nx2 array of points.
        epsilon (float): The maximum distance allowed between the original line and the simplified line.
    Returns:
        np.ndarray: Simplified polyline as a Nx2 array.
    """
    # Find the point with the maximum distance from the line segment
    def perpendicular_distance(point, start, end):
        if np.all(start == end):  # Avoid division by zero
            return euclidean(point, start)
        return np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

    if len(polyline) < 3:
        return polyline  # A polyline with less than 3 points cannot be simplified further

    start, end = polyline[0], polyline[-1]
    distances = [perpendicular_distance(pt, start, end) for pt in polyline[1:-1]]
    max_distance_idx = np.argmax(distances)
    max_distance = distances[max_distance_idx]

    if max_distance > epsilon:
        # Recursively simplify
        idx = max_distance_idx + 1
        left = douglas_peucker(polyline[:idx + 1], epsilon)
        right = douglas_peucker(polyline[idx:], epsilon)
        return np.vstack((left[:-1], right))  # Merge the two halves
    else:
        return np.array([start, end])

# Step 13: Apply the Douglas-Peucker algorithm to optimize polylines
epsilon = 1.0  # Tolerance for simplification; lower values retain more detail
optimized_polylines = {}
for shape_id, polyline in polylines.items():
    optimized_polylines[shape_id] = douglas_peucker(polyline, epsilon)

# Step 14: Visualize the optimized polylines
plt.figure(figsize=(12, 12))
plt.imshow(image, cmap='gray', alpha=0.5)  # Show the original image in the background

# Generate unique colors for optimized polylines
optimized_colors = plt.cm.tab20(np.linspace(0, 1, len(optimized_polylines)))

for shape_id, optimized_polyline in optimized_polylines.items():
    plt.plot(
        optimized_polyline[:, 1],  # Use y-coordinate for the second index
        optimized_polyline[:, 0],  # Use x-coordinate for the first index
        '-',
        linewidth=2,
        color=optimized_colors[shape_id],
        label=f"Optimized Shape {shape_id}",
    )

plt.title("Optimized Polylines with Douglas-Peucker")
plt.xlabel("X (Column)")
plt.ylabel("Y (Row)")
plt.gca().invert_yaxis()
plt.legend()
plt.show()

# Plot only the optimized polylines on an enlarged canvas
plt.figure(figsize=(20, 20))
for shape_id, optimized_polyline in optimized_polylines.items():
    plt.plot(
        optimized_polyline[:, 1],  # Use y-coordinate for the second index
        optimized_polyline[:, 0],  # Use x-coordinate for the first index
        '-',
        linewidth=2,
        color=optimized_colors[shape_id],
        label=f"Optimized Shape {shape_id}",
    )

plt.title("Optimized Polylines Only (Enlarged Canvas)")
plt.xlabel("X (Column)")
plt.ylabel("Y (Row)")
plt.gca().invert_yaxis()
plt.legend()
plt.show()






