# IFC file Import and extraction of the data, indexing the walls




import ifcopenshell
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon, MultiPoint
from collections import defaultdict
import random

# Open the IFC file (replace with your IFC file path)
ifc_file = ifcopenshell.open("C:/Users/MBodrov/training_set_cnn/bim/bim7.ifc")

# Function to calculate angle from center to a point
def calculate_angle(center, point):
    dx, dy = point[0] - center[0], point[1] - center[1]
    return np.arctan2(dy, dx) % (2 * np.pi)

# Create transformation matrix
def create_transformation_matrix(translation, rotation_angle):
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return translation_matrix @ rotation_matrix

# Apply transformation to a point
def apply_transformation(point, transformation_matrix):
    x, y = point
    point_vector = np.array([x, y, 1])  # Homogeneous coordinates
    transformed_point = transformation_matrix @ point_vector
    return transformed_point[0], transformed_point[1]

# Extract cumulative transformation from IfcLocalPlacement
def get_cumulative_transformation(placement):
    cumulative_transformation = np.identity(3)
    while placement:
        if placement.is_a("IfcLocalPlacement"):
            relative_placement = placement.RelativePlacement
            translation = [0, 0]
            rotation_angle = 0
            if relative_placement:
                if relative_placement.Location:
                    coords = relative_placement.Location.Coordinates
                    translation = [coords[0], coords[1]]
                if hasattr(relative_placement, 'RefDirection') and relative_placement.RefDirection:
                    dir_ratios = relative_placement.RefDirection.DirectionRatios
                    rotation_angle = np.arctan2(dir_ratios[1], dir_ratios[0])
            local_transformation = create_transformation_matrix(translation, rotation_angle)
            cumulative_transformation = local_transformation @ cumulative_transformation
            placement = placement.PlacementRelTo
        else:
            break
    return cumulative_transformation

# Function to extract wall length from IFC data
def get_wall_length(wall):
    if wall.IsDefinedBy:
        for rel in wall.IsDefinedBy:
            if rel.is_a("IfcRelDefinesByProperties") and rel.RelatingPropertyDefinition:
                prop_def = rel.RelatingPropertyDefinition
                if prop_def.is_a("IfcElementQuantity"):
                    for quantity in prop_def.Quantities:
                        if quantity.is_a("IfcQuantityLength") and quantity.Name == "Length":
                            return quantity.LengthValue
    return None

# Function to extract global geometry for straight walls
def extract_straight_wall_points(wall):
    wall_points = []
    transformation = get_cumulative_transformation(wall.ObjectPlacement)
    if wall.Representation:
        for rep in wall.Representation.Representations:
            if rep.is_a("IfcShapeRepresentation"):
                for item in rep.Items:
                    if item.is_a("IfcExtrudedAreaSolid"):
                        profile = item.SweptArea
                        if hasattr(profile, 'OuterCurve') and profile.OuterCurve.is_a("IfcPolyline"):
                            for p in profile.OuterCurve.Points:
                                if hasattr(p, 'Coordinates'):
                                    x, y = p.Coordinates[0], p.Coordinates[1]
                                    transformed_x, transformed_y = apply_transformation((x, y), transformation)
                                    wall_points.append((transformed_x, transformed_y))
    return wall_points

# Function to extract arc data for curved wall profiles using IfcTrimmedCurve and IfcCircle
def extract_arc_data_with_transform(wall):
    arc_data = []
    transformation = get_cumulative_transformation(wall.ObjectPlacement)
    
    if wall.Representation:
        for rep in wall.Representation.Representations:
            if rep.is_a("IfcShapeRepresentation"):
                for item in rep.Items:
                    if item.is_a("IfcExtrudedAreaSolid"):
                        profile = item.SweptArea
                        if hasattr(profile, "OuterCurve"):
                            outer_curve = profile.OuterCurve
                            if outer_curve.is_a("IfcCompositeCurve"):
                                for segment in outer_curve.Segments:
                                    if segment.ParentCurve.is_a("IfcTrimmedCurve"):
                                        trimmed_curve = segment.ParentCurve
                                        basis_curve = trimmed_curve.BasisCurve
                                        if basis_curve.is_a("IfcCircle"):
                                            # Extract circle data
                                            radius = basis_curve.Radius
                                            circle_position = basis_curve.Position
                                            center_x, center_y = circle_position.Location.Coordinates[:2]
                                            center = apply_transformation((center_x, center_y), transformation)

                                            # Get start and end angles from Trim1 and Trim2
                                            trim1_param = trimmed_curve.Trim1[0]
                                            trim2_param = trimmed_curve.Trim2[0]

                                            if hasattr(trim1_param, 'Coordinates'):
                                                trim1_coords = apply_transformation(
                                                    (trim1_param.Coordinates[0], trim1_param.Coordinates[1]),
                                                    transformation)
                                                start_angle = calculate_angle(center, trim1_coords)
                                            else:
                                                start_angle = float(trim1_param) % (2 * np.pi)

                                            if hasattr(trim2_param, 'Coordinates'):
                                                trim2_coords = apply_transformation(
                                                    (trim2_param.Coordinates[0], trim2_param.Coordinates[1]),
                                                    transformation)
                                                end_angle = calculate_angle(center, trim2_coords)
                                            else:
                                                end_angle = float(trim2_param) % (2 * np.pi)

                                            # Ensure the start angle and end angle are ordered correctly
                                            if end_angle < start_angle:
                                                end_angle += 2 * np.pi

                                            # Append arc data with ordered angles and convex check
                                            arc_data.append({
                                                "center": center,
                                                "radius": radius,
                                                "start_angle": start_angle,
                                                "end_angle": end_angle,
                                                "trim1": trim1_coords if 'trim1_coords' in locals() else None,
                                                "trim2": trim2_coords if 'trim2_coords' in locals() else None,
                                                "sense_agreement": trimmed_curve.SenseAgreement
                                            })

    # Sort arcs by radius to ensure one is outer and the other is inner
    if len(arc_data) == 2:
        arc_data = sorted(arc_data, key=lambda arc: arc['radius'], reverse=True)
    return arc_data

# Function to check if two arcs have the same direction
def is_same_direction(arc1, arc2):
    return arc1["sense_agreement"] == arc2["sense_agreement"]

# Function to plot arc segments
def plot_arc(ax, center, radius, start_angle, end_angle, color='blue', label=None):
    if end_angle <= start_angle:
        end_angle += 2 * np.pi
    theta = np.linspace(start_angle, end_angle, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color=color, label=label)
    return (x[0], y[0]), (x[-1], y[-1])

# Function to calculate precise mid-axis between two arcs
def calculate_precise_mid_axis(outer_arc, inner_arc):
    # Ensure angles are in the correct range
    start_angle = outer_arc["start_angle"]
    end_angle = outer_arc["end_angle"]
    if end_angle <= start_angle:
        end_angle += 2 * np.pi
    mid_radius = (outer_arc["radius"] + inner_arc["radius"]) / 2
    mid_theta = np.linspace(start_angle, end_angle, 100)
    mid_x = outer_arc["center"][0] + mid_radius * np.cos(mid_theta)
    mid_y = outer_arc["center"][1] + mid_radius * np.sin(mid_theta)
    mid_axis_line = LineString([(mid_x[i], mid_y[i]) for i in range(len(mid_x))])
    return mid_axis_line

# Function to extend and shorten curved wall axis radially
def extend_and_shorten_arc_axis(arc_axis, extend_angle=0.1):
    # Extend the start and end angles by a small angle (radians)
    start_angle = arc_axis["start_angle"] - extend_angle
    end_angle = arc_axis["end_angle"] + extend_angle
    return start_angle, end_angle

# Function to calculate robust end-lines for curved walls
def get_end_lines_for_arcs(outer_arc, inner_arc):
    # Calculate the start and end points for both arcs based on angles
    outer_start = (
        outer_arc["center"][0] + outer_arc["radius"] * np.cos(outer_arc["start_angle"]),
        outer_arc["center"][1] + outer_arc["radius"] * np.sin(outer_arc["start_angle"])
    )
    outer_end = (
        outer_arc["center"][0] + outer_arc["radius"] * np.cos(outer_arc["end_angle"]),
        outer_arc["center"][1] + outer_arc["radius"] * np.sin(outer_arc["end_angle"])
    )
    inner_start = (
        inner_arc["center"][0] + inner_arc["radius"] * np.cos(inner_arc["start_angle"]),
        inner_arc["center"][1] + inner_arc["radius"] * np.sin(inner_arc["start_angle"])
    )
    inner_end = (
        inner_arc["center"][0] + inner_arc["radius"] * np.cos(inner_arc["end_angle"]),
        inner_arc["center"][1] + inner_arc["radius"] * np.sin(inner_arc["end_angle"])
    )

    # Define the end-lines, start-to-start and end-to-end
    end_line_1 = LineString([outer_start, inner_start])
    end_line_2 = LineString([outer_end, inner_end])

    return end_line_1, end_line_2

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

# Extract wall data and classify as straight or curved
straight_wall_geometries = {}
curved_wall_geometries = {}
wall_axes = {}

for wall in ifc_file.by_type("IfcWall") + ifc_file.by_type("IfcWallStandardCase"):
    arc_info = extract_arc_data_with_transform(wall)
    if arc_info:
        curved_wall_geometries[wall.GlobalId] = arc_info
    else:
        straight_wall_points = extract_straight_wall_points(wall)
        if straight_wall_points:
            straight_wall_geometries[wall.GlobalId] = straight_wall_points

# Process and plot straight walls
for wall_id, wall_points in straight_wall_geometries.items():
    if len(wall_points) > 1:
        # Plot wall profile
        x_coords = [point[0] for point in wall_points]
        y_coords = [point[1] for point in wall_points]
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        ax.plot(x_coords, y_coords, 'g--', linewidth=1,
                label="Straight Wall Profile" if wall_id == list(straight_wall_geometries.keys())[0] else "")

        # Create a Polygon object to represent the wall profile for intersection calculations
        wall_polygon = Polygon(wall_points)

        # Get wall length or use default
        wall_length = get_wall_length(ifc_file.by_guid(wall_id)) or 10  # Fallback to default length if not found

        # Calculate midpoints of opposite edges to define axis orientation
        midpoint_start = ((wall_points[0][0] + wall_points[len(wall_points) // 2][0]) / 2,
                          (wall_points[0][1] + wall_points[len(wall_points) // 2][1]) / 2)
        midpoint_end = ((wall_points[1][0] + wall_points[(len(wall_points) // 2) + 1][0]) / 2,
                        (wall_points[1][1] + wall_points[(len(wall_points) // 2) + 1][1]) / 2)

        # Compute direction vector
        direction_vector = (midpoint_end[0] - midpoint_start[0], midpoint_end[1] - midpoint_start[1])
        direction_length = np.hypot(*direction_vector)
        normalized_direction_vector = (direction_vector[0] / direction_length, direction_vector[1] / direction_length)

        # Extend the axis beyond the wall length
        extended_start = (
            midpoint_start[0] - (wall_length / 2) * normalized_direction_vector[0],
            midpoint_start[1] - (wall_length / 2) * normalized_direction_vector[1]
        )
        extended_end = (
            midpoint_end[0] + (wall_length / 2) * normalized_direction_vector[0],
            midpoint_end[1] + (wall_length / 2) * normalized_direction_vector[1]
        )
        extended_axis = LineString([extended_start, extended_end])

        # Shorten the axis to match wall end faces by finding intersections
        intersection_points = extended_axis.intersection(wall_polygon)

        # Adjust axis based on intersection points with end faces
        if isinstance(intersection_points, MultiPoint) and len(intersection_points) >= 2:
            points = sorted(intersection_points, key=lambda p: (p.x, p.y))
            shortened_axis = LineString([points[0], points[-1]])  # Shortened to within end faces
        elif isinstance(intersection_points, LineString):
            shortened_axis = intersection_points  # If a single LineString intersection is found
        else:
            print(f"No valid intersection found for wall axis: {wall_id}")
            shortened_axis = extended_axis  # Default to extended axis if no intersection

        # Plot the shortened axis
        ax.plot(*shortened_axis.xy, 'b-', linewidth=2,
                label="Straight Wall Axis" if wall_id == list(straight_wall_geometries.keys())[0] else "")
        wall_axes[wall_id] = shortened_axis  # Store the shortened axis for indexing
    else:
        continue  # Skip walls with insufficient points

# Process and plot curved walls with mid-axes and robust end-lines
for wall_id, arcs in curved_wall_geometries.items():
    if len(arcs) >= 2:
        outer_arc = arcs[0]
        inner_arc = arcs[1]

        # Ensure inner and outer arcs have matching direction
        if not is_same_direction(outer_arc, inner_arc):
            inner_arc["start_angle"], inner_arc["end_angle"] = inner_arc["end_angle"], inner_arc["start_angle"]

        # Plot outer and inner arcs
        color_outer = 'blue' if outer_arc['sense_agreement'] else 'red'
        plot_arc(ax, outer_arc["center"], outer_arc["radius"], outer_arc["start_angle"], outer_arc["end_angle"],
                 color=color_outer)

        color_inner = 'green' if inner_arc['sense_agreement'] else 'purple'
        plot_arc(ax, inner_arc["center"], inner_arc["radius"], inner_arc["start_angle"], inner_arc["end_angle"],
                 color=color_inner)

        # Robust end-line calculation
        end_line_1, end_line_2 = get_end_lines_for_arcs(outer_arc, inner_arc)

        # Plot the end-lines
        ax.plot(*end_line_1.xy, color='black', linewidth=2)
        ax.plot(*end_line_2.xy, color='black', linewidth=2)

        # Mid-axis for curved wall
        mid_axis = calculate_precise_mid_axis(outer_arc, inner_arc)

        # Extend the axis radially
        extend_angle = 0.1  # Adjust as needed
        extended_start_angle, extended_end_angle = extend_and_shorten_arc_axis({
            "start_angle": outer_arc["start_angle"],
            "end_angle": outer_arc["end_angle"]
        }, extend_angle=extend_angle)

        # Create extended mid-axis as an arc
        extended_mid_theta = np.linspace(extended_start_angle, extended_end_angle, 100)
        mid_radius = (outer_arc["radius"] + inner_arc["radius"]) / 2
        mid_x = outer_arc["center"][0] + mid_radius * np.cos(extended_mid_theta)
        mid_y = outer_arc["center"][1] + mid_radius * np.sin(extended_mid_theta)
        extended_mid_axis = LineString([(mid_x[i], mid_y[i]) for i in range(len(mid_x))])

        # Calculate the midpoint of the original extended axis for reference
        midpoint_index = len(extended_mid_theta) // 2
        midpoint_original = Point(mid_x[midpoint_index], mid_y[midpoint_index])

        # Shorten the extended mid-axis by finding intersections with end-lines
        intersection_points_1 = extended_mid_axis.intersection(end_line_1)
        intersection_points_2 = extended_mid_axis.intersection(end_line_2)

        # Ensure we maintain the middle point by choosing the correct arc section
        if isinstance(intersection_points_1, Point) and isinstance(intersection_points_2, Point):
            # Calculate angles for intersection points to redefine arc range
            angle_start = calculate_angle(outer_arc["center"], (intersection_points_1.x, intersection_points_1.y))
            angle_end = calculate_angle(outer_arc["center"], (intersection_points_2.x, intersection_points_2.y))

            # Ensure start and end angles are in the correct order
            if angle_start > angle_end:
                angle_start, angle_end = angle_end, angle_start

            # Create candidate arcs for both directions to validate against the midpoint
            shortened_theta_forward = np.linspace(angle_start, angle_end, 100)
            shortened_theta_backward = np.linspace(angle_end, angle_start + 2 * np.pi, 100)

            # Generate points for forward and backward arcs
            shortened_mid_x_forward = outer_arc["center"][0] + mid_radius * np.cos(shortened_theta_forward)
            shortened_mid_y_forward = outer_arc["center"][1] + mid_radius * np.sin(shortened_theta_forward)
            shortened_axis_forward = LineString([(shortened_mid_x_forward[i], shortened_mid_y_forward[i]) 
                                                 for i in range(len(shortened_mid_x_forward))])

            shortened_mid_x_backward = outer_arc["center"][0] + mid_radius * np.cos(shortened_theta_backward)
            shortened_mid_y_backward = outer_arc["center"][1] + mid_radius * np.sin(shortened_theta_backward)
            shortened_axis_backward = LineString([(shortened_mid_x_backward[i], shortened_mid_y_backward[i]) 
                                                  for i in range(len(shortened_mid_x_backward))])

            # Select the arc containing the midpoint of the extended axis
            if shortened_axis_forward.distance(midpoint_original) < shortened_axis_backward.distance(midpoint_original):
                shortened_axis = shortened_axis_forward
            else:
                shortened_axis = shortened_axis_backward
        else:
            # Fallback if intersections aren't found
            print(f"No valid intersections found for mid-axis on wall {wall_id}")
            shortened_axis = extended_mid_axis

        # Plot the shortened curved mid-axis
        ax.plot(*shortened_axis.xy, 'b--', linewidth=2,
                label="Shortened Curved Wall Axis" if wall_id == list(curved_wall_geometries.keys())[0] else "")
        wall_axes[wall_id] = shortened_axis  # Store the shortened axis for indexing
    else:
        # Handle cases with only one arc (e.g., half-circle walls)
        arc = arcs[0]
        color_arc = 'blue' if arc['sense_agreement'] else 'red'
        plot_arc(ax, arc["center"], arc["radius"], arc["start_angle"], arc["end_angle"], color=color_arc)

        # Mid-axis at half the radius
        mid_radius = arc["radius"] / 2
        mid_theta = np.linspace(arc["start_angle"], arc["end_angle"], 100)
        mid_x = arc["center"][0] + mid_radius * np.cos(mid_theta)
        mid_y = arc["center"][1] + mid_radius * np.sin(mid_theta)
        mid_axis = LineString([(mid_x[i], mid_y[i]) for i in range(len(mid_x))])

        # Extend and shorten the axis radially
        extend_angle = 0.1  # Adjust as needed
        extended_start_angle = arc["start_angle"] - extend_angle
        extended_end_angle = arc["end_angle"] + extend_angle

        # Create extended mid-axis
        extended_mid_theta = np.linspace(extended_start_angle, extended_end_angle, 100)
        extended_mid_x = arc["center"][0] + mid_radius * np.cos(extended_mid_theta)
        extended_mid_y = arc["center"][1] + mid_radius * np.sin(extended_mid_theta)
        extended_mid_axis = LineString([(extended_mid_x[i], extended_mid_y[i]) for i in range(len(extended_mid_x))])

        # No end-lines in single arc case, plot extended mid-axis directly
        ax.plot(extended_mid_x, extended_mid_y, 'b--',
                label="Curved Wall Axis" if wall_id == list(curved_wall_geometries.keys())[0] else "")
        wall_axes[wall_id] = extended_mid_axis  # Store the axis for indexing

# Set plot properties
plt.title("Aligned Wall Profiles with Mid-Axes")
plt.xlabel("X-axis (meters)")
plt.ylabel("Y-axis (meters)")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Initialize wall connectivity dictionary from IFC relationships
wall_connectivity = defaultdict(list)

# Extract relationships from the IFC file to establish connectivity
for rel in ifc_file.by_type("IfcRelConnectsElements"):
    relating_wall = rel.RelatingElement
    related_wall = rel.RelatedElement
    if relating_wall.is_a("IfcWall") and related_wall.is_a("IfcWall"):
        # Add each wall's connection to the other's neighbor list
        wall_connectivity[relating_wall.GlobalId].append(related_wall.GlobalId)
        wall_connectivity[related_wall.GlobalId].append(relating_wall.GlobalId)

# Sequentially index walls based on IFC-defined connectivity
ordered_walls = []
visited_walls = set()

# Pick an arbitrary starting wall and build the sequence based on connectivity
if wall_connectivity:
    starting_wall = next(iter(wall_connectivity))  # Begin with the first wall in connectivity data
    current_wall = starting_wall

    while current_wall:
        ordered_walls.append(current_wall)
        visited_walls.add(current_wall)

        # Find the next wall in sequence from unvisited neighbors
        next_wall = None
        for neighbor in wall_connectivity[current_wall]:
            if neighbor not in visited_walls:
                next_wall = neighbor
                break

        # Move to the next wall
        current_wall = next_wall

# Ensure a fallback in case there are no connections
if not ordered_walls:
    ordered_walls = list(wall_axes.keys())

# Randomize the starting index for visualization consistency
random.seed(33)
start_index = random.randint(0, len(ordered_walls) - 1)
ordered_walls = ordered_walls[start_index:] + ordered_walls[:start_index]

# Visualization of indexed wall axes with sequential ordering
plt.figure(figsize=(10, 10))
for idx, wall_id in enumerate(ordered_walls):
    if wall_id in wall_axes:
        axis = wall_axes[wall_id]
        x_coords, y_coords = axis.xy

        # Plot the wall axis
        plt.plot(x_coords, y_coords, 'b-', linewidth=2)

        # Calculate midpoint for label positioning
        midpoint_index = len(x_coords) // 2
        midpoint = ((x_coords[0] + x_coords[-1]) / 2, (y_coords[0] + y_coords[-1]) / 2)

        # Place index label at the midpoint
        plt.text(midpoint[0], midpoint[1], str(idx + 1), fontsize=12, color='red', fontweight='bold', ha='center')

# Set plot properties
plt.title("Indexed Wall Axes with Sequential Ordering")
plt.xlabel("X-axis (meters)")
plt.ylabel("Y-axis (meters)")
plt.grid(True)
plt.axis('equal')
plt.legend(['Wall Axes (blue solid)'], loc='upper right')
plt.show()





# Dictionary to store the original axes before any extension, with sequential index and wall ID
original_axes_before_extension = {}

# Populate the dictionary with the original shortened axes for straight walls
for idx, wall_id in enumerate(ordered_walls):
    if wall_id in straight_wall_geometries:
        # Retrieve the shortened axis for this wall
        original_line = wall_axes[wall_id]
        original_axes_before_extension[wall_id] = {
            "index": idx + 1,
            "type": "line",
            "geometry": original_line
        }

# Populate the dictionary with the original shortened axes for curved walls
for idx, wall_id in enumerate(ordered_walls):
    if wall_id in curved_wall_geometries:
        # Retrieve the shortened mid-axis for this curved wall
        original_circle = wall_axes[wall_id]
        original_axes_before_extension[wall_id] = {
            "index": idx + 1,
            "type": "circle",
            "geometry": original_circle
        }

# Optional: print contents to verify
print("\nContents of original_axes_before_extension:")
for wall_id, data in original_axes_before_extension.items():
    if data["type"] == "line":
        print(f"Wall ID: {wall_id}, Index: {data['index']}, Type: Line, Geometry: {list(data['geometry'].coords)}")
    elif data["type"] == "circle":
        print(f"Wall ID: {wall_id}, Index: {data['index']}, Type: Circle, Geometry Center: {data['geometry'].centroid}")








from shapely.geometry import LineString, Point
import numpy as np

# Initialize groups based on sequential pairing with intersection points
sequential_groups = {}

# Sort the original dictionary by index to ensure sequential order
sorted_items = sorted(original_axes_before_extension.items(), key=lambda item: item[1]["index"])
n = len(sorted_items)

# Iterate through pairs to calculate intersection points and compare endpoints
for i in range(n):
    current_item = sorted_items[i]
    next_item = sorted_items[(i + 1) % n]  # Wrap-around for the last element to pair with the first

    # Extract wall IDs, data, and type information for the pair of elements
    wall_id_1, data_1 = current_item
    wall_id_2, data_2 = next_item

    # Initialize group with wall IDs, indexes, and types
    group_id = f"Group_{i + 1}"
    sequential_groups[group_id] = {
        "element_1": {
            "wall_id": wall_id_1,
            "index": data_1["index"],
            "type": data_1["type"],
            "geometry": data_1["geometry"]
        },
        "element_2": {
            "wall_id": wall_id_2,
            "index": data_2["index"],
            "type": data_2["type"],
            "geometry": data_2["geometry"]
        },
        "intersection": {
            "points": None,  # Placeholder for intersection point(s)
            "most_similar_ends": None,  # Placeholder for most similar endpoints
            "average_coordinates": None  # Placeholder for average coordinates of most similar ends
        }
    }

    # Determine the geometry types of the elements
    geom_1 = data_1["geometry"]
    geom_2 = data_2["geometry"]

    # Extract endpoints for comparison
    if isinstance(geom_1, LineString):
        start_1, end_1 = Point(geom_1.coords[0]), Point(geom_1.coords[-1])
    else:
        arc_center_1 = Point(geom_1["center"])
        arc_radius_1 = geom_1["radius"]
        arc_start_angle_1 = geom_1["start_angle"]
        arc_end_angle_1 = geom_1["end_angle"]
        start_1 = Point(arc_center_1.x + arc_radius_1 * np.cos(arc_start_angle_1),
                        arc_center_1.y + arc_radius_1 * np.sin(arc_start_angle_1))
        end_1 = Point(arc_center_1.x + arc_radius_1 * np.cos(arc_end_angle_1),
                      arc_center_1.y + arc_radius_1 * np.sin(arc_end_angle_1))

    if isinstance(geom_2, LineString):
        start_2, end_2 = Point(geom_2.coords[0]), Point(geom_2.coords[-1])
    else:
        arc_center_2 = Point(geom_2["center"])
        arc_radius_2 = geom_2["radius"]
        arc_start_angle_2 = geom_2["start_angle"]
        arc_end_angle_2 = geom_2["end_angle"]
        start_2 = Point(arc_center_2.x + arc_radius_2 * np.cos(arc_start_angle_2),
                        arc_center_2.y + arc_radius_2 * np.sin(arc_start_angle_2))
        end_2 = Point(arc_center_2.x + arc_radius_2 * np.cos(arc_end_angle_2),
                      arc_center_2.y + arc_radius_2 * np.sin(arc_end_angle_2))

    # Calculate the distances between each pair of endpoints
    distances = {
        "StartA-StartB": start_1.distance(start_2),
        "StartA-EndB": start_1.distance(end_2),
        "EndA-StartB": end_1.distance(start_2),
        "EndA-EndB": end_1.distance(end_2)
    }

    # Identify the most similar (closest) endpoints
    most_similar = min(distances, key=distances.get)

    # Calculate the average coordinates of the most similar endpoints
    if most_similar == "StartA-StartB":
        average_coords = ((start_1.x + start_2.x) / 2, (start_1.y + start_2.y) / 2)
    elif most_similar == "StartA-EndB":
        average_coords = ((start_1.x + end_2.x) / 2, (start_1.y + end_2.y) / 2)
    elif most_similar == "EndA-StartB":
        average_coords = ((end_1.x + start_2.x) / 2, (end_1.y + start_2.y) / 2)
    elif most_similar == "EndA-EndB":
        average_coords = ((end_1.x + end_2.x) / 2, (end_1.y + end_2.y) / 2)

    # Store the most similar endpoint pair and average coordinates in the dictionary
    sequential_groups[group_id]["intersection"]["most_similar_ends"] = {
        "pair": most_similar,
        "distance": distances[most_similar]
    }
    sequential_groups[group_id]["intersection"]["average_coordinates"] = average_coords

    # Calculate intersection based on geometry types
    if isinstance(geom_1, LineString) and isinstance(geom_2, LineString):
        # Intersection of two line segments
        intersection = geom_1.intersection(geom_2)
        if not intersection.is_empty:
            sequential_groups[group_id]["intersection"]["points"] = intersection

    elif isinstance(geom_1, LineString) and isinstance(geom_2, dict):  # Line and Arc
        arc_segment = LineString([start_2, end_2])
        intersection = geom_1.intersection(arc_segment)

        if not intersection.is_empty:
            sequential_groups[group_id]["intersection"]["points"] = (
                list(intersection.geoms) if intersection.geom_type == 'MultiPoint' else [intersection]
            )

    elif isinstance(geom_1, dict) and isinstance(geom_2, LineString):  # Arc and Line
        arc_segment = LineString([start_1, end_1])
        intersection = arc_segment.intersection(geom_2)

        if not intersection.is_empty:
            sequential_groups[group_id]["intersection"]["points"] = (
                list(intersection.geoms) if intersection.geom_type == 'MultiPoint' else [intersection]
            )

    elif isinstance(geom_1, dict) and isinstance(geom_2, dict):  # Arc and Arc
        arc_segment_1 = LineString([start_1, end_1])
        arc_segment_2 = LineString([start_2, end_2])
        intersection = arc_segment_1.intersection(arc_segment_2)

        if not intersection.is_empty:
            sequential_groups[group_id]["intersection"]["points"] = (
                list(intersection.geoms) if intersection.geom_type == 'MultiPoint' else [intersection]
            )

# Optional: Print the results to verify intersections, similar endpoints, and average coordinates
print("\nContents of sequential_groups with intersection points, most similar endpoints, and average coordinates:")
for group_id, group_data in sequential_groups.items():
    print(f"{group_id}:")
    print(f"  Element 1 - Wall ID: {group_data['element_1']['wall_id']}, "
          f"Index: {group_data['element_1']['index']}, "
          f"Type: {group_data['element_1']['type']}")
    print(f"  Element 2 - Wall ID: {group_data['element_2']['wall_id']}, "
          f"Index: {group_data['element_2']['index']}, "
          f"Type: {group_data['element_2']['type']}")
    print(f"  Intersection Points: {group_data['intersection']['points']}")
    print(f"  Most Similar Ends: {group_data['intersection']['most_similar_ends']}")
    print(f"  Average Coordinates of Most Similar Ends: {group_data['intersection']['average_coordinates']}\n")







# Define a dictionary to store all the data extracted and processed in cell(1)
data_store = {
    "straight_wall_geometries": straight_wall_geometries,
    "curved_wall_geometries": curved_wall_geometries,
    "wall_axes": wall_axes,
    "wall_connectivity": wall_connectivity,
    "ordered_walls": ordered_walls,
    "original_axes_before_extension": original_axes_before_extension,
    "sequential_groups": sequential_groups
}

# Optional: print contents of the dictionary to verify correctness
print("\nContents of data_store:")
for key, value in data_store.items():
    print(f"{key}: {'(Too large to display)' if len(str(value)) > 100 else value}")









#generation solid shape out of it (was quite challenging)



def generate_solid_shape(random_seed=None, ax=None, return_plot_data=False):
    import numpy as np
    from shapely.geometry import Point, LineString, MultiPoint, LinearRing, Polygon, MultiPolygon
    from shapely.affinity import scale
    from shapely.validation import make_valid
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import random
    import math

    # Set the random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)


    import numpy as np
    from shapely.geometry import Point, LineString
    import matplotlib.pyplot as plt
    
    
    
    
    
    
    # Initialize containers for straight and curved wall properties and original geometries
    straight_wall_data = []
    curved_wall_data = []
    original_line_axes = {}    # Dictionary to store original line axes for later reference
    original_circle_axes = {}  # Dictionary to store original circle axes for later reference
    
    # Process straight walls
    for wall_id, wall_points in straight_wall_geometries.items():
        if len(wall_points) > 1:
            # Use the first and middle points to define direction
            start_point = wall_points[0]
            mid_index = len(wall_points) // 2
            end_point = wall_points[mid_index]
    
            # Calculate midpoint between start and end points
            midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
    
            # Calculate direction vector
            direction_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
            direction_length = np.hypot(*direction_vector)
    
            if direction_length == 0:
                normalized_direction_vector = None  # If direction vector length is zero
            else:
                normalized_direction_vector = (
                    direction_vector[0] / direction_length, direction_vector[1] / direction_length
                )
    
            # Store original line data with axis as a LineString
            original_line_axes[wall_id] = LineString([start_point, end_point])
    
            # Store straight wall data for potential later modification
            straight_wall_data.append({
                "wall_id": wall_id,
                "midpoint": midpoint,
                "direction_vector": normalized_direction_vector
            })
    
    # Process curved walls
    for wall_id, arcs in curved_wall_geometries.items():
        if len(arcs) >= 2:
            outer_arc = arcs[0]
            inner_arc = arcs[1]
    
            # Store original full circle data as center and radius for reference
            original_circle_axes[wall_id] = {
                "center": outer_arc["center"],
                "radius": outer_arc["radius"]
            }
    
            # Calculate the midpoint of the arc segment
            start_angle = outer_arc["start_angle"]
            end_angle = outer_arc["end_angle"]
            mid_angle = (start_angle + end_angle) / 2
    
            # Calculate midpoint position on the arc
            mid_x = outer_arc["center"][0] + outer_arc["radius"] * np.cos(mid_angle)
            mid_y = outer_arc["center"][1] + outer_arc["radius"] * np.sin(mid_angle)
            midpoint = (mid_x, mid_y)
    
            # Store curved wall data for potential later modification
            curved_wall_data.append({
                "wall_id": wall_id,
                "center": outer_arc["center"],
                "radius": outer_arc["radius"],
                "midpoint": midpoint
            })
    
    
    
    
    
    from shapely.affinity import scale
    
    # Dictionary to store intersections between each element and its immediate neighbors
    original_intersections = {}
    
    # Loop through each wall in sequential order based on `ordered_walls`
    for i in range(len(ordered_walls)):
        current_wall_id = ordered_walls[i]
        current_element = original_line_axes.get(current_wall_id, original_circle_axes.get(current_wall_id))
        
        # Extend each element by a factor of 3 for intersection exploration
        if isinstance(current_element, LineString):
            extended_current_element = scale(current_element, xfact=3, yfact=3, origin='center')
        else:  # For circles, create an extended boundary based on the radius
            extended_current_element = Point(current_element["center"]).buffer(current_element["radius"]).boundary
    
        # Determine neighboring indices with wrap-around behavior
        next_index = (i + 1) % len(ordered_walls)
        prev_index = (i - 1) if i > 0 else len(ordered_walls) - 1
    
        # Next neighbor
        next_wall_id = ordered_walls[next_index]
        next_element = original_line_axes.get(next_wall_id, original_circle_axes.get(next_wall_id))
        if isinstance(next_element, LineString):
            extended_next_element = scale(next_element, xfact=3, yfact=3, origin='center')
        else:
            extended_next_element = Point(next_element["center"]).buffer(next_element["radius"]).boundary
    
        # Calculate intersections with the next element
        next_intersection = extended_current_element.intersection(extended_next_element)
        next_points = list(next_intersection.geoms) if isinstance(next_intersection, MultiPoint) else [next_intersection]
    
        # Previous neighbor
        prev_wall_id = ordered_walls[prev_index]
        prev_element = original_line_axes.get(prev_wall_id, original_circle_axes.get(prev_wall_id))
        if isinstance(prev_element, LineString):
            extended_prev_element = scale(prev_element, xfact=3, yfact=3, origin='center')
        else:
            extended_prev_element = Point(prev_element["center"]).buffer(prev_element["radius"]).boundary
    
        # Calculate intersections with the previous element
        prev_intersection = extended_current_element.intersection(extended_prev_element)
        prev_points = list(prev_intersection.geoms) if isinstance(prev_intersection, MultiPoint) else [prev_intersection]
    
        # Store intersections in the dictionary
        original_intersections[current_wall_id] = {
            "next_neighbor": {"id": next_wall_id, "points": next_points},
            "prev_neighbor": {"id": prev_wall_id, "points": prev_points}
        }
    
    
    
    # Loop through each wall to check intersection points and compare Y coordinates if there are 2 points
    for wall_id, neighbors in original_intersections.items():
        
    
        # Process the next neighbor's intersection points
        next_neighbor_id = neighbors["next_neighbor"]["id"]
        next_points = neighbors["next_neighbor"]["points"]
    
        if len(next_points) == 2:
            # Compare Y-coordinates of the two intersection points
            if next_points[0].y > next_points[1].y:
                next_point1 = next_points[0]
                next_point2 = next_points[1]
                
            elif next_points[0].y < next_points[1].y:
                next_point1 = next_points[1]
                next_point2 = next_points[0]
                
            else:  # Y-coordinates are equal; compare X-coordinates
                if next_points[0].x < next_points[1].x:
                    next_point1 = next_points[0]
                    next_point2 = next_points[1]
                    
                else:
                    next_point1 = next_points[1]
                    next_point2 = next_points[0]
                    
    
        # Process the previous neighbor's intersection points
        prev_neighbor_id = neighbors["prev_neighbor"]["id"]
        prev_points = neighbors["prev_neighbor"]["points"]
    
        if len(prev_points) == 2:
            # Compare Y-coordinates of the two intersection points
            if prev_points[0].y > prev_points[1].y:
                prev_point1 = prev_points[0]
                prev_point2 = prev_points[1]
                
            elif prev_points[0].y < prev_points[1].y:
                prev_point1 = prev_points[1]
                prev_point2 = prev_points[0]
                
            else:  # Y-coordinates are equal; compare X-coordinates
                if prev_points[0].x < prev_points[1].x:
                    prev_point1 = prev_points[0]
                    prev_point2 = prev_points[1]
                    
                else:
                    prev_point1 = prev_points[1]
                    prev_point2 = prev_points[0]
                    
    
    
    
    from shapely.affinity import scale
    from shapely.geometry import LineString, Point, MultiPoint
    import numpy as np
    
    # Define extension factor for straight walls
    extension_factor = 3  # Adjust as necessary for visibility and intersection detection
    
    # Initialize groups based on sequential pairing with intersection points for Phase 2
    phase2_groups = {}
    
    # Total number of elements
    n = len(ordered_walls)
    
    # Iterate through pairs to calculate and store intersections by group
    for i in range(n):
        # Current and next items based on sequential indexes (wrap-around for last element)
        current_wall_id = ordered_walls[i]
        next_wall_id = ordered_walls[(i + 1) % n]
    
        # Retrieve and extend the geometries as necessary
        current_element = original_line_axes.get(current_wall_id, original_circle_axes.get(current_wall_id))
        next_element = original_line_axes.get(next_wall_id, original_circle_axes.get(next_wall_id))
    
        # Initialize the group
        group_id = f"Group_{i + 1}"
        phase2_groups[group_id] = {
            "element_1": {
                "wall_id": current_wall_id,
                "index": i + 1,
                "geometry": current_element
            },
            "element_2": {
                "wall_id": next_wall_id,
                "index": (i + 2) if i + 2 <= n else 1,
                "geometry": next_element
            },
            "intersection": []  # Placeholder for intersection points
        }
    
        # Prepare geometries for intersection by extending lines as needed
        if isinstance(current_element, LineString):
            geom_1 = scale(current_element, xfact=extension_factor, yfact=extension_factor, origin='center')
        else:
            center_1 = Point(current_element.get("center", (0, 0)))
            radius_1 = current_element.get("radius", 0)
            geom_1 = center_1.buffer(radius_1).boundary
    
        if isinstance(next_element, LineString):
            geom_2 = scale(next_element, xfact=extension_factor, yfact=extension_factor, origin='center')
        else:
            center_2 = Point(next_element.get("center", (0, 0)))
            radius_2 = next_element.get("radius", 0)
            geom_2 = center_2.buffer(radius_2).boundary
    
        # Determine types of the elements
        current_type = "circle" if isinstance(current_element, dict) and "center" in current_element else "line"
        next_type = "circle" if isinstance(next_element, dict) and "center" in next_element else "line"
    
        # Handle circle-to-circle tangency specifically
        if current_type == "circle" and next_type == "circle":
            center_1 = Point(current_element["center"])
            radius_1 = current_element["radius"]
            center_2 = Point(next_element["center"])
            radius_2 = next_element["radius"]
    
            distance_between_centers = center_1.distance(center_2)
    
            # Check for tangency (distance equals sum or absolute difference of radii)
            if np.isclose(distance_between_centers, radius_1 + radius_2, atol=1e-5) or \
               np.isclose(distance_between_centers, abs(radius_1 - radius_2), atol=1e-5):
                # Calculate tangent points
                direction_vector = np.array([center_2.x - center_1.x, center_2.y - center_1.y])
                direction_unit = direction_vector / np.linalg.norm(direction_vector)
                tangent_point_self = Point(center_1.x + direction_unit[0] * radius_1,
                                           center_1.y + direction_unit[1] * radius_1)
                tangent_point_neighbor = Point(center_2.x - direction_unit[0] * radius_2,
                                               center_2.y - direction_unit[1] * radius_2)
    
                # Add tangent points to the group
                phase2_groups[group_id]["intersection"] = [tangent_point_self, tangent_point_neighbor]
            else:
                # Standard intersection if not tangent
                intersection = geom_1.intersection(geom_2)
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        phase2_groups[group_id]["intersection"] = [intersection]
                    elif isinstance(intersection, MultiPoint):
                        phase2_groups[group_id]["intersection"] = list(intersection.geoms)
    
        # Handle line-to-circle tangency
        elif current_type == "line" and next_type == "circle":
            line = geom_1
            circle_center = Point(next_element["center"])
            radius = next_element["radius"]
    
            # Calculate perpendicular distance from line to circle center
            distance_to_center = line.distance(circle_center)
    
            # Check for tangency
            if np.isclose(distance_to_center, radius, atol=1e-5):
                # Calculate tangent point
                nearest_point = line.interpolate(line.project(circle_center))  # Closest point on line to circle center
                tangent_vector = np.array([circle_center.x - nearest_point.x, circle_center.y - nearest_point.y])
                tangent_unit = tangent_vector / np.linalg.norm(tangent_vector)
    
                tangent_point = Point(circle_center.x - tangent_unit[0] * radius,
                                      circle_center.y - tangent_unit[1] * radius)
    
                # Add tangent point to the group
                phase2_groups[group_id]["intersection"] = [tangent_point]
            else:
                # Standard intersection if not tangent
                intersection = geom_1.intersection(geom_2)
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        phase2_groups[group_id]["intersection"] = [intersection]
                    elif isinstance(intersection, MultiPoint):
                        phase2_groups[group_id]["intersection"] = list(intersection.geoms)
    
        # Handle all other combinations
        else:
            intersection = geom_1.intersection(geom_2)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    phase2_groups[group_id]["intersection"] = [intersection]
                elif isinstance(intersection, MultiPoint):
                    phase2_groups[group_id]["intersection"] = list(intersection.geoms)
    
        # Check if no intersection points were found for the group
        if not phase2_groups[group_id]["intersection"]:
            if current_type == "line" and next_type == "circle":
                # If line-circle, add the closest point on the line to the circle center as the tangent point
                nearest_point = geom_1.interpolate(geom_1.project(Point(next_element["center"])))
                phase2_groups[group_id]["intersection"] = [nearest_point]
    


            
    
    
    # Continuation of the existing code block
    
    # Loop through each group in Phase 2 to apply the intersection point comparison
    for group_id, group_data in phase2_groups.items():
        intersection_points = group_data["intersection"]
        
        # Check if there are exactly 2 intersection points to compare
        if len(intersection_points) == 2:
            point_1, point_2 = intersection_points
            
            # Compare Y-coordinates first
            if point_1.y > point_2.y:
                point_1_index, point_2_index = 1, 2
            elif point_1.y < point_2.y:
                point_1_index, point_2_index = 2, 1
            else:
                # If Y-coordinates are equal, compare X-coordinates
                if point_1.x > point_2.x:
                    point_1_index, point_2_index = 1, 2
                else:
                    point_1_index, point_2_index = 2, 1
    
            
    
    
    # Store which indexed point is considered the original intersection point in Phase 2
    original_point_index = {}  # Stores {group_id: 1 or 2}
    
    
    from shapely.geometry import Point
    import math
    
    # Iterate through each group and compare Phase-1 average coordinates with Phase-2 intersection points
    for group_id in phase2_groups.keys():
        # Retrieve the average intersection coordinates and intersection points from Phase 1
        phase1_group = sequential_groups.get(group_id, {})
        phase1_intersection = phase1_group.get("intersection", {})
        phase1_avg_coords = phase1_intersection.get("average_coordinates")
        phase2_intersections = phase2_groups[group_id]["intersection"]
    
        # Ensure we have average coordinates to compare
        if not phase1_avg_coords:
            
            continue
    
        # Initialize variables to find the original intersection point
        original_intersection_point = None
        min_distance = float('inf')
    
        # If only one intersection point in Phase 2
        if len(phase2_intersections) == 1:
            single_point = phase2_intersections[0]
            distance = math.hypot(single_point.x - phase1_avg_coords[0], single_point.y - phase1_avg_coords[1])
    
            # Output details
           
            # No need to add to `original_point_index` since there is only one point
    
        # If there are 2 intersection points, add indexing and find the closest to Phase-1 average coordinates
        elif len(phase2_intersections) == 2:
            
    
            # Extract the two points for indexing and distance comparison
            point_1, point_2 = phase2_intersections
    
            # Step 1: Assign index 1 or 2 based on Y and X coordinate comparisons
            if point_1.y > point_2.y:
                point_1_index, point_2_index = 1, 2
            elif point_1.y < point_2.y:
                point_1_index, point_2_index = 2, 1
            else:
                # If Y-coordinates are equal, compare X-coordinates
                if point_1.x > point_2.x:
                    point_1_index, point_2_index = 1, 2
                else:
                    point_1_index, point_2_index = 2, 1
    
            # Step 2: Calculate distances to the Phase-1 average coordinates
            distance_1 = math.hypot(point_1.x - phase1_avg_coords[0], point_1.y - phase1_avg_coords[1])
            distance_2 = math.hypot(point_2.x - phase1_avg_coords[0], point_2.y - phase1_avg_coords[1])
    
            # Print indexed points and distances
           
    
            # Step 3: Determine and print the original intersection point (closest to average coordinates)
            if distance_1 < distance_2:
                original_intersection_point = point_1
                original_point_index[group_id] = point_1_index  # Store "1" as the original point
                
            else:
                original_intersection_point = point_2
                original_point_index[group_id] = point_2_index  # Store "2" as the original point
                
    
    
    # Store which indexed point is considered the original intersection point in Phase 2
    original_point_index = {}  # Stores {group_id: 1 or 2}
    
    
    from shapely.geometry import Point
    import math
    
    # Iterate through each group and compare Phase-1 average coordinates with Phase-2 intersection points
    for group_id in phase2_groups.keys():
        # Retrieve the average intersection coordinates and intersection points from Phase 1
        phase1_group = sequential_groups.get(group_id, {})
        phase1_intersection = phase1_group.get("intersection", {})
        phase1_avg_coords = phase1_intersection.get("average_coordinates")
        phase2_intersections = phase2_groups[group_id]["intersection"]
    
        # Ensure we have average coordinates to compare
        if not phase1_avg_coords:
            
            continue
    
        # Initialize variables to find the original intersection point
        original_intersection_point = None
        min_distance = float('inf')
    
        # If only one intersection point in Phase 2
        if len(phase2_intersections) == 1:
            single_point = phase2_intersections[0]
            distance = math.hypot(single_point.x - phase1_avg_coords[0], single_point.y - phase1_avg_coords[1])
    
            
            # No need to add to `original_point_index` since there is only one point
    
        # If there are 2 intersection points, add indexing and find the closest to Phase-1 average coordinates
        elif len(phase2_intersections) == 2:
            
    
            # Extract the two points for indexing and distance comparison
            point_1, point_2 = phase2_intersections
    
            # Step 1: Assign index 1 or 2 based on Y and X coordinate comparisons
            if point_1.y > point_2.y:
                point_1_index, point_2_index = 1, 2
            elif point_1.y < point_2.y:
                point_1_index, point_2_index = 2, 1
            else:
                # If Y-coordinates are equal, compare X-coordinates
                if point_1.x > point_2.x:
                    point_1_index, point_2_index = 1, 2
                else:
                    point_1_index, point_2_index = 2, 1
    
            # Step 2: Calculate distances to the Phase-1 average coordinates
            distance_1 = math.hypot(point_1.x - phase1_avg_coords[0], point_1.y - phase1_avg_coords[1])
            distance_2 = math.hypot(point_2.x - phase1_avg_coords[0], point_2.y - phase1_avg_coords[1])
    
            
    
            # Step 3: Determine and print the original intersection point (closest to average coordinates)
            if distance_1 < distance_2:
                original_intersection_point = point_1
                original_point_index[group_id] = point_1_index  # Store "1" as the original point
                
            else:
                original_intersection_point = point_2
                original_point_index[group_id] = point_2_index  # Store "2" as the original point
                
    
    
    
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    import math
    
    
    
    # Colors for lines and intersection points
    line_color = 'blue'
    intersection_color = 'red'
    original_point_color = 'green'
    
    # Iterate through each group and compare Phase-1 average coordinates with Phase-2 intersection points
    for group_id in phase2_groups.keys():
        # Check for the existence of the "intersection" and "average_coordinates" keys in Phase 1
        phase1_group = sequential_groups.get(group_id, {})
        phase1_intersection = phase1_group.get("intersection", {})
        phase1_avg_coords = phase1_intersection.get("average_coordinates")
        phase2_intersections = phase2_groups[group_id]["intersection"]  # Assuming this is a list directly
    
        # Ensure we have average coordinates to compare
        if not phase1_avg_coords:
            
            continue
    
        # Initialize variables to find the closest original intersection point
        original_intersection_point = None
        min_distance = float('inf')
    
        # Plot each element's geometry (both lines and arcs)
        for element_key in ["element_1", "element_2"]:
            element = phase2_groups[group_id][element_key]
            geometry = element["geometry"]
    
            # Plot lines or arcs based on geometry type
            if isinstance(geometry, LineString):
                x, y = geometry.xy
                
            else:
                # Assuming an arc is represented with a center and radius
                center = geometry["center"]
                radius = geometry["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                
    
        # If only one intersection point in Phase 2, calculate distance to Phase 1 average coordinates
        if len(phase2_intersections) == 1:
            single_point = phase2_intersections[0]
            distance = math.hypot(single_point.x - phase1_avg_coords[0], single_point.y - phase1_avg_coords[1])
    
           
    
        # If there are 2 intersection points, find the closest to the Phase-1 average coordinates
        elif len(phase2_intersections) == 2:
           
    
            # Plot each intersection point and calculate its distance
            for point in phase2_intersections:
                distance = math.hypot(point.x - phase1_avg_coords[0], point.y - phase1_avg_coords[1])
               
                # Update the minimum distance and set the original intersection point
                if distance < min_distance:
                    min_distance = distance
                    original_intersection_point = point
    

    
    import random
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString, Point
    import numpy as np
    
    # Set a random seed for reproducibility of random movements
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Scale factor for random movement distance to ensure visibility
    movement_scale = 1000
    
    
    
    # Step 1: Map the inherited indexes from `ordered_walls` to each wall in `straight_wall_data` and `curved_wall_data`
    wall_index_map = {wall_id: idx + 1 for idx, wall_id in enumerate(ordered_walls)}
    
    # Assign the inherited indexes to both straight walls and curved walls
    for data in straight_wall_data + curved_wall_data:
        wall_id = data["wall_id"]
        if wall_id in wall_index_map:
            data["index"] = wall_index_map[wall_id]
    
    # Track the modified elements (lines and circles) with inherited indexing
    modified_axes = []
    adapted_midpoints = {}  # Dictionary to store the adapted midpoints for arcs
    
    # Part 1: Modify straight walls by moving the orientation point and rebuilding the axis
    for data in straight_wall_data:
        wall_id = data["wall_id"]
        orientation_point = data["midpoint"]
        direction_vector = data["direction_vector"]
        index = data["index"]  # Use inherited sequential index
    
        if direction_vector:
            # Move the orientation point randomly
            new_orientation_point = (
                orientation_point[0] + random.uniform(-movement_scale, movement_scale),
                orientation_point[1] + random.uniform(-movement_scale, movement_scale)
            )
    
            # Extend the new axis from the moved orientation point in both directions
            extension_length = 5000  # Original length or scale factor for visibility
            extended_start = (
                new_orientation_point[0] - extension_length * direction_vector[0],
                new_orientation_point[1] - extension_length * direction_vector[1]
            )
            extended_end = (
                new_orientation_point[0] + extension_length * direction_vector[0],
                new_orientation_point[1] + extension_length * direction_vector[1]
            )
            new_axis = LineString([extended_start, extended_end])
    
           
            
    
            # Label the axis with its index at the midpoint for visualization
            midpoint = ((extended_start[0] + extended_end[0]) / 2, (extended_start[1] + extended_end[1]) / 2)
           
    
            # Append to modified axes with original index
            modified_axes.append({
                "wall_id": wall_id,
                "index": index,
                "type": "line",
                "axis": new_axis,
                "midpoint": new_orientation_point,
                "direction_vector": direction_vector
            })
    
    # Part 2: Modify arcs by moving the center and midpoint in the same direction and distance
    for data in curved_wall_data:
        wall_id = data["wall_id"]
        center = data["center"]
        radius = data["radius"]
        arc_midpoint = data["midpoint"]
        index = data["index"]  # Use inherited sequential index
    
        # Generate a random movement vector
        movement_vector = (
            random.uniform(-movement_scale, movement_scale),
            random.uniform(-movement_scale, movement_scale)
        )
    
        # Apply the movement vector to both the center and midpoint
        new_center = (center[0] + movement_vector[0], center[1] + movement_vector[1])
        new_midpoint = (arc_midpoint[0] + movement_vector[0], arc_midpoint[1] + movement_vector[1])
    
        # Rebuild the full circle based on the moved center and original radius
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = new_center[0] + radius * np.cos(theta)
        circle_y = new_center[1] + radius * np.sin(theta)
    
       
    
        
    
        # Append to modified axes with original index
        modified_axes.append({
            "wall_id": wall_id,
            "index": index,
            "type": "circle",
            "center": new_center,
            "radius": radius,
            "midpoint": new_midpoint
        })
    
    # Sort modified axes based on inherited sequential index
    modified_axes.sort(key=lambda x: x["index"])
    
    # Part 3: Adaptation1 - Apply radius adjustments according to the specified rules
    for i, element in enumerate(modified_axes):
        if element["type"] != "circle":
            continue  # Skip lines; only apply adaptation to circles
    
        current_index = element["index"]
        current_center = Point(element["center"])
        current_radius = element["radius"]
    
        # Determine neighbor indices (wrap-around for first and last indices)
        next_index = (i + 1) % len(modified_axes)
        prev_index = i - 1 if i > 0 else len(modified_axes) - 1
        next_neighbor = modified_axes[next_index]
        prev_neighbor = modified_axes[prev_index]
    
        # Initialize neighbor types
        line_neighbors = []
        circle_neighbors = []
    
        # Check each neighbor's type and add accordingly
        for neighbor in [next_neighbor, prev_neighbor]:
            if neighbor["type"] == "circle":
                circle_neighbors.append(neighbor)
            else:
                line_neighbors.append(neighbor)
    
        # Case 1: Two Neighboring Lines
        if len(line_neighbors) == 2:
            max_perpendicular_distance = 0
            for line_neighbor in line_neighbors:
                line_point = np.array(line_neighbor["midpoint"])
                direction_vector = np.array(line_neighbor["direction_vector"])
    
                # Calculate perpendicular distance
                direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
                center_vector = np.array(element["center"])
                perpendicular_vector = center_vector - line_point
                perpendicular_distance = np.abs(np.cross(direction_vector_normalized, perpendicular_vector))
    
                # Update maximum perpendicular distance
                max_perpendicular_distance = max(max_perpendicular_distance, perpendicular_distance)
    
            # Adjust radius if needed
            if current_radius < max_perpendicular_distance:
                element["radius"] = max_perpendicular_distance
    
        # Case 2: One Neighboring Line and One Neighboring Circle
        elif len(line_neighbors) == 1 and len(circle_neighbors) == 1:
            line_neighbor = line_neighbors[0]
            circle_neighbor = circle_neighbors[0]
    
            # Calculate the perpendicular distance to the line
            line_point = np.array(line_neighbor["midpoint"])
            direction_vector = np.array(line_neighbor["direction_vector"])
    
            # Perpendicular distance to the line
            direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
            center_vector = np.array(element["center"])
            perpendicular_vector = center_vector - line_point
            perpendicular_distance = np.abs(np.cross(direction_vector_normalized, perpendicular_vector))
    
            # Calculate "a" for the circle neighbor
            neighbor_center = Point(circle_neighbor["center"])
            neighbor_radius = circle_neighbor["radius"]
            distance_between_centers = current_center.distance(neighbor_center)
            a = distance_between_centers - neighbor_radius
    
            # Determine required radius adjustment
            if perpendicular_distance > a:
                if current_radius < perpendicular_distance:
                    element["radius"] = perpendicular_distance
            else:
                if current_radius < a:
                    increase = (a - current_radius) / 2
                    element["radius"] += increase
                    circle_neighbor["radius"] += increase
    
        # Case 3: Two Neighboring Circles
        elif len(circle_neighbors) == 2:
            max_a = 0
            for neighbor in circle_neighbors:
                neighbor_center = Point(neighbor["center"])
                neighbor_radius = neighbor["radius"]
                distance_between_centers = current_center.distance(neighbor_center)
                a = distance_between_centers - neighbor_radius
                max_a = max(max_a, a)
    
            if current_radius < max_a:
                increase = (max_a - current_radius) / 2
                element["radius"] += increase
                for neighbor in circle_neighbors:
                    neighbor["radius"] += increase
    
        # Step 4: Calculate and store the new adapted midpoint
        midpoint_vector = np.array(element["midpoint"]) - np.array(element["center"])
        unit_vector = midpoint_vector / np.linalg.norm(midpoint_vector)
        new_adapted_midpoint = np.array(element["center"]) + unit_vector * element["radius"]
        
        # Store the adapted midpoint in the dictionary with wall_id as key
        adapted_midpoints[element["wall_id"]] = new_adapted_midpoint
    

    
    from shapely.geometry import Point, MultiPoint, LineString
    import numpy as np
    
    # Initialize the dictionary to store intersections grouped by pairs
    grouped_intersections = {}
    
    # Loop through each pair in sequential order based on modified_axes
    for i in range(len(modified_axes)):
        # Current and next elements for intersection pairing
        current_element = modified_axes[i]
        next_element = modified_axes[(i + 1) % len(modified_axes)]  # Wrap-around for last element
    
        # Define the group ID
        group_id = f"Group {i + 1}"
        grouped_intersections[group_id] = {
            "element_1": {"wall_id": current_element["wall_id"], "type": current_element["type"]},
            "element_2": {"wall_id": next_element["wall_id"], "type": next_element["type"]},
            "intersection_points": []
        }
    
        # Retrieve geometries based on element types
        current_geometry = current_element.get("axis", current_element.get("geometry"))
        next_geometry = next_element.get("axis", next_element.get("geometry"))
    
        # Ensure both geometries are valid
        if not current_geometry or not next_geometry:
            
            continue
    
        # Handle different combinations of types
        intersection_points = []
    
        # Circle-Line Intersection
        if current_element["type"] == "circle" and next_element["type"] == "line":
            circle_center = Point(current_element["center"])
            circle_radius = current_element["radius"]
            circle_boundary = circle_center.buffer(circle_radius).boundary
    
            # Compute intersection
            intersection = circle_boundary.intersection(next_geometry)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    intersection_points.append(intersection)
                elif isinstance(intersection, MultiPoint):
                    intersection_points.extend(intersection.geoms)
    
            # Handle tangency as a fallback
            distance_to_line = next_geometry.distance(circle_center)
            if np.isclose(distance_to_line, circle_radius, atol=1e-5):
                tangent_point = next_geometry.interpolate(next_geometry.project(circle_center))
                intersection_points.append(tangent_point)
                
    
        # Line-Circle Intersection
        elif current_element["type"] == "line" and next_element["type"] == "circle":
            circle_center = Point(next_element["center"])
            circle_radius = next_element["radius"]
            circle_boundary = circle_center.buffer(circle_radius).boundary
    
            # Compute intersection
            intersection = circle_boundary.intersection(current_geometry)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    intersection_points.append(intersection)
                elif isinstance(intersection, MultiPoint):
                    intersection_points.extend(intersection.geoms)
    
            # Handle tangency as a fallback
            distance_to_line = current_geometry.distance(circle_center)
            if np.isclose(distance_to_line, circle_radius, atol=1e-5):
                tangent_point = current_geometry.interpolate(current_geometry.project(circle_center))
                intersection_points.append(tangent_point)
                
    
        # Circle-Circle Intersection
        elif current_element["type"] == "circle" and next_element["type"] == "circle":
            current_center = Point(current_element["center"])
            current_radius = current_element["radius"]
            next_center = Point(next_element["center"])
            next_radius = next_element["radius"]
    
            # Compute intersection
            distance_between_centers = current_center.distance(next_center)
            if np.isclose(distance_between_centers, current_radius + next_radius, atol=1e-5) or \
               np.isclose(distance_between_centers, abs(current_radius - next_radius), atol=1e-5):
                # Tangent circles
                direction_vector = np.array([next_center.x - current_center.x, next_center.y - current_center.y])
                direction_unit = direction_vector / np.linalg.norm(direction_vector)
                tangent_point = Point(current_center.x + direction_unit[0] * current_radius,
                                      current_center.y + direction_unit[1] * current_radius)
                intersection_points.append(tangent_point)
                
            else:
                current_boundary = current_center.buffer(current_radius).boundary
                next_boundary = next_center.buffer(next_radius).boundary
                intersection = current_boundary.intersection(next_boundary)
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        intersection_points.append(intersection)
                    elif isinstance(intersection, MultiPoint):
                        intersection_points.extend(intersection.geoms)
    
        # Line-Line Intersection or other cases
        else:
            intersection = current_geometry.intersection(next_geometry)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    intersection_points.append(intersection)
                elif isinstance(intersection, MultiPoint):
                    intersection_points.extend(intersection.geoms)
    
        # Store the intersection points for the current group
        if intersection_points:
            grouped_intersections[group_id]["intersection_points"] = intersection_points
        
    


    import numpy as np
    from shapely.geometry import Point, LineString, MultiPoint
    from shapely.affinity import scale
    import matplotlib.pyplot as plt
    
    # Block(5): Modified to handle empty intersection points and print them
    
    import numpy as np
    from shapely.geometry import Point, LineString, MultiPoint
    from shapely.affinity import scale
    import matplotlib.pyplot as plt
    
    # Reinitialize the plot to add intersections and plot only the arcs containing adapted midpoints
    
    
    # Plot the existing adapted circles (dashed green) and straight lines (blue) for context
    for element in modified_axes:
        if element["type"] == "line":
            # Extend line substantially
            line_axis = element["axis"]
            extended_line = scale(line_axis, xfact=3, yfact=3, origin='center')
            
        elif element["type"] == "circle":
            # Plot adapted circles with dashed green line
            center = element["center"]
            radius = element["radius"]
            theta = np.linspace(0, 2 * np.pi, 100)
            adapted_circle_x = center[0] + radius * np.cos(theta)
            adapted_circle_y = center[1] + radius * np.sin(theta)
            
    
    # Initialize lists to store selected arcs and intersection points
    selected_arcs = []  # Store selected arcs with IDs for final plot
    group_intersection_points = {}  # Store intersection points for each group
    
    for element in modified_axes:
        if element["type"] != "circle":
            continue  # Only process circles
    
        # Retrieve properties for the current circle
        center = element["center"]
        radius = element["radius"]
        current_index = element["index"]
        circle_boundary = Point(center).buffer(radius).boundary  # Define circle boundary for intersection
    
        # Calculate the adapted midpoint by extending from the center to the original midpoint
        original_midpoint = np.array(element["midpoint"])
        midpoint_vector = original_midpoint - np.array(center)
        unit_vector = midpoint_vector / np.linalg.norm(midpoint_vector)
        adapted_midpoint = np.array(center) + unit_vector * radius
    
        # Calculate the angle for the adapted midpoint
        adapted_angle = np.arctan2(adapted_midpoint[1] - center[1], adapted_midpoint[0] - center[0])
    
        # Identify the two neighbors by neighboring indexes
        prev_index = (current_index - 1) if current_index > 1 else len(modified_axes)
        next_index = (current_index + 1) if current_index < len(modified_axes) else 1
    
        prev_neighbor = next((e for e in modified_axes if e["index"] == prev_index), None)
        next_neighbor = next((e for e in modified_axes if e["index"] == next_index), None)
    
        intersection_points = []
    
        # Find intersection with each neighbor
        for neighbor in [prev_neighbor, next_neighbor]:
            if neighbor is None:
                continue
            if neighbor["type"] == "line":
                # Extend neighbor line and find intersection with the circle
                line_axis = neighbor["axis"]
                extended_line = scale(line_axis, xfact=3, yfact=3, origin='center')  # Extend line further
                intersection = circle_boundary.intersection(extended_line)
    
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        intersection_points.append(intersection)
                    elif isinstance(intersection, MultiPoint):
                        intersection_points.extend(intersection.geoms)
                else:
                    # Handle case where line and circle do not intersect
                    # Find closest point on the line to the circle's center
                    closest_point = line_axis.interpolate(line_axis.project(Point(center)))
                    intersection_points.append(closest_point)
    
            elif neighbor["type"] == "circle":
                # Find intersection between two circles
                neighbor_center = Point(neighbor["center"])
                neighbor_radius = neighbor["radius"]
                neighbor_circle_boundary = neighbor_center.buffer(neighbor_radius).boundary
    
                intersection = circle_boundary.intersection(neighbor_circle_boundary)
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        intersection_points.append(intersection)
                    elif isinstance(intersection, MultiPoint):
                        intersection_points.extend(intersection.geoms)
                else:
                    # Handle case where circles do not intersect
                    # Find the closest point on the neighbor circle boundary to the current circle center
                    closest_point = neighbor_circle_boundary.interpolate(neighbor_circle_boundary.project(Point(center)))
                    intersection_points.append(closest_point)
    
        # Store intersection points for this group
        group_key = f"Circle {element['wall_id']} with Neighbors"
        group_intersection_points[group_key] = intersection_points
    

    
        # Proceed with processing the intersection points
        # Sort intersection points by angle around the circle center
        if len(intersection_points) >= 2:
            intersection_points = sorted(
                intersection_points,
                key=lambda pt: np.arctan2(pt.y - center[1], pt.x - center[0])
            )
    
            # Find the arc that contains the adapted midpoint or snap to the nearest valid arc
            arc_found = False
            best_arc_theta = None
            min_angle_diff = float('inf')  # Initialize minimum angle difference
    
            for i in range(len(intersection_points)):
                point1 = intersection_points[i]
                point2 = intersection_points[(i + 1) % len(intersection_points)]
    
                # Calculate angles for arc endpoints
                angle1 = np.arctan2(point1.y - center[1], point1.x - center[0])
                angle2 = np.arctan2(point2.y - center[1], point2.x - center[0])
    
                # Handle angle wrap-around
                if angle2 < angle1:
                    angle2 += 2 * np.pi
    
                # Check if adapted midpoint is within this arc
                if angle1 <= adapted_angle <= angle2:
                    arc_found = True
                    # Generate the arc
                    arc_theta = np.linspace(angle1, angle2, 50)
                    arc_x = center[0] + radius * np.cos(arc_theta)
                    arc_y = center[1] + radius * np.sin(arc_theta)
    
                    # Store the arc with wall ID for the final plot
                    selected_arcs.append((arc_x, arc_y, element["wall_id"]))
                    break  # Stop searching once a valid arc is found
    
                # Fallback: Calculate angular difference and keep track of the closest segment
                midpoint_angle_diff = min(abs(adapted_angle - angle1), abs(adapted_angle - angle2))
                if midpoint_angle_diff < min_angle_diff:
                    min_angle_diff = midpoint_angle_diff
                    best_arc_theta = np.linspace(angle1, angle2, 50)  # Save the closest arc
    
            # If no valid arc is found, use the closest arc as fallback
            if not arc_found and best_arc_theta is not None:
                
                arc_x = center[0] + radius * np.cos(best_arc_theta)
                arc_y = center[1] + radius * np.sin(best_arc_theta)
                selected_arcs.append((arc_x, arc_y, element["wall_id"]))
    
   
    from shapely.geometry import Point, LineString, MultiPoint
    import numpy as np
    
    # Initialize Phase 3 groups based on sequential pairing of modified axes
    phase3_groups = {}
    
    # Sort `modified_axes` by index to ensure sequential order
    sorted_modified_axes = sorted(modified_axes, key=lambda item: item["index"])
    n = len(sorted_modified_axes)
    
    # Step 1: Process intersections for each pair and assign them to `phase3_groups`
    for i in range(n):
        # Current and next elements for intersection pairing
        current_element = sorted_modified_axes[i]
        next_element = sorted_modified_axes[(i + 1) % n]
    
        # Define the group ID
        group_id = f"Group_{i + 1}"
        phase3_groups[group_id] = {
            "element_1": {
                "wall_id": current_element["wall_id"],
                "index": current_element["index"],
                "type": current_element["type"],
                "geometry": current_element
            },
            "element_2": {
                "wall_id": next_element["wall_id"],
                "index": next_element["index"],
                "type": next_element["type"],
                "geometry": next_element
            },
            "intersection": []  # Placeholder for intersection points
        }
    
        # Prepare geometries based on element types
        if current_element["type"] == "line":
            geom_1 = current_element["axis"]
        elif current_element["type"] == "circle":
            center_1 = Point(current_element["center"])
            radius_1 = current_element["radius"]
            geom_1 = center_1.buffer(radius_1).boundary
    
        if next_element["type"] == "line":
            geom_2 = next_element["axis"]
        elif next_element["type"] == "circle":
            center_2 = Point(next_element["center"])
            radius_2 = next_element["radius"]
            geom_2 = center_2.buffer(radius_2).boundary
    
        # Handle circle-to-circle tangency specifically
        if current_element["type"] == "circle" and next_element["type"] == "circle":
            distance_between_centers = center_1.distance(center_2)
            if np.isclose(distance_between_centers, radius_1 + radius_2, atol=1e-5):
                # Calculate tangent point for each circle
                direction_vector = np.array([center_2.x - center_1.x, center_2.y - center_1.y])
                direction_unit = direction_vector / np.linalg.norm(direction_vector)
                tangent_point_self = Point(center_1.x + direction_unit[0] * radius_1, center_1.y + direction_unit[1] * radius_1)
                tangent_point_neighbor = Point(center_2.x - direction_unit[0] * radius_2, center_2.y - direction_unit[1] * radius_2)
                
                # Add both tangent points as intersection points
                phase3_groups[group_id]["intersection"] = [tangent_point_self, tangent_point_neighbor]
            else:
                # Standard intersection if not tangent
                intersection = geom_1.intersection(geom_2)
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        phase3_groups[group_id]["intersection"] = [intersection]
                    elif isinstance(intersection, MultiPoint):
                        phase3_groups[group_id]["intersection"] = list(intersection.geoms)
        else:
            # Calculate intersections for other combinations
            intersection = geom_1.intersection(geom_2)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    phase3_groups[group_id]["intersection"] = [intersection]
                elif isinstance(intersection, MultiPoint):
                    phase3_groups[group_id]["intersection"] = list(intersection.geoms)
            else:
                # Robustness addition starts here
                # Handle circle-line cases where intersection is empty
                if (current_element["type"] == "circle" and next_element["type"] == "line"):
                    # current_element is circle, next_element is line
                    circle_center = Point(current_element["center"])
                    line = geom_2
    
                    # Find closest point on the line to the circle's center
                    closest_point = line.interpolate(line.project(circle_center))
                    phase3_groups[group_id]["intersection"] = [closest_point]
                    
    
                elif (current_element["type"] == "line" and next_element["type"] == "circle"):
                    # current_element is line, next_element is circle
                    circle_center = Point(next_element["center"])
                    line = geom_1
    
                    # Find closest point on the line to the circle's center
                    closest_point = line.interpolate(line.project(circle_center))
                    phase3_groups[group_id]["intersection"] = [closest_point]
                    
    
                
                    
    
    # Step 2: Index and filter intersections based on Phase 2 original points
    from shapely.geometry import Point
    import math
    import matplotlib.pyplot as plt
    
    # Dictionary `original_point_index` holds the correct index (1 or 2) of the original intersection point from Phase 2
    filtered_points_by_group = {}  # For storing selected points
    
    # Filter intersections for each group
    for group_id, group_data in phase3_groups.items():
        intersection_points = group_data["intersection"]
       
    
        if len(intersection_points) == 2:
            point_1, point_2 = intersection_points
    
            # Assign indexes based on Y, then X coordinates
            if point_1.y > point_2.y:
                point_1_index, point_2_index = 1, 2
            elif point_1.y < point_2.y:
                point_1_index, point_2_index = 2, 1
            else:
                point_1_index, point_2_index = (1, 2) if point_1.x > point_2.x else (2, 1)
    
            # Determine which point to keep based on `original_point_index`
            required_index = original_point_index.get(group_id, None)  # Avoid KeyError by returning None if not found
           
    
            # Select point according to the index in Phase 2
            if required_index == point_1_index:
                selected_point = point_1
            elif required_index == point_2_index:
                selected_point = point_2
            else:
               
                selected_point = None
    
            # Store the selected point in the dictionary
            if selected_point:
                filtered_points_by_group[group_id] = selected_point
                
    
        elif len(intersection_points) == 1:
            single_point = intersection_points[0]
            filtered_points_by_group[group_id] = single_point
           

    
    # Plot the adapted circles (dashed green) and modified lines (blue)
    for element in modified_axes:
        if element["type"] == "circle":
            # Plot the adapted circle
            center = element["center"]
            radius = element["radius"]
            theta = np.linspace(0, 2 * np.pi, 200)
            adapted_circle_x = center[0] + radius * np.cos(theta)
            adapted_circle_y = center[1] + radius * np.sin(theta)
            
        elif element["type"] == "line":
            # Extend and plot the modified line
            line_axis = element["axis"]
            extended_line = scale(line_axis, xfact=3, yfact=3, origin='center')
            
    
    # Consolidate legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    
  
    import numpy as np
    from shapely.geometry import Point, LineString
    from shapely.affinity import scale
    import matplotlib.pyplot as plt
    
    # Helper function to normalize an angle to the range [0, 2*pi)
    def normalize_angle(angle):
        return angle % (2 * np.pi)
    
    # Helper function to compute the angular difference and generate arc theta values
    def generate_arc_angles(angle_start, angle_end, adapted_angle):
        angle_start = normalize_angle(angle_start)
        angle_end = normalize_angle(angle_end)
        adapted_angle = normalize_angle(adapted_angle)
        
        # Determine if the angle range crosses the 2*pi boundary
        if angle_end < angle_start:
            angle_end += 2 * np.pi  # Adjust angle_end to ensure it is greater than angle_start
        
        # Adjust adapted_angle if necessary
        if adapted_angle < angle_start:
            adapted_angle += 2 * np.pi
        elif adapted_angle > angle_end:
            adapted_angle -= 2 * np.pi
        
        # Check if adapted_angle is within the range
        if angle_start <= adapted_angle <= angle_end:
            # Generate theta values for the arc
            arc_theta = np.linspace(angle_start, angle_end, 100)
            return arc_theta
        else:
            # The adapted midpoint is not between the two intersection points
            # Try the other direction by adding 2*pi to angle_end
            angle_end += 2 * np.pi
            if angle_start <= adapted_angle <= angle_end:
                arc_theta = np.linspace(angle_start, angle_end, 100)
                return arc_theta
            else:
                # The adapted midpoint is still not between the two intersection points
                # Return None to indicate that the arc cannot be generated
                return None
    
    # Create a dictionary to map wall_id to elements in modified_axes
    modified_axes_dict = {element["wall_id"]: element for element in modified_axes}
    
    # Initialize the plot to visualize intersections and final selected segments
    
    selected_segments = []  # Store selected line segments for final canvas
    selected_arcs = []  # Store selected arc segments for final canvas
    
    # Plot the existing adapted circles (dashed green) and straight lines (blue) for context
    for element in modified_axes:
        if element["type"] == "line":
            # Extend and plot the line for visualization
            line_axis = element["axis"]
            extended_line = scale(line_axis, xfact=3, yfact=3, origin='center')
            
    
            # Plot the modified midpoint
            modified_midpoint = element["midpoint"]
            
    
        elif element["type"] == "circle":
            # Plot the adapted circle in dashed green
            center = element["center"]
            radius = element["radius"]
            theta = np.linspace(0, 2 * np.pi, 200)
            adapted_circle_x = center[0] + radius * np.cos(theta)
            adapted_circle_y = center[1] + radius * np.sin(theta)
            
    
    from shapely.geometry import LineString, Point
    from shapely.affinity import scale
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    import numpy as np
    from shapely.geometry import Point, LineString
    import matplotlib.pyplot as plt
    
    # Initialize the plot to visualize selected line segments
    
    selected_line_segments = []  # Store selected line segments for final visualization
    
    # Process each modified line
    for element in modified_axes:
        if element["type"] != "line":
            continue  # Only process lines
    
        wall_id = element["wall_id"]
        modified_line = element["axis"]
    
        
    
        # Retrieve intersection points for the current line
        intersection_points = [
            filtered_points_by_group.get(group_id)
            for group_id, group_data in phase3_groups.items()
            if group_data["element_1"]["wall_id"] == wall_id or group_data["element_2"]["wall_id"] == wall_id
        ]
        intersection_points = [pt for pt in intersection_points if pt is not None]
    
        if len(intersection_points) < 2:
            
            continue
    
        
    
        # Sort intersection points along the line
        intersection_points = sorted(intersection_points, key=lambda pt: modified_line.project(pt))
    
        # Select the segment directly between the first two intersection points
        segment_start = intersection_points[0]
        segment_end = intersection_points[1]
        selected_segment = LineString([segment_start, segment_end])
    
        # Add to the list of selected line segments
        selected_line_segments.append((wall_id, selected_segment))
    
        # Plot the selected segment
        
        midpoint = selected_segment.interpolate(0.5, normalized=True)
        
    
    
    # Consolidate the legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels and label != '':
            unique_labels[label] = handle
    

    
    # Part 1: Highlighting Representative Halves and Storing Selected Halves with S1, S2, and Half Description
    
    
    selected_halves = {}  # Dictionary to store selected halves for each element
    
    # Process each element
    for element in modified_axes:
        element_id = element["wall_id"]
        element_type = element["type"]
    
        if element_type != "circle":
            continue
    
        center = element["center"]
        radius = element["radius"]
        arc_midpoint = adapted_midpoints[element_id]
    
        # Retrieve intersection points
        intersection_points = [
            filtered_points_by_group.get(group_id)
            for group_id in phase3_groups
            if group_id in filtered_points_by_group and
               (phase3_groups[group_id]["element_1"]["wall_id"] == element_id or
                phase3_groups[group_id]["element_2"]["wall_id"] == element_id)
        ]
        intersection_points = [pt for pt in intersection_points if pt is not None]
    
        # If no intersection points, define default angles
        if len(intersection_points) == 0:
            angle_S1 = 0
            angle_S2 = 2 * np.pi
        elif len(intersection_points) == 1:
            angle_S1 = np.arctan2(intersection_points[0].y - center[1], intersection_points[0].x - center[0]) % (2 * np.pi)
            angle_S2 = (angle_S1 + np.pi) % (2 * np.pi)
        elif len(intersection_points) >= 2:
            angle_S1 = np.arctan2(intersection_points[0].y - center[1], intersection_points[0].x - center[0]) % (2 * np.pi)
            angle_S2 = np.arctan2(intersection_points[1].y - center[1], intersection_points[1].x - center[0]) % (2 * np.pi)
            if (angle_S2 - angle_S1) % (2 * np.pi) < (angle_S1 - angle_S2) % (2 * np.pi):
                angle_S1, angle_S2 = angle_S2, angle_S1
        else:
            continue
    
        # Function to compute midpoint between two angles considering wrapping
        def angle_midpoint(angle1, angle2):
            diff = (angle2 - angle1) % (2 * np.pi)
            return (angle1 + diff / 2) % (2 * np.pi)
    
        # Compute midpoints of segments
        midpoint_sm1_angle = angle_midpoint(angle_S1, angle_S2)
        midpoint_sm2_angle = angle_midpoint(angle_S2, angle_S1 + 2 * np.pi)
    
        midpoint_sm1 = (
            center[0] + radius * np.cos(midpoint_sm1_angle),
            center[1] + radius * np.sin(midpoint_sm1_angle)
        )
        midpoint_sm2 = (
            center[0] + radius * np.cos(midpoint_sm2_angle),
            center[1] + radius * np.sin(midpoint_sm2_angle)
        )
    
        # Compute perpendicular angles for S1 and S2
        perpendicular_angle_1 = (midpoint_sm1_angle + np.pi / 2) % (2 * np.pi)
        perpendicular_angle_2 = (midpoint_sm1_angle - np.pi / 2) % (2 * np.pi)
    
        s1 = (
            center[0] + radius * np.cos(perpendicular_angle_1),
            center[1] + radius * np.sin(perpendicular_angle_1)
        )
        s2 = (
            center[0] + radius * np.cos(perpendicular_angle_2),
            center[1] + radius * np.sin(perpendicular_angle_2)
        )
    
 
    
        # Calculate the equator (perpendicular bisector)
        dx = midpoint_sm2[0] - midpoint_sm1[0]
        dy = midpoint_sm2[1] - midpoint_sm1[1]
        if dy != 0:
            equator_slope = -dx / dy
        else:
            equator_slope = np.inf
    
        equator_length = radius
    
        line_midpoints_center = (
            (midpoint_sm1[0] + midpoint_sm2[0]) / 2,
            (midpoint_sm1[1] + midpoint_sm2[1]) / 2
        )
    
        # Define the equator line
        if equator_slope == np.inf:
            x_eq = line_midpoints_center[0]
            y_eq = np.linspace(line_midpoints_center[1] - equator_length / 2, line_midpoints_center[1] + equator_length / 2, 2)
            x_eq = [x_eq, x_eq]
        else:
            x_eq = np.linspace(line_midpoints_center[0] - equator_length / 2, line_midpoints_center[0] + equator_length / 2, 2)
            y_eq = equator_slope * (x_eq - line_midpoints_center[0]) + line_midpoints_center[1]
    
        
        equator_line = lambda x, y: (y - line_midpoints_center[1]) - equator_slope * (x - line_midpoints_center[0]) if equator_slope != np.inf else x - line_midpoints_center[0]
    
        theta_full = np.linspace(0, 2 * np.pi, 360)
        half1_theta = []
        half2_theta = []
    
        for theta in theta_full:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            if equator_line(x, y) > 0:
                half1_theta.append(theta)
            else:
                half2_theta.append(theta)
    
        half1_x = center[0] + radius * np.cos(half1_theta)
        half1_y = center[1] + radius * np.sin(half1_theta)
        half2_x = center[0] + radius * np.cos(half2_theta)
        half2_y = center[1] + radius * np.sin(half2_theta)
    
        mid_sign = equator_line(arc_midpoint[0], arc_midpoint[1])
    
        if mid_sign > 0:
            selected_half_x = half1_x
            selected_half_y = half1_y
            selected_half_label = "Representative Half 1"
            selected_half_theta = half1_theta
            half_description = f"S1 < selected_half < S2"
        else:
            selected_half_x = half2_x
            selected_half_y = half2_y
            selected_half_label = "Representative Half 2"
            selected_half_theta = half2_theta
            half_description = f"S2 < selected_half < 360 and 0 < selected_half < S1"
    
       
        selected_halves[element_id] = {
            "start_angle": selected_half_theta[0],
            "end_angle": selected_half_theta[-1],
            "theta_list": selected_half_theta,
            "label": selected_half_label,
            "equator_line": equator_line,
            "s1": s1,
            "s2": s2,
            "half_description": half_description
        }
    

    # Part 4: Correctly Splitting Circle into Segments 1 and 2 with Robust Midpoints
    
    
    segment_midpoints = {}  # Dictionary to store midpoints and their angles for each element
    
    # Process each element
    for element in modified_axes:
        element_id = element["wall_id"]
        element_type = element["type"]
    
        if element_type != "circle":
            continue
    
        center = element["center"]
        radius = element["radius"]
    
        # Retrieve intersection points
        intersection_points = [
            filtered_points_by_group.get(group_id)
            for group_id in phase3_groups
            if group_id in filtered_points_by_group and
               (phase3_groups[group_id]["element_1"]["wall_id"] == element_id or
                phase3_groups[group_id]["element_2"]["wall_id"] == element_id)
        ]
        intersection_points = [pt for pt in intersection_points if pt is not None]
    
        if len(intersection_points) != 2:
            continue
    
        # Plot the adapted circle
        theta_full = np.linspace(0, 2 * np.pi, 200)
        adapted_circle_x = center[0] + radius * np.cos(theta_full)
        adapted_circle_y = center[1] + radius * np.sin(theta_full)
        
    
        # Calculate angles for the intersection points
        angle_P1 = np.arctan2(intersection_points[0].y - center[1], intersection_points[0].x - center[0]) % (2 * np.pi)
        angle_P2 = np.arctan2(intersection_points[1].y - center[1], intersection_points[1].x - center[0]) % (2 * np.pi)
    
        # Ensure angle_P2 > angle_P1 for consistent ordering
        if (angle_P2 - angle_P1) % (2 * np.pi) < (angle_P1 - angle_P2) % (2 * np.pi):
            angle_P1, angle_P2 = angle_P2, angle_P1
    
        # Function to generate angles between two angles considering wrapping
        def generate_theta(start_angle, end_angle, num_points=100):
            if end_angle >= start_angle:
                return np.linspace(start_angle, end_angle, num_points)
            else:
                # Wrap around 2π
                return np.linspace(start_angle, end_angle + 2 * np.pi, num_points) % (2 * np.pi)
    
        # Generate theta values for Segment 1 and Segment 2
        segment1_theta = generate_theta(angle_P1, angle_P2)
        segment2_theta = generate_theta(angle_P2, angle_P1)
    
        # Compute coordinates for Segment 1 and Segment 2
        segment1_x = center[0] + radius * np.cos(segment1_theta)
        segment1_y = center[1] + radius * np.sin(segment1_theta)
    
        segment2_x = center[0] + radius * np.cos(segment2_theta)
        segment2_y = center[1] + radius * np.sin(segment2_theta)
    
       
    
        # Function to compute midpoint between two angles considering wrapping
        def angle_midpoint(angle1, angle2):
            diff = (angle2 - angle1) % (2 * np.pi)
            return (angle1 + diff / 2) % (2 * np.pi)
    
        # Robust midpoint calculation using segment angles
        # Midpoint for Segment 1
        midpoint_segment1_angle = angle_midpoint(angle_P1, angle_P2)
        midpoint_segment1 = (
            center[0] + radius * np.cos(midpoint_segment1_angle),
            center[1] + radius * np.sin(midpoint_segment1_angle)
        )
    
        # Midpoint for Segment 2
        midpoint_segment2_angle = angle_midpoint(angle_P2, angle_P1 + 2 * np.pi)
        midpoint_segment2 = (
            center[0] + radius * np.cos(midpoint_segment2_angle),
            center[1] + radius * np.sin(midpoint_segment2_angle)
        )
    
        
    
        # Store midpoints and angles for use in Part 3
        segment_midpoints[element_id] = {
            'midpoint_segment1': midpoint_segment1,
            'midpoint_segment2': midpoint_segment2,
            'midpoint_segment1_angle': midpoint_segment1_angle,
            'midpoint_segment2_angle': midpoint_segment2_angle,
            'angle_P1': angle_P1,
            'angle_P2': angle_P2,
            'segment1_theta': segment1_theta,
            'segment2_theta': segment2_theta,
        }
    
        # Avoid duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
    
    
    # Part 3(2): Highlighting the Single Midpoint That Belongs to the Selected Half
    
    
    selected_midpoints = {}  # Dictionary to store selected midpoints for each element
    
    # Function to check if an angle is between two angles considering wrapping around 2π
    def is_angle_between(angle, start_angle, end_angle):
        angle = angle % (2 * np.pi)
        start_angle = start_angle % (2 * np.pi)
        end_angle = end_angle % (2 * np.pi)
        if start_angle <= end_angle:
            return start_angle <= angle <= end_angle
        else:
            # The range crosses 0 angle
            return angle >= start_angle or angle <= end_angle
    
    # Process each element
    for element in modified_axes:
        element_id = element["wall_id"]
        element_type = element["type"]
    
        if element_type != "circle":
            continue
    
        center = element["center"]
        radius = element["radius"]
        arc_midpoint = adapted_midpoints[element_id]
    
        # Retrieve the selected half data from Part 1
        selected_half_data = selected_halves.get(element_id)
        segment_data = segment_midpoints.get(element_id)  # Inheriting midpoints from Part 4
        if not selected_half_data:
           
            continue
    
        # Retrieve necessary data from selected_half_data
        selected_half_theta = selected_half_data["theta_list"]
        s1 = selected_half_data["s1"]  # S1 from Part 1
        s2 = selected_half_data["s2"]  # S2 from Part 1
    
        # Recalculate S1 and S2 angles
        angle_S1 = np.arctan2(s1[1] - center[1], s1[0] - center[0]) % (2 * np.pi)
        angle_S2 = np.arctan2(s2[1] - center[1], s2[0] - center[0]) % (2 * np.pi)
    
        # Retrieve the midpoints from Part 4
        if segment_data:
            midpoint_sm1 = segment_data["midpoint_segment1"]
            midpoint_sm2 = segment_data["midpoint_segment2"]
            angle_sm1 = np.arctan2(midpoint_sm1[1] - center[1], midpoint_sm1[0] - center[0]) % (2 * np.pi)
            angle_sm2 = np.arctan2(midpoint_sm2[1] - center[1], midpoint_sm2[0] - center[0]) % (2 * np.pi)
        else:
            
            continue
    
        # Check which midpoint lies within the selected half
        if is_angle_between(angle_sm1, angle_S1, angle_S2):
            selected_midpoint = midpoint_sm1
            selected_midpoint_label = "Midpoint Segment 1"
            selected_midpoints[element_id] = {"midpoint": selected_midpoint, "segment_label": "Segment 1"}
            selected_color = 'orange'
        elif is_angle_between(angle_sm2, angle_S1, angle_S2):
            selected_midpoint = midpoint_sm2
            selected_midpoint_label = "Midpoint Segment 2"
            selected_midpoints[element_id] = {"midpoint": selected_midpoint, "segment_label": "Segment 2"}
            selected_color = 'purple'
        else:
            # In case neither midpoint falls within the selected half
            
            continue
    
        # Plot the adapted circle
        theta_full = np.linspace(0, 2 * np.pi, 200)
        adapted_circle_x = center[0] + radius * np.cos(theta_full)
        adapted_circle_y = center[1] + radius * np.sin(theta_full)
        
     
    
        # Plot the selected half for visualization
        selected_half_x = center[0] + radius * np.cos(selected_half_theta)
        selected_half_y = center[1] + radius * np.sin(selected_half_theta)
        
    
    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    
    # Block(2.2): Highlighting Selected Segments and Storing Selected Arc Segments

    
    # Initialize the list to store selected arc segments
    selected_arc_segments = []
    
    # Process each element
    for element in modified_axes:
        element_id = element["wall_id"]
        element_type = element["type"]
    
        if element_type != "circle":
            continue
    
        center = element["center"]
        radius = element["radius"]
    
        # Retrieve the selected midpoint from Part 3
        selected_midpoint_data = selected_midpoints.get(element_id)
        if not selected_midpoint_data:
            
            continue
    
        # Get the label for the selected segment
        selected_segment_label = selected_midpoint_data["segment_label"]
    
        # Retrieve midpoints and intersection angles
        segment_data = segment_midpoints.get(element_id)
        if not segment_data:
            
            continue
    
        midpoint_sm1 = segment_data["midpoint_segment1"]
        midpoint_sm2 = segment_data["midpoint_segment2"]
    
        # Retrieve the angles for Segment 1 and Segment 2
        segment1_theta = segment_data["segment1_theta"]
        segment2_theta = segment_data["segment2_theta"]
    
        # Compute coordinates for Segment 1 and Segment 2
        segment1_x = center[0] + radius * np.cos(segment1_theta)
        segment1_y = center[1] + radius * np.sin(segment1_theta)
    
        segment2_x = center[0] + radius * np.cos(segment2_theta)
        segment2_y = center[1] + radius * np.sin(segment2_theta)
    
        # Store and plot the segment corresponding to the selected midpoint
        if selected_segment_label == "Segment 1":
            # Store the selected arc segment
            points = list(zip(segment1_x, segment1_y))
            arc_segment = LineString(points)
            selected_arc_segments.append((element_id, arc_segment))
    
            
        elif selected_segment_label == "Segment 2":
            # Store the selected arc segment
            points = list(zip(segment2_x, segment2_y))
            arc_segment = LineString(points)
            selected_arc_segments.append((element_id, arc_segment))
    
            
    
        # Optionally, plot the adapted circle for reference
        theta_full = np.linspace(0, 2 * np.pi, 200)
        adapted_circle_x = center[0] + radius * np.cos(theta_full)
        adapted_circle_y = center[1] + radius * np.sin(theta_full)
        
    
    # Consolidate the legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    
    import matplotlib.pyplot as plt
    


# Loop through the stored selected line segments
    for wall_id, segment in selected_line_segments:
        # Check if segment is valid
        if segment is not None and hasattr(segment, 'xy'):
            # Plot each segment
                
                
            # Optionally, add a label or midpoint marker
             midpoint = segment.interpolate(0.5, normalized=True)  # Calculate midpoint for labeling
      
                
    

    
        # Consolidate the legend to avoid duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
       
    
    
    # Block(d): Plotting Selected Line and Arc Segments with Original Sequential Indexes
    
    import matplotlib.pyplot as plt
    
    # Create a dictionary mapping indexes to selected segments (lines and arcs)
    selected_segments_dict = {}
    
    # Populate the dictionary for line segments
    for wall_id, segment in selected_line_segments:
        # Safely retrieve the index from the original axes mapping
        segment_data = original_axes_before_extension.get(wall_id)
        if segment_data:
            segment_index = segment_data["index"]
            selected_segments_dict[segment_index] = {
                "type": "line",
                "geometry": segment,
                "wall_id": wall_id
            }
    
    # Populate the dictionary for arc segments
    for element_id, arc_segment in selected_arc_segments:
        # Safely retrieve the index from the original axes mapping
        arc_data = original_axes_before_extension.get(element_id)
        if arc_data:
            arc_index = arc_data["index"]
            selected_segments_dict[arc_index] = {
                "type": "arc",
                "geometry": arc_segment,
                "wall_id": element_id
            }
    
    # Sort the dictionary by sequential index for clarity
    selected_segments_dict = dict(sorted(selected_segments_dict.items()))
    
 
    # Plot line segments
    for idx, segment_info in selected_segments_dict.items():
        if segment_info["type"] == "line":
            segment = segment_info["geometry"]
            wall_id = segment_info["wall_id"]
            # Plot the line segment
            
            # Add index near the midpoint of the segment
            midpoint = segment.interpolate(0.5, normalized=True)
            
    
    # Plot arc segments
    for idx, segment_info in selected_segments_dict.items():
        if segment_info["type"] == "arc":
            arc_segment = segment_info["geometry"]
            wall_id = segment_info["wall_id"]
            # Plot the arc segment
            
            # Add index near the midpoint of the arc
            midpoint = arc_segment.interpolate(0.5, normalized=True)
            
    

    
    # Consolidate the legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels and label != '':
            unique_labels[label] = handle
    




    import numpy as np
    from shapely.geometry import Point, LineString
    import matplotlib.pyplot as plt
    

    
    # Ensure selected_segments_dict exists and is sorted
    sorted_indices = sorted(selected_segments_dict.keys())
    
    
    # Step 1: Build a mapping of each group to its corresponding index in `selected_segments_dict`
    group_to_index = {f"Group_{i + 1}": idx for i, idx in enumerate(sorted_indices)}
    

    # Step 2: Build the intersections dictionary using `filtered_points_by_group`
    intersections = {}
    
    for group_id, idx in group_to_index.items():
        if group_id in filtered_points_by_group:
            intersection_point = filtered_points_by_group[group_id]
            intersections.setdefault(idx, {}).setdefault('end_point', intersection_point)
            
            # Identify the next index in the sequence (wrap-around for circular connectivity)
            next_idx = sorted_indices[(sorted_indices.index(idx) + 1) % len(sorted_indices)]
            intersections.setdefault(next_idx, {}).setdefault('start_point', intersection_point)
       
            
    
    # Step 3: Interpolate elements sequentially
    interpolation_points = []
    global_interpolation_index = 0  # Global index for interpolation points
    
    
    
    for idx in sorted_indices:
        current_element = selected_segments_dict[idx]
        element_type = current_element["type"]
        start_point = intersections.get(idx, {}).get('start_point')
        end_point = intersections.get(idx, {}).get('end_point')
    
        # Skip elements with missing start or end points
        if not start_point or not end_point:
            
            continue
    
        # Handle line elements
        if element_type == "line":
            line_segment = current_element["geometry"]
            start_distance = line_segment.project(start_point)
            end_distance = line_segment.project(end_point)
    
            # Generate interpolation points along the line
            num_points = max(int(np.ceil(abs(end_distance - start_distance) / line_segment.length * 100)), 2)
            fractions = np.linspace(start_distance, end_distance, num_points) / line_segment.length
    
            if start_distance > end_distance:  # Reverse if needed
                fractions = fractions[::-1]
    
            interpolated_points = [line_segment.interpolate(frac, normalized=True) for frac in fractions]
    
            for pt in interpolated_points:
                interpolation_points.append((pt, global_interpolation_index))
                global_interpolation_index += 1
    
        # Handle arc elements
        elif element_type == "arc":
            arc_segment = current_element["geometry"]  # Geometry is expected to be a LineString for arcs
    
            # Generate points along the arc using the start and end intersection points
            arc_points = [Point(pt) for pt in arc_segment.coords]
    
            # Find points closest to start and end intersections
            start_arc_point = min(arc_points, key=lambda pt: pt.distance(start_point))
            end_arc_point = min(arc_points, key=lambda pt: pt.distance(end_point))
    
            # Identify indices and sequence points
            start_idx = arc_points.index(start_arc_point)
            end_idx = arc_points.index(end_arc_point)
    
            if start_idx <= end_idx:
                interpolated_points = arc_points[start_idx:end_idx + 1]
            else:
                interpolated_points = arc_points[end_idx:start_idx + 1][::-1]
    
            for pt in interpolated_points:
                interpolation_points.append((pt, global_interpolation_index))
                global_interpolation_index += 1
    
    # Step 4: Ensure closed shape by appending the first point to the end if necessary
    if interpolation_points and not interpolation_points[0][0].equals(interpolation_points[-1][0]):
        interpolation_points.append((interpolation_points[0][0], global_interpolation_index))
        global_interpolation_index += 1
    
    # Step 5: Plot the interpolation points
    
    x_coords = [pt.x for pt, idx in interpolation_points]
    y_coords = [pt.y for pt, idx in interpolation_points]
    
    

    from shapely.geometry import LinearRing, Polygon, MultiPolygon
    from shapely.validation import make_valid
    import matplotlib.pyplot as plt
    
    # Step 1: Extract coordinates for the polygon from the interpolated points
    polygon_coords = [(pt.x, pt.y) for pt, idx in interpolation_points]
    
    # Ensure the shape is closed by adding the first point at the end if necessary
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])
    
    # Step 2: Create a LinearRing and then a Polygon
    try:
        linear_ring = LinearRing(polygon_coords)
        solid_shape = Polygon(linear_ring)
    except ValueError as e:
       
        solid_shape = None
    
    # Step 3: Validate the polygon and attempt to fix it if invalid
    if solid_shape and not solid_shape.is_valid:
        
        solid_shape = make_valid(solid_shape)
    
    # Step 4: Handle the MultiPolygon case
    polygons = []
    if isinstance(solid_shape, MultiPolygon):
        polygons = list(solid_shape.geoms)  # Extract individual polygons
    elif isinstance(solid_shape, Polygon):
        polygons = [solid_shape]  # Single polygon case
    
    

    
    # Plotting
    for poly in polygons:
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.fill(x, y, color='lightblue', alpha=0.3, label=f"Solid Shape (Seed {random_seed})")
            ax.plot(x, y, 'k-', linewidth=2, alpha=0.4)

    # Setting labels and title only if ax was created inside the function
    if ax is not None and ax.get_title() == '':
        ax.set_title("Overlapping Solid Shapes")
        ax.set_xlabel("X-axis (meters)")
        ax.set_ylabel("Y-axis (meters)")
        ax.grid(True)
        ax.axis('equal')

    # Optionally return plot data
    if return_plot_data:
        total_area = sum(poly.area for poly in polygons if isinstance(poly, Polygon))
        return polygons, total_area














#----------------------------------








#Multiple iterations





import matplotlib.pyplot as plt

random_seeds = [123, 456, 789, 101113, 131215, 335, 452, 56735, 73452345]  # List of random seeds

# Create a shared figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Loop through each seed and execute the function
for seed in random_seeds:
    print(f"Running with random_seed={seed}")
    # Call the function and pass the shared axis
    generate_solid_shape(random_seed=seed, ax=ax)

# Finalize the combined plot
ax.set_title("Overlapping Solid Shapes")
ax.set_xlabel("X-axis (meters)")
ax.set_ylabel("Y-axis (meters)")
ax.grid(True)
ax.axis('equal')

# Consolidate legend to avoid duplicates
handles, labels = ax.get_legend_handles_labels()
unique_labels = {}
for handle, label in zip(handles, labels):
    if label not in unique_labels and label != '':
        unique_labels[label] = handle
ax.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')





# Show the combined plot
plt.tight_layout()

# Save the plot as a JPEG file
plt.savefig("optimized_shape2.jpeg", format='jpeg', dpi=300)  # Save with high resolution
plt.show()
































