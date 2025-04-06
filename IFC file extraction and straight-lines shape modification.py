# IFC file s an input -> extracting geomtry from it -> indexing the walls 




import ifcopenshell
import ifcopenshell.util.placement
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint, MultiLineString, Point
from shapely.ops import unary_union, linemerge

# Load the IFC file
ifc_file = ifcopenshell.open('C:/Users/MBodrov/training_set_cnn/bim/bim4.ifc')

# Extract wall elements
walls = ifc_file.by_type('IfcWall') + ifc_file.by_type('IfcWallStandardCase')

# Function to create a transformation matrix for translation and rotation
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

# Function to apply a transformation matrix to a point
def apply_transformation(point, transformation_matrix):
    x, y = point
    point_vector = np.array([x, y, 1])  # Homogeneous coordinates
    transformed_point = transformation_matrix @ point_vector
    return transformed_point[0], transformed_point[1]

# Function to recursively accumulate transformations from IfcLocalPlacement
def get_cumulative_transformation(placement):
    cumulative_transformation = np.identity(3)
    while placement and placement.is_a("IfcLocalPlacement"):
        translation = [0, 0]
        rotation_angle = 0

        relative_placement = placement.RelativePlacement
        if relative_placement:
            if relative_placement.Location:
                location = relative_placement.Location
                translation[0] = location.Coordinates[0]
                translation[1] = location.Coordinates[1]

            if hasattr(relative_placement, 'RefDirection') and relative_placement.RefDirection:
                ref_direction = relative_placement.RefDirection
                if ref_direction.is_a("IfcDirection"):
                    ref_x, ref_y = ref_direction.DirectionRatios[:2]
                    rotation_angle = np.arctan2(ref_y, ref_x)

        local_transformation = create_transformation_matrix(translation, rotation_angle)
        cumulative_transformation = local_transformation @ cumulative_transformation
        placement = getattr(placement, 'PlacementRelTo', None)
    return cumulative_transformation

# Function to extract global geometry of a wall
def get_wall_global_geometry(wall):
    wall_points = []
    # Get cumulative transformation
    transformation = get_cumulative_transformation(wall.ObjectPlacement)

    # Extract wall's geometric representation
    if wall.Representation:
        for rep in wall.Representation.Representations:
            if rep.is_a("IfcShapeRepresentation"):
                for item in rep.Items:
                    if item.is_a("IfcExtrudedAreaSolid"):
                        profile = item.SweptArea
                        solid_transformation = get_cumulative_transformation(item.Position)
                        total_transformation = transformation @ solid_transformation

                        # Extract points from the profile
                        if hasattr(profile, 'OuterCurve') and profile.OuterCurve.is_a("IfcPolyline"):
                            points = profile.OuterCurve.Points
                            for p in points:
                                if hasattr(p, 'Coordinates'):
                                    x, y = p.Coordinates[0], p.Coordinates[1]
                                    transformed_x, transformed_y = apply_transformation((x, y), total_transformation)
                                    wall_points.append((transformed_x, transformed_y))
    return wall_points

# Extract global geometries of all walls
wall_geometries = {}
for wall in walls:
    wall_points = get_wall_global_geometry(wall)
    if wall_points:
        wall_geometries[wall.GlobalId] = wall_points

# Function to get the length of the wall
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

# Visualization of the original wall profiles
plt.figure(figsize=(10, 10))

# Plot original wall profiles
plt.plot([], [], 'g--', linewidth=1, label='Wall Profiles (green dashed)')  # Add label for wall profiles only once
for wall_id, wall_points in wall_geometries.items():
    if len(wall_points) > 1:
        x_coords = [point[0] for point in wall_points]
        y_coords = [point[1] for point in wall_points]

        # Close the loop if necessary
        if (x_coords[0], y_coords[0]) != (x_coords[-1], y_coords[-1]):
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])

        plt.plot(x_coords, y_coords, 'g--', linewidth=1)

# Plot shortened wall axes
axes_plotted = False  # Flag to track if the label has already been added
shortened_axes = []  # List to store all shortened axes
for wall in walls:
    if wall.GlobalId in wall_geometries:
        wall_points = wall_geometries[wall.GlobalId]
        if len(wall_points) > 1:
            # Create a polygon from wall points to represent the wall profile
            wall_polygon = Polygon(wall_points)

            # Calculate the midpoint between opposite wall edges to determine the centerline direction
            point_a = wall_points[0]
            point_b = wall_points[len(wall_points) // 2]
            midpoint_start = ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)

            # Calculate a second midpoint further along the wall to create a direction vector
            point_c = wall_points[1]
            point_d = wall_points[(len(wall_points) // 2) + 1]
            midpoint_end = ((point_c[0] + point_d[0]) / 2, (point_c[1] + point_d[1]) / 2)

            # Calculate direction vector for axis extension
            direction_vector = (midpoint_end[0] - midpoint_start[0], midpoint_end[1] - midpoint_start[1])
            direction_vector_length = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
            normalized_direction_vector = (direction_vector[0] / direction_vector_length, direction_vector[1] / direction_vector_length)

            # Get the length of the wall from IFC data
            wall_length = get_wall_length(wall)
            if wall_length is None:
                wall_length = 10  # Default length if not found

            # Extend the axis based on the length of the wall
            extended_start = (
                midpoint_start[0] - (wall_length / 2) * normalized_direction_vector[0],
                midpoint_start[1] - (wall_length / 2) * normalized_direction_vector[1]
            )
            extended_end = (
                midpoint_end[0] + (wall_length / 2) * normalized_direction_vector[0],
                midpoint_end[1] + (wall_length / 2) * normalized_direction_vector[1]
            )
            extended_axis = LineString([extended_start, extended_end])

            # Find the intersection points between the extended axis and the wall polygon
            intersection_points = extended_axis.intersection(wall_polygon)

            # Shorten the axis to only the part between intersections with end faces
            if isinstance(intersection_points, MultiPoint) and len(intersection_points) >= 2:
                # Sort intersection points to find the two furthest apart (representing the entire axis within the wall)
                points = sorted(intersection_points, key=lambda p: (p.x, p.y))
                shortened_axis = LineString([points[0], points[-1]])
                shortened_axes.append(shortened_axis)
                if not axes_plotted:
                    plt.plot(*shortened_axis.xy, 'b-', linewidth=2, label='Wall Axes (blue solid)')
                    axes_plotted = True
                else:
                    plt.plot(*shortened_axis.xy, 'b-', linewidth=2)
            elif isinstance(intersection_points, LineString):
                shortened_axes.append(intersection_points)
                if not axes_plotted:
                    plt.plot(*intersection_points.xy, 'b-', linewidth=2, label='Wall Axes (blue solid)')
                    axes_plotted = True
                else:
                    plt.plot(*intersection_points.xy, 'b-', linewidth=2)
            else:
                print(f"No valid intersection found for wall axis: {wall.GlobalId}")

plt.title("Original Wall Profiles and Wall Axes (Shortened to End Faces)")
plt.xlabel("X-axis (meters)")
plt.ylabel("Y-axis (meters)")
plt.grid(True)
plt.axis('equal')
plt.legend(loc='upper right')
plt.show()

print("Visualization complete.")

# Extract connectivity information from the IFC file
wall_connectivity = {}
for rel_connects in ifc_file.by_type("IfcRelConnectsElements"):
    if rel_connects.is_a("IfcRelConnectsElements"):
        relating_wall = rel_connects.RelatingElement
        related_wall = rel_connects.RelatedElement
        if relating_wall.is_a("IfcWall") and related_wall.is_a("IfcWall"):
            if relating_wall.GlobalId not in wall_connectivity:
                wall_connectivity[relating_wall.GlobalId] = []
            if related_wall.GlobalId not in wall_connectivity:
                wall_connectivity[related_wall.GlobalId] = []

            wall_connectivity[relating_wall.GlobalId].append(related_wall.GlobalId)
            wall_connectivity[related_wall.GlobalId].append(relating_wall.GlobalId)

# Sequential indexing of wall axes using connectivity information
# Start with a wall and index each wall connected to it
ordered_walls = []
visited_walls = set()

# Pick an arbitrary starting wall
starting_wall = next(iter(wall_connectivity))
current_wall = starting_wall
while current_wall:
    ordered_walls.append(current_wall)
    visited_walls.add(current_wall)
    next_wall = None
    for connected_wall in wall_connectivity[current_wall]:
        if connected_wall not in visited_walls:
            next_wall = connected_wall
            break
    current_wall = next_wall

# Create a mapping between wall IDs and their shortened axes
wall_id_to_shortened_axis = {wall.GlobalId: axis for wall, axis in zip(walls, shortened_axes)}

# Visualization of the combined wall axes with indexing
plt.figure(figsize=(10, 10))

import random

# Set a random seed for reproducibility
random.seed(33)  # Change this number to any integer to vary the results

# Randomly choose a starting index
start_index = random.randint(0, len(ordered_walls) - 1)

# Rotate the list to start from the chosen index
ordered_walls = ordered_walls[start_index:] + ordered_walls[:start_index]

# Plot the shortened wall axes with ordered indexing
for idx, wall_id in enumerate(ordered_walls):
    if wall_id in wall_id_to_shortened_axis:
        shortened_axis = wall_id_to_shortened_axis[wall_id]
        x_coords, y_coords = shortened_axis.xy

        # Plot shortened wall axes and add index labels
        plt.plot(x_coords, y_coords, 'b-', linewidth=2)

        # Calculate the geometric midpoint between the two endpoints of the shortened axis
        start_point = (x_coords[0], y_coords[0])
        end_point = (x_coords[-1], y_coords[-1])
        midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

        # Plot the index at the midpoint
        plt.text(midpoint[0], midpoint[1], str(idx + 1), fontsize=12, color='red', fontweight='bold', ha='center')


plt.title("Combined Shortened Wall Axes with Sequential Indexing Using Connectivity")
plt.xlabel("X-axis (meters)")
plt.ylabel("Y-axis (meters)")
plt.grid(True)
plt.axis('equal')
plt.legend(['Combined Wall Axes (blue solid)'], loc='upper right')
plt.show()

print("Sequential indexing visualization of shortened wall axes complete.")



# creating points to avoid and optimizing the given shape without changing the orientation of the walls to maximiza the are



import random
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, MultiPoint
import matplotlib.pyplot as plt

# For reproducibility, set a fixed random seed (general seed for the entire process)
random.seed(42)

# Get the ordered shortened axes from the previous cell
shortened_axes_ordered = []
for wall_id in ordered_walls:
    if wall_id in wall_id_to_shortened_axis:
        axis = wall_id_to_shortened_axis[wall_id]
        shortened_axes_ordered.append((wall_id, axis))  # Store wall ID with axis

# Ensure that shortened_axes_ordered has the correct length
if len(shortened_axes_ordered) != len(ordered_walls):
    print("Mismatch in number of walls and axes.")
else:
    print(f"Number of sides: {len(shortened_axes_ordered)}")

# Step 1: Represent each curve as an orientation point (midpoint) and store direction vectors
orientation_points = []  # List of tuples: (wall_id, orientation_point)
direction_vectors = []   # List of tuples: (wall_id, direction_vector)
for wall_id, axis in shortened_axes_ordered:
    x_coords, y_coords = axis.xy
    start_point = (x_coords[0], y_coords[0])
    end_point = (x_coords[-1], y_coords[-1])
    midpoint = ((x_coords[0] + x_coords[-1]) / 2, (y_coords[0] + y_coords[-1]) / 2)
    orientation_points.append((wall_id, midpoint))

    # Calculate direction vector
    direction_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    direction_length = (direction_vector[0]**2 + direction_vector[1]**2)**0.5
    if direction_length == 0:
        print(f"Warning: Side {wall_id} has zero length.")
        normalized_direction = (0, 0)
    else:
        normalized_direction = (direction_vector[0]/direction_length, direction_vector[1]/direction_length)
    direction_vectors.append((wall_id, normalized_direction))

# Step 2: Generate 7 random points around the shape
random_points = []
random.seed(505)  # Seed for reproducibility of random points
# Calculate the centroid of the initial orientation points
if len(orientation_points) > 0:
    x_coords = [pt[1][0] for pt in orientation_points]
    y_coords = [pt[1][1] for pt in orientation_points]
    shape_centroid = Point(sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))
else:
    shape_centroid = Point(0, 0)  # Default centroid if no orientation points

radius = 7000  # Adjust the radius to place points around the shape

for _ in range(7):
    angle = random.uniform(0, 2 * 3.14159)
    random_radius = radius * random.uniform(0.7, 1.3)
    random_x = shape_centroid.x + random_radius * np.cos(angle)
    random_y = shape_centroid.y + random_radius * np.sin(angle)
    random_points.append((random_x, random_y))

random_points_geom = [Point(p) for p in random_points]

# Step 3: Optimization Loop to Maximize Area while Avoiding Random Points
num_iterations = 2000
movement_distance = 200  # Small step size for optimization
best_area = 0
best_orientation_points = orientation_points.copy()
best_unique_intersection_points = []
best_unique_side_wall_ids = []

iteration_data = []  # To store shapes for visualization at every 100th iteration

for iteration in range(num_iterations):
    # Create a copy of orientation points to apply movement
    new_orientation_points = best_orientation_points.copy()

    # Move each orientation point randomly
    for idx in range(len(new_orientation_points)):
        wall_id, old_orientation_point = new_orientation_points[idx]
        movement_vector = (random.uniform(-movement_distance, movement_distance),
                           random.uniform(-movement_distance, movement_distance))
        new_orientation_point = (old_orientation_point[0] + movement_vector[0],
                                 old_orientation_point[1] + movement_vector[1])
        new_orientation_points[idx] = (wall_id, new_orientation_point)

    # Step 4: For each side, define an extended line based on the moved orientation point and original direction
    extended_axes = []
    for idx in range(len(new_orientation_points)):
        wall_id, orientation_point = new_orientation_points[idx]
        _, direction_vector = direction_vectors[idx]

        # Extend the line in both directions from the orientation point
        extension_factor = 10000  # Large number to extend the lines significantly
        extended_start = (
            orientation_point[0] - extension_factor * direction_vector[0],
            orientation_point[1] - extension_factor * direction_vector[1]
        )
        extended_end = (
            orientation_point[0] + extension_factor * direction_vector[0],
            orientation_point[1] + extension_factor * direction_vector[1]
        )
        extended_axis = LineString([extended_start, extended_end])
        extended_axes.append((wall_id, extended_axis))

    # Step 5: Find intersection points between each extended line and its neighbor
    intersection_points = []
    side_wall_ids = []  # To keep track of wall IDs associated with each side

    for idx in range(len(extended_axes)):
        current_wall_id, current_axis = extended_axes[idx]
        next_wall_id, next_axis = extended_axes[(idx + 1) % len(extended_axes)]  # Wrap around

        # Find intersection point
        intersection = current_axis.intersection(next_axis)

        if intersection.is_empty:
            continue

        if isinstance(intersection, Point):
            intersection_points.append((intersection.x, intersection.y))
            side_wall_ids.append(current_wall_id)
        elif isinstance(intersection, MultiPoint):
            # Take the middle point if multiple intersection points exist
            points = list(intersection)
            mid_idx = len(points) // 2
            midpoint = points[mid_idx]
            intersection_points.append((midpoint.x, midpoint.y))
            side_wall_ids.append(current_wall_id)
        elif isinstance(intersection, LineString):
            x_coords, y_coords = intersection.xy
            mid_idx = len(x_coords) // 2
            midpoint = (x_coords[mid_idx], y_coords[mid_idx])
            intersection_points.append(midpoint)
            side_wall_ids.append(current_wall_id)

    # Remove duplicate points (if any)
    unique_intersection_points = []
    unique_side_wall_ids = []
    for pt, wall_id in zip(intersection_points, side_wall_ids):
        if pt not in unique_intersection_points:
            unique_intersection_points.append(pt)
            unique_side_wall_ids.append(wall_id)

    # Step 6: Create new closed shape and evaluate area
    if len(unique_intersection_points) >= 3:
        new_shape = Polygon(unique_intersection_points)

        # Check if new shape is valid and avoid random points
        if new_shape.is_valid and all(not new_shape.contains(pt) for pt in random_points_geom):
            new_area = new_shape.area
            if new_area > best_area:
                best_area = new_area
                best_orientation_points = new_orientation_points
                best_unique_intersection_points = unique_intersection_points.copy()
                best_unique_side_wall_ids = unique_side_wall_ids.copy()

    # Store data for visualization every 100 iterations
    if (iteration + 1) % 100 == 0:
        iteration_data.append((iteration + 1, new_shape))

# Step 7: Visualization of the optimized shape over iterations
plt.figure(figsize=(10, 10))

# Plot the original blue shape (initial configuration)
for wall_id, axis in shortened_axes_ordered:
    x_coords, y_coords = axis.xy
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.5)

# Plot the random points to avoid
random_points_x, random_points_y = zip(*random_points)
plt.scatter(random_points_x, random_points_y, color='red', s=100, marker='x', label='Random Points to Avoid')

# Plot shapes from every 100th iteration
for iteration, shape in iteration_data:
    if shape and shape.is_valid:
        x_iter, y_iter = shape.exterior.xy
        plt.plot(x_iter, y_iter, label=f"Iteration {iteration}", alpha=0.3)

# Plot the final optimized shape
if best_orientation_points:
    optimized_axes = []
    for idx in range(len(best_orientation_points)):
        wall_id, orientation_point = best_orientation_points[idx]
        _, direction_vector = direction_vectors[idx]

        extended_start = (
            orientation_point[0] - extension_factor * direction_vector[0],
            orientation_point[1] - extension_factor * direction_vector[1]
        )
        extended_end = (
            orientation_point[0] + extension_factor * direction_vector[0],
            orientation_point[1] + extension_factor * direction_vector[1]
        )
        optimized_axes.append((wall_id, LineString([extended_start, extended_end])))

    intersection_points = []
    side_wall_ids = []

    for idx in range(len(optimized_axes)):
        current_wall_id, current_axis = optimized_axes[idx]
        next_wall_id, next_axis = optimized_axes[(idx + 1) % len(optimized_axes)]
        intersection = current_axis.intersection(next_axis)
        if isinstance(intersection, Point):
            intersection_points.append((intersection.x, intersection.y))
            side_wall_ids.append(current_wall_id)

    if len(intersection_points) >= 3:
        optimized_shape = Polygon(intersection_points)
        if optimized_shape.is_valid:
            x_opt, y_opt = optimized_shape.exterior.xy
            plt.plot(x_opt, y_opt, 'g-', linewidth=2, label='Optimized Shape')
            plt.fill(x_opt, y_opt, color='lightgreen', alpha=0.3)

            # Label the sides with wall IDs
            for idx in range(len(intersection_points)):
                start_point = intersection_points[idx]
                end_point = intersection_points[(idx + 1) % len(intersection_points)]
                mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
                wall_id = side_wall_ids[idx]
                plt.text(mid_point[0], mid_point[1], f"ID: {wall_id}", fontsize=8, color='purple', ha='center')

# Move the legend outside of the plot
plt.title("Shape Optimization Over 1000 Iterations (Showing Every 100th Iteration)")
plt.xlabel("X-axis (meters)")
plt.ylabel("Y-axis (meters)")
plt.grid(True)
plt.axis('equal')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust to make space for the legend outside the plot
plt.show()

print(f"Optimization complete. Maximum area found: {best_area:.2f}")







# Exporting file back to IFC format





import ifcopenshell
import uuid
import os
import numpy as np

def create_guid():
    return str(uuid.uuid4()).replace('-', '').upper()

# Create a new IFC file with the IFC2X3 schema for ArchiCAD compatibility
new_ifc = ifcopenshell.file(schema="IFC2X3")

# Basic owner history required for IFC files
owner_history = new_ifc.create_entity("IfcOwnerHistory", CreationDate=0)

# Complete IFC project structure with project, site, building, and building storey
project = new_ifc.create_entity("IfcProject", GlobalId=create_guid(), Name="OptimizedProject", OwnerHistory=owner_history)
context = new_ifc.create_entity("IfcGeometricRepresentationContext", ContextIdentifier="Model", ContextType="Model", Precision=1e-5)
project.RepresentationContexts = [context]

site = new_ifc.create_entity("IfcSite", GlobalId=create_guid(), Name="Site", OwnerHistory=owner_history)
building = new_ifc.create_entity("IfcBuilding", GlobalId=create_guid(), Name="Building", OwnerHistory=owner_history)
storey = new_ifc.create_entity("IfcBuildingStorey", GlobalId=create_guid(), Name="Storey 1", OwnerHistory=owner_history)

# Link spatial structure
new_ifc.create_entity("IfcRelAggregates", GlobalId=create_guid(), RelatingObject=project, RelatedObjects=[site])
new_ifc.create_entity("IfcRelAggregates", GlobalId=create_guid(), RelatingObject=site, RelatedObjects=[building])
new_ifc.create_entity("IfcRelAggregates", GlobalId=create_guid(), RelatingObject=building, RelatedObjects=[storey])

# Function to extract properties from the original wall
def get_wall_properties(wall):
    properties = {}
    properties['Name'] = wall.Name or f"Wall_{wall.GlobalId}"
    properties['Thickness'] = None
    properties['Height'] = None
    properties['Material'] = None

    # Extract thickness and height from geometry
    if wall.Representation:
        for rep in wall.Representation.Representations:
            if rep.is_a("IfcShapeRepresentation"):
                for item in rep.Items:
                    if item.is_a("IfcExtrudedAreaSolid"):
                        # Get the extrusion depth (Height)
                        properties['Height'] = float(item.Depth)
                        # Get the profile to get the Thickness
                        profile = item.SweptArea
                        if profile.is_a("IfcRectangleProfileDef"):
                            # For walls, XDim is thickness
                            properties['Thickness'] = float(profile.XDim)
                        elif profile.is_a("IfcArbitraryClosedProfileDef"):
                            # Handle other profile types if necessary
                            pass
                        elif profile.is_a("IfcCircleProfileDef"):
                            # Handle circular profiles
                            pass
                        elif profile.is_a("IfcDerivedProfileDef"):
                            # Handle derived profiles
                            pass
                        # Break if we've found both properties
                        if properties['Thickness'] and properties['Height']:
                            break

    # If thickness not found, try to get it from IfcMaterialLayerSetUsage
    if properties['Thickness'] is None:
        if hasattr(wall, 'HasAssociations'):
            for rel in wall.HasAssociations:
                if rel.is_a("IfcRelAssociatesMaterial"):
                    material = rel.RelatingMaterial
                    if material.is_a("IfcMaterialLayerSetUsage"):
                        layer_set = material.ForLayerSet
                        if layer_set and hasattr(layer_set, 'MaterialLayers'):
                            total_thickness = 0.0
                            for layer in layer_set.MaterialLayers:
                                total_thickness += layer.LayerThickness
                            properties['Thickness'] = float(total_thickness)
                            break

    # Get Material (if assigned)
    if hasattr(wall, 'HasAssociations'):
        for rel in wall.HasAssociations:
            if rel.is_a("IfcRelAssociatesMaterial"):
                material = rel.RelatingMaterial
                if material.is_a("IfcMaterial"):
                    properties['Material'] = material.Name
                elif material.is_a("IfcMaterialLayerSetUsage"):
                    layer_set = material.ForLayerSet
                    if layer_set and hasattr(layer_set, 'MaterialLayers'):
                        # Collect material names from layers
                        materials = []
                        for layer in layer_set.MaterialLayers:
                            if layer.Material:
                                materials.append(layer.Material.Name)
                        properties['Material'] = ','.join(materials)
                elif material.is_a("IfcMaterialList"):
                    # Handle IfcMaterialList if necessary
                    pass

    # Provide default values if properties are not found
    if properties['Thickness'] is None:
        properties['Thickness'] = 0.2  # Default thickness
    if properties['Height'] is None:
        properties['Height'] = 3.0     # Default height
    if properties['Material'] is None:
        properties['Material'] = "DefaultMaterial"

    return properties

# Extract wall properties from the original IFC file
wall_properties = {}
for wall in walls:
    properties = get_wall_properties(wall)
    wall_properties[wall.GlobalId] = properties
    # Print properties for debugging
    print(f"Wall ID: {wall.GlobalId}")
    print(f"  Name: {properties['Name']}")
    print(f"  Thickness: {properties['Thickness']} meters")
    print(f"  Height: {properties['Height']} meters")
    print(f"  Material: {properties['Material']}")

# Create new walls in the IFC file based on the optimized shape
for idx in range(len(best_unique_intersection_points)):
    start_point = best_unique_intersection_points[idx]
    end_point = best_unique_intersection_points[(idx + 1) % len(best_unique_intersection_points)]
    wall_id = best_unique_side_wall_ids[idx]
    properties = wall_properties.get(wall_id, {})

    # Ensure start and end points are 3D points with plain floats
    start_point_3d = (float(start_point[0]), float(start_point[1]), 0.0)
    end_point_3d = (float(end_point[0]), float(end_point[1]), 0.0)

    # Calculate the direction vector for the wall
    direction_vector = (
        end_point_3d[0] - start_point_3d[0],
        end_point_3d[1] - start_point_3d[1],
        end_point_3d[2] - start_point_3d[2]
    )

    # Calculate the length of the wall
    wall_length = (direction_vector[0]**2 + direction_vector[1]**2)**0.5

    if wall_length == 0:
        print(f"Warning: Wall {wall_id} has zero length.")
        continue

    # Normalize the direction vector
    direction_vector_normalized = (
        direction_vector[0]/wall_length,
        direction_vector[1]/wall_length,
        0.0
    )

    # Ensure all elements are plain floats
    direction_vector_normalized = tuple(float(val) for val in direction_vector_normalized)

    # Create IfcWallStandardCase with inherited name and owner history
    wall = new_ifc.create_entity("IfcWallStandardCase", GlobalId=create_guid(), Name=properties.get('Name', f"Wall_{idx+1}"), OwnerHistory=owner_history)

    # Define the wall's local placement
    wall_placement = new_ifc.create_entity("IfcLocalPlacement",
        PlacementRelTo=storey.ObjectPlacement,
        RelativePlacement=new_ifc.create_entity("IfcAxis2Placement3D",
            Location=new_ifc.create_entity("IfcCartesianPoint", Coordinates=[float(coord) for coord in start_point_3d]),
            RefDirection=new_ifc.create_entity("IfcDirection", DirectionRatios=[float(val) for val in direction_vector_normalized]),
            Axis=new_ifc.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0])
        )
    )
    wall.ObjectPlacement = wall_placement

    # Geometry definition: axis representation
    axis_points = [
        new_ifc.create_entity("IfcCartesianPoint", Coordinates=[0.0, 0.0, 0.0]),
        new_ifc.create_entity("IfcCartesianPoint", Coordinates=[float(wall_length), 0.0, 0.0])
    ]
    axis_polyline = new_ifc.create_entity("IfcPolyline", Points=axis_points)
    axis_representation = new_ifc.create_entity("IfcShapeRepresentation", ContextOfItems=context, RepresentationIdentifier="Axis", RepresentationType="Curve3D", Items=[axis_polyline])

    # Profile definition using inherited thickness
    thickness = properties.get('Thickness', 0.2)
    profile = new_ifc.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=float(thickness), YDim=float(thickness))

    # Extrusion direction and depth using inherited height
    extrude_direction = new_ifc.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0])
    depth = float(properties.get('Height', 3.0))

    # Positioning the profile at the axis origin (centered)
    profile_placement = new_ifc.create_entity("IfcAxis2Placement2D",
        Location=new_ifc.create_entity("IfcCartesianPoint", Coordinates=[-float(thickness)/2.0, -float(thickness)/2.0])
    )

    # Create the swept area solid
    swept_area_solid = new_ifc.create_entity("IfcExtrudedAreaSolid", SweptArea=profile, Position=profile_placement, ExtrudedDirection=extrude_direction, Depth=depth)

    # Body representation
    body_representation = new_ifc.create_entity("IfcShapeRepresentation", ContextOfItems=context, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[swept_area_solid])

    # Assign representations to the wall
    wall.Representation = new_ifc.create_entity("IfcProductDefinitionShape", Representations=[axis_representation, body_representation])

    # Contain the wall in the storey spatial structure
    new_ifc.create_entity("IfcRelContainedInSpatialStructure", GlobalId=create_guid(), RelatingStructure=storey, RelatedElements=[wall])

    # Add property set with inherited properties
    pset = new_ifc.create_entity("IfcPropertySet", GlobalId=create_guid(), Name="Pset_WallCommon")
    properties_list = [
        new_ifc.create_entity("IfcPropertySingleValue", Name="Reference", NominalValue=new_ifc.create_entity("IfcIdentifier", properties.get('Name', f"Wall_{idx+1}"))),
        new_ifc.create_entity("IfcPropertySingleValue", Name="IsExternal", NominalValue=new_ifc.create_entity("IfcBoolean", True)),
        new_ifc.create_entity("IfcPropertySingleValue", Name="LoadBearing", NominalValue=new_ifc.create_entity("IfcBoolean", False)),
        new_ifc.create_entity("IfcPropertySingleValue", Name="Thickness", NominalValue=new_ifc.create_entity("IfcPositiveLengthMeasure", float(thickness))),
        new_ifc.create_entity("IfcPropertySingleValue", Name="Height", NominalValue=new_ifc.create_entity("IfcPositiveLengthMeasure", float(depth))),
        new_ifc.create_entity("IfcPropertySingleValue", Name="Material", NominalValue=new_ifc.create_entity("IfcLabel", properties.get('Material', 'DefaultMaterial')))
    ]
    pset.HasProperties = properties_list
    new_ifc.create_entity("IfcRelDefinesByProperties", GlobalId=create_guid(), RelatedObjects=[wall], RelatingPropertyDefinition=pset)

    # Assign material to the wall if available
    material_name = properties.get('Material', 'DefaultMaterial')
    material = new_ifc.create_entity("IfcMaterial", Name=material_name)
    new_ifc.create_entity("IfcRelAssociatesMaterial", GlobalId=create_guid(), RelatingMaterial=material, RelatedObjects=[wall])

# Write the IFC file
output_file = 'optimized_shape.ifc'
new_ifc.write(output_file)

print("Optimized walls have been exported to:", os.path.abspath(output_file))







