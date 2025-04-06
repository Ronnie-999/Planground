import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point


# Generate a list of random points that the quadrilateral should avoid
num_points_to_avoid = 5
random_points = np.random.uniform(0, 10, (num_points_to_avoid, 2))

print("Random Points to Avoid:")
print(random_points)



# Objective function to maximize area while avoiding certain random points
def irregular_area_with_avoidance(params):
    # Unpack points
    x1, y1, x2, y2, x3, y3, x4, y4 = params

    # Create a quadrilateral using Shapely Polygon
    quadrilateral = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    # Ensure the quadrilateral is valid (e.g., not self-intersecting)
    if not quadrilateral.is_valid or quadrilateral.area <= 0:
        return 1e6  # High penalty for invalid shapes

    # Check if any of the random points are inside the quadrilateral
    penalty = 0
    for point in random_points:
        p = Point(point)
        if quadrilateral.contains(p):
            penalty += 1000  # Large penalty if the quadrilateral contains the point

    # Calculate the area of the quadrilateral
    area = quadrilateral.area

    # Return the negative of area minus penalty (maximize area)
    return -(area - penalty)

print("Objective function for maximizing area while avoiding points defined")



# List to store progress of each iteration
iteration_data = []

def callback(params, convergence):
    # Store the current parameters and the current value of the objective function
    x1, y1, x2, y2, x3, y3, x4, y4 = params
    current_area = -(irregular_area_with_avoidance(params))  # Negate back to get the positive area
    iteration_data.append((x1, y1, x2, y2, x3, y3, x4, y4, current_area))




# Define bounds for each point coordinate (x1, y1, x2, y2, x3, y3, x4, y4)
bounds = [(0, 10)] * 8

# Run differential evolution with the defined objective function and callback
result = differential_evolution(irregular_area_with_avoidance, bounds, callback=callback)

print(f"Optimized Parameters: {result.x}")
print(f"Maximized Area: {-result.fun:.2f}")




# Visualization of the quadrilateral evolution (every tenth iteration)
plt.figure(figsize=(10, 10))

# Plot the random points to avoid
plt.scatter(random_points[:, 0], random_points[:, 1], color='red', s=100, marker='x', label='Points to Avoid')

# Iterate over the recorded data every tenth iteration
for i, data in enumerate(iteration_data):
    if i % 10 == 0:  # Every tenth iteration
        x1, y1, x2, y2, x3, y3, x4, y4, area = data

        # Coordinates to plot the quadrilateral (closing the loop)
        x_coords = [x1, x2, x3, x4, x1]
        y_coords = [y1, y2, y3, y4, y1]

        # Plot each quadrilateral evolution
        plt.plot(x_coords, y_coords, label=f"Iteration {i}", alpha=0.3)

# Plot the final quadrilateral in a different color
x1, y1, x2, y2, x3, y3, x4, y4 = result.x
x_coords = [x1, x2, x3, x4, x1]
y_coords = [y1, y2, y3, y4, y1]

plt.plot(x_coords, y_coords, 'b-', linewidth=2, label="Final Quadrilateral")
plt.fill(x_coords, y_coords, 'skyblue', alpha=0.3)

plt.title("Quadrilateral Evolution Avoiding Random Points (Every Tenth Iteration)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()




