# Objective function to maximize area of an irregular quadrilateral
def irregular_area(params):
    # Unpack points
    x1, y1, x2, y2, x3, y3, x4, y4 = params

    # Shoelace formula for area of a quadrilateral
    area = 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

    # Penalty for invalid configuration (e.g., crossing lines)
    penalty = 0

    # Constraints to ensure proper shape, e.g., convex shape
    if not (x1 < x2 and x4 < x3 and y4 > y1 and y3 > y2):
        penalty += 1000  # Arbitrary large penalty for invalid configuration

    return -(area - penalty)

print("New objective function for irregular quadrilateral defined")



# List to store progress of each iteration
iteration_data = []

def callback(params, convergence):
    # Store the current parameters and the current value of the objective function
    x1, y1, x2, y2, x3, y3, x4, y4 = params
    current_area = -(irregular_area(params))  # Negate back to get the positive area
    iteration_data.append((x1, y1, x2, y2, x3, y3, x4, y4, current_area))


# Define bounds for each point coordinate (x1, y1, x2, y2, x3, y3, x4, y4)
bounds = [(0, 10)] * 8

# Run differential evolution with the new objective function and callback
result = differential_evolution(irregular_area, bounds, callback=callback)

print(f"Optimized Parameters: {result.x}")
print(f"Maximized Area: {-result.fun:.2f}")




# Visualization of the quadrilateral evolution (every tenth iteration)
plt.figure(figsize=(10, 10))

# Iterate over the recorded data every tenth iteration
for i, data in enumerate(iteration_data):
    if i % 10 == 0:  # Every tenth iteration
        x1, y1, x2, y2, x3, y3, x4, y4, area = data

        # Coordinates to plot the quadrilateral (closing the loop)
        x_coords = [x1, x2, x3, x4, x1]
        y_coords = [y1, y2, y3, y4, y1]

        # Plot each quadrilateral evolution
        plt.plot(x_coords, y_coords, label=f"Iteration {i}", alpha=0.5)

# Plot the final quadrilateral in a different color
x1, y1, x2, y2, x3, y3, x4, y4 = result.x
x_coords = [x1, x2, x3, x4, x1]
y_coords = [y1, y2, y3, y4, y1]

plt.plot(x_coords, y_coords, 'r-', linewidth=2, label="Final Quadrilateral")
plt.fill(x_coords, y_coords, 'skyblue', alpha=0.3)
plt.title("Quadrilateral Evolution During Optimization (Every Tenth Iteration)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()





