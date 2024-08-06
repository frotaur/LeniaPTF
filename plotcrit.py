import matplotlib.pyplot as plt
import pickle as pk
# List of points
with open('./crit_points_list.pk','rb') as f:
    points = pk.load(f)

# Separate the points into x and y coordinates
y, x = zip(*points)

# Create the scatter plot
plt.scatter(x, y)
# Set axis limits
plt.xlim(0, 0.12)
plt.ylim(0.1, 0.5)
# Add labels and title if needed
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Points')

# Show the plot
plt.show()