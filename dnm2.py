import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the track map and start position data
maps_data = np.load(r"C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\tracks\circle_cw\maps\maps.npz")
starts_data = np.load(r"C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\tracks\circle_cw\maps\starts.npz")

# Extract the drivable area and start positions
drivable_area = np.uint8(maps_data['drivable_area']) * 255
start_positions = starts_data['data']

# Find contours in the drivable area
contours, hierarchy = cv2.findContours(drivable_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Assume the largest contours are the outer and inner track lines
outer_track_line = sorted_contours[0].squeeze()
inner_track_line = sorted_contours[1].squeeze() if len(sorted_contours) > 1 else None

# Use the first start position to translate the track lines
start_point = start_positions[0, :2]
outer_track_line = outer_track_line.astype(np.float64)
outer_track_line -= start_point

if inner_track_line is not None:
    inner_track_line = inner_track_line.astype(np.float64)
    inner_track_line -= start_point

# Function to find the closest point on a track line
def find_closest_point(point, points):
    distances = np.linalg.norm(points - point, axis=1)
    closest_point_index = np.argmin(distances)
    return points[closest_point_index]

# Determine the track line with more points
if len(outer_track_line) > len(inner_track_line):
    longer_track_line, shorter_track_line = outer_track_line, inner_track_line
else:
    longer_track_line, shorter_track_line = inner_track_line, outer_track_line

# Calculate the centerline by finding midpoints between the closest points on the track lines
centerline_points = []
for point in longer_track_line:
    closest_point = find_closest_point(point, shorter_track_line)
    midpoint = (point + closest_point) / 2
    centerline_points.append(midpoint)

# Convert the list of midpoints to a numpy array
centerline = np.array(centerline_points)

# Plot the outer track line, inner track line, and centerline
plt.figure(figsize=(10, 10))
plt.plot(outer_track_line[:, 0], outer_track_line[:, 1], 'b-', label='Outer Track Line')
if inner_track_line is not None:
    plt.plot(inner_track_line[:, 0], inner_track_line[:, 1], 'r-', label='Inner Track Line')
plt.plot(centerline[:, 0], centerline[:, 1], 'g-', label='Center Line')
plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordinates
plt.title('Track with Center Line')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.axis('equal')  # Equal scaling for x and y axes
plt.grid(True)
plt.show()
