import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from waypoint_creation import create_waypoints


waypoints = create_waypoints()
print(waypoints.shape)
# map_value = np.load(r"C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\tracks\barcelona\maps\maps.npz")
# starts_value = np.load(r"C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\tracks\barcelona\maps\starts.npz")

# keys = np.array(map_value.files)
# starts_value_keys  = np.array(starts_value.files)
# print("Keys as NumPy array:", keys)
# array_dict = {key: map_value[key] for key in map_value.files}
# print("Keys as NumPy array:", starts_value_keys)
# starts_dict = {key: starts_value[key] for key in starts_value.files}
# # Now array_dict is a dictionary with your data
 
# drive = array_dict["drivable_area"]
# norm_dist_start = array_dict["norm_distance_from_start"]
# norm_dist_to = array_dict["norm_distance_to"]
# norm_dist_obstacle = array_dict["norm_distance_to_obstacle"]
# img_drive = norm_dist_obstacle.astype(int)


# data = starts_dict["data"]
# print(data)



# plt.imshow(norm_dist_obstacle, cmap="gray")
# # plt.colorbar()
# print(array_dict["properties"])
# plt.show()


# print(5)

# def find_closest_point(point, points):
#     """ Find the closest point from a list of points. """
#     # Calculate the distance from the point to all other points
#     distances = np.sqrt((points[:,0] - point[0])**2 + (points[:,1] - point[1])**2)
#     # Find the index of the closest point
#     closest_point_idx = np.argmin(distances)
#     # Return the closest point
#     return points[closest_point_idx]

# # Determine which array is longer to iterate over it
# if len(trans_outer_x) > len(trans_inner_x):
#     longer_line_x, longer_line_y = trans_outer_x, trans_outer_y
#     shorter_line = translated_inner_track_line
# else:
#     longer_line_x, longer_line_y = trans_inner_x, trans_inner_y
#     shorter_line = translated_outer_track_line

# # Initialize the centerline points array
# centerline_points = []

# # Iterate over the longer line and find midpoints to the closest points on the shorter line
# for i in range(len(longer_line_x)):
#     point_longer_line = np.array([longer_line_x[i], longer_line_y[i]])
#     closest_point_shorter_line = find_closest_point(point_longer_line, shorter_line)
#     midpoint = (point_longer_line + closest_point_shorter_line) / 2
#     centerline_points.append(midpoint)

# # Convert the centerline points to an array
# centerline_points_array = np.array(centerline_points)

# # Extract the x and y coordinates for plotting
# centerline_x = centerline_points_array[:, 0]
# centerline_y = centerline_points_array[:, 1]

# # Plot the centerline
# plt.figure(figsize=(10, 10))
# plt.plot(trans_outer_x, trans_outer_y, 'b-', label='Outer Track Line')
# plt.plot(trans_inner_x, trans_inner_y, 'r-', label='Inner Track Line')
# plt.plot(centerline_x, centerline_y, 'g-', label='Center Line')
# plt.plot(0, 0, 'ko', label='Start Point (Origin)')

# plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordinates
# plt.title('Track with Center Line')
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.legend()
# plt.axis('equal')  # Equal scaling for x and y axes
# plt.grid(True)
# plt.show()
