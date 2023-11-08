import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

SCALE_CONSTANT = 413/20


def create_waypoints():
    # Read the map and starts values
    maps_data = np.load(r"C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\tracks\barcelona\maps\maps.npz")
    # starts_value = np.load(r"C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\tracks\barcelona\maps\starts.npz")


    # Convert the boolean array to uint8 type
    drivable_area_uint8 = np.uint8(maps_data['drivable_area']) * 255

    # Find contours from the drivable area - this should give us the track boundaries
    contours, hierarchy = cv2.findContours(drivable_area_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Since there might be multiple contours, we pick the longest two which should correspond to the inner and outer track lines
    # Sorting contours by length in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assuming the two largest contours correspond to the outer and inner track lines
    outer_track_line = sorted_contours[0]
    inner_track_line = sorted_contours[1] if len(sorted_contours) > 1 else None

    # Convert contours to a more manageable form (list of points)
    outer_track_line_points = outer_track_line.squeeze().tolist()
    inner_track_line_points = inner_track_line.squeeze().tolist() if inner_track_line is not None else None

    outer_track_line_array = np.array(outer_track_line_points)
    inner_track_line_array = np.array(inner_track_line_points)

    outer_is_closed = np.array_equal(outer_track_line_array[0], outer_track_line_array[-1])
    inner_is_closed = np.array_equal(inner_track_line_array[0], inner_track_line_array[-1])

    # If not closed, append the first point to the end of the array
    if not outer_is_closed:
        outer_track_line_array = np.vstack([outer_track_line_array, outer_track_line_array[0]])
    if not inner_is_closed:
        inner_track_line_array = np.vstack([inner_track_line_array, inner_track_line_array[0]])



    # Re-extract x and y coordinates from the updated closed track lines for plotting
    closed_outer_x, closed_outer_y = outer_track_line_array[:, 0], outer_track_line_array[:, 1]
    closed_inner_x, closed_inner_y = inner_track_line_array[:, 0], inner_track_line_array[:, 1]

    # # Create a plot to visualize the closed track lines
    # plt.figure(1)

    # # Plotting the closed outer track line
    # plt.plot(closed_outer_x, closed_outer_y, 'b-', label='Closed Outer Track Line')

    # # Plotting the closed inner track line
    # plt.plot(closed_inner_x, closed_inner_y, 'r-', label='Closed Inner Track Line')

    # plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordin    ates
    # plt.title('Closed Track Lines')
    # plt.xlabel('X coordinate')
    # plt.ylabel('Y coordinate')
    # plt.legend()
    # plt.axis('equal')  # Equal scaling for x and y axes
    # plt.show()

    outer_track_line_array_translated = (outer_track_line_array - np.ones_like(outer_track_line_array)*1000) / SCALE_CONSTANT
    inner_track_line_array_translated = (inner_track_line_array - np.ones_like(inner_track_line_array)*1000) / SCALE_CONSTANT

    # plt.figure(2)
    # plt.xlim(-1000, 1000)
    # plt.ylim(1000, -1000)

    closed_outer_xT, closed_outer_yT = outer_track_line_array_translated[:, 0], outer_track_line_array_translated[:, 1]
    closed_inner_xT, closed_inner_yT = inner_track_line_array_translated[:, 0], inner_track_line_array_translated[:, 1]
    # # Plotting the closed outer track line
    # plt.plot(closed_outer_xT, -closed_outer_yT, 'b-', label='Closed Outer Track Line')

    # # Plotting the closed inner track line
    # plt.plot(closed_inner_xT, -closed_inner_yT, 'r-', label='Closed Inner Track Line')

    # plt.show()


    def find_closest_point(point, points):
        """ Find the closest point from a list of points. """
        # Calculate the distance from the point to all other points
        distances = np.sqrt((points[:,0] - point[0])**2 + (points[:,1] - point[1])**2)
        # Find the index of the closest point
        closest_point_idx = np.argmin(distances)
        # Return the closest point
        return points[closest_point_idx]

    # Determine which array is longer to iterate over it
    if len(closed_outer_xT) > len(closed_inner_xT):
        longer_line_x, longer_line_y = closed_outer_xT, closed_outer_yT
        shorter_line = inner_track_line_array_translated
    else:
        longer_line_x, longer_line_y = closed_inner_xT, closed_inner_yT
        shorter_line = outer_track_line_array_translated

    # Initialize the centerline points array
    centerline_points = []

    # Iterate over the longer line and find midpoints to the closest points on the shorter line
    for i in range(len(longer_line_x)):
        point_longer_line = np.array([longer_line_x[i], longer_line_y[i]])
        closest_point_shorter_line = find_closest_point(point_longer_line, shorter_line)
        midpoint = (point_longer_line + closest_point_shorter_line) / 2
        centerline_points.append(midpoint)

    # Convert the centerline points to an array
    centerline_points_array = np.array(centerline_points)

    # Extract the x and y coordinates for plotting
    centerline_x = centerline_points_array[:, 0]
    centerline_y = centerline_points_array[:, 1]

    sparsed_arr =  waypoints_sparsification(centerline_points_array)
    arr = waypoint_add(-sparsed_arr)
    sparsed_arr_x = sparsed_arr[:, 0]
    sparsed_arr_y = sparsed_arr[:, 1]
    # arr_path = compute_path_from_wp(sparsed_arr_x,sparsed_arr_y)

    arr_x = arr[:, 0]
    arr_y = arr[:, 1]
    plt.figure(3)
    plt.plot(-closed_outer_xT, -closed_outer_yT, 'b-', label='Outer Track Line')
    plt.plot(-closed_inner_xT, -closed_inner_yT, 'r-', label='Inner Track Line')
    plt.plot(arr_x, arr_y, 'k*', label='Center Line')
    # plt.plot(arr_path[0,:], -arr_path[1,:], 'k*', label='Center Line')
    # plt.plot(0, 0, 'ko', label='Start Point (Origin)')  # Mark the start point at the origin

    plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordinates
    plt.title('Track with Center Line')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.grid(True)  # Enable grid for easier visualization of the origin
    plt.show()

    return arr


def distance(x,y):
    return np.sqrt(x**2 + y**2)

def waypoints_sparsification(waypoints,threshold=1.0):
    THRESHOLD = threshold # Trail/Error ?? 
    new_waypoints = waypoints[0]
    wp_x = waypoints[:, 0]
    wp_y = waypoints[:, 1]

    # Find the differences 
    dist_arr = distance(np.diff(wp_x),np.diff(wp_y))
    j = 0
    for i,dist in enumerate(dist_arr):
        if j > i:
            continue
        else:
            if dist >= THRESHOLD:
                new_waypoints = np.vstack((new_waypoints, waypoints[i+1]))
            else:
                temp = 0
                j = i+1
                while temp < THRESHOLD:
                    if j != len(dist_arr):
                        temp += dist_arr[j]
                        j += 1
                    else:
                        break
                new_waypoints = np.vstack((new_waypoints, waypoints[j]))
        
    new_waypoints_arr = np.array(new_waypoints)

    return new_waypoints_arr

def waypoint_add(waypoints):
    THRESHOLD = 1
    new_waypoints = waypoints[0]
    wp_x = waypoints[:, 0]
    wp_y = waypoints[:, 1]

    dist_arr = distance(np.diff(wp_x),np.diff(wp_y))

    for idx, dist in enumerate(dist_arr):
        if dist > THRESHOLD:
            #  find the x and y distance
            numofPts = np.floor(dist / THRESHOLD).astype(int)
            wp_xs = waypoints[idx : idx + 2,0]
            wp_ys = waypoints[idx : idx + 2,1]
            
            pts_xs = np.linspace(wp_xs[0], wp_xs[1], numofPts + 2)
            pts_ys = np.linspace(wp_ys[0], wp_ys[1], numofPts + 2)
            if np.all(new_waypoints[-1] == waypoints[idx]): 
                    temp =  np.zeros((len(pts_xs[1:]),2))
                    temp[:,0] = pts_xs[1:]
                    temp[:,1] = pts_ys[1:]
            else:
                    temp =  np.zeros((len(pts_xs),2))
                    temp[:,0] = pts_xs
                    temp[:,1] = pts_ys
            new_waypoints = np.vstack((new_waypoints, temp))
        else:
            if np.all(new_waypoints[-1] == waypoints[idx]):
                continue
            else:
                new_waypoints = np.vstack((new_waypoints, waypoints[idx])) 
    
    # new_waypoints = waypoints_sparsification(new_waypoints, 1)
    return new_waypoints


def compute_path_from_wp(start_xp, start_yp, step=2):
    """

    Args:
        start_xp (array-like): 1D array of x-positions
        start_yp (array-like): 1D array of y-positions
        step (float): intepolation step

    Returns:
        ndarray of shape (3,N) representing the  path as x,y,heading
    """
    final_xp = []
    final_yp = []
    delta = step  # [m]
    for idx in range(len(start_xp) - 1):
        section_len = np.sum(
            np.sqrt(
                np.power(np.diff(start_xp[idx : idx + 2]), 2)
                + np.power(np.diff(start_yp[idx : idx + 2]), 2)
            )
        )
        interp_range = np.linspace(0, 1, np.floor(section_len / delta).astype(int))
        fx = interp1d(np.linspace(0, 1, 2), start_xp[idx : idx + 2], kind=1)
        fy = interp1d(np.linspace(0, 1, 2), start_yp[idx : idx + 2], kind=1)
        # watch out to duplicate points!
        final_xp = np.append(final_xp, fx(interp_range)[1:])
        final_yp = np.append(final_yp, fy(interp_range)[1:])
    dx = np.append(0, np.diff(final_xp))
    dy = np.append(0, np.diff(final_yp))
    theta = np.arctan2(dy, dx)
    return np.vstack((final_xp, final_yp, theta))


def compute_path(waypoints):
    
    wp_x = waypoints[:, 0]
    wp_y = waypoints[:, 1]

    dx = np.append(0, np.diff(wp_x))
    dy = np.append(0, np.diff(wp_y))
    theta = np.arctan2(dy, dx)

    return np.vstack((wp_x, wp_y, theta))


if __name__ == "__main__":
    waypoints = create_waypoints()
    # final_wp = compute_path_from_wp(waypoints[:,0],waypoints[:,1])
    final_wp_2 = compute_path(waypoints)
    # if the waypoints are two close to each other Ignore the next one
