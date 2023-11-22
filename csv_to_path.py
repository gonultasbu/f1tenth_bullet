import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_from_csv():
    csv_file_path = r'C:\Users\ogpoy\Documents\GitHub\f1tenth_bullet\data\friction\ftg_friction_ftg_cnst_vel_18.csv'
    data = pd.read_csv(csv_file_path)

    # Extract the first and last column to create a 2D numpy array named 'waypoints'
    waypoints = data.iloc[:, [0, -1]].values

    # Check how many items in the array
    num_items = waypoints.size

    # If the number of items are bigger than 4000, decrease the size of the array by half
    # by selecting value in every 2 elements
    if num_items > 4000:
        waypoints = waypoints[::150]

    # Get the new number of items in the array after potential downsizing
    new_num_items = waypoints.size

    return waypoints

def compute_path(waypoints):
    
    wp_x = waypoints[:, 0]
    wp_y = waypoints[:, 1]

    dx = np.append(0, np.diff(wp_x))
    dy = np.append(0, np.diff(-wp_y))
    theta = np.arctan2(dy, dx)
    path = np.vstack((wp_x[5:-1], -wp_y[5:-1], theta[5:-1]))
    path = path[:, ::-1]
    return path

if __name__ == "__main__":
    waypoints = generate_from_csv()
    final_wp_2 = compute_path(waypoints)

    plt.figure(4)
    plt.plot(final_wp_2[0,:], final_wp_2[1,:], 'k*', label='Center Line')
    plt.plot(final_wp_2[:,-1][0],final_wp_2[:,-1][1], 'ro')
    plt.plot(final_wp_2[:,0][0],final_wp_2[:,0][1], 'bo')
    plt.plot(final_wp_2[:,1][0],final_wp_2[:,1][1], 'go')
    # # plt.plot(arr_path[0,:], -arr_path[1,:], 'k*', label='Center Line')
    # # plt.plot(0, 0, 'ko', label='Start Point (Origin)')  # Mark the start point at the origin

    # plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordinates
    # plt.title('Track with Center Line')
    # plt.xlabel('X coordinate')
    # plt.ylabel('Y coordinate')
    # plt.legend()
    # plt.axis('equal')  # Equal scaling for x and y axes
    # plt.grid(True)  # Enable grid for easier visualization of the origin
    plt.show()