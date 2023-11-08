import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import place_poles
import math
# Initial State


Q = np.diag([0.3, 0.3, np.deg2rad(1.0), 0.3])**2
R = np.diag([0.5, 0.5, np.deg2rad(3.0), 0.1])**2
dt = 1  # Updated timestep

L = 4.0  # Wheelbase
phi = np.deg2rad(0)  # Steering angle

def bicycle_model(state, dt, L, phi):
    x, y, theta, v = state
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(phi) * dt
    return np.array([x, y, theta, v])

def get_measurement(state, R):
    return state + np.sqrt(R) @ np.random.randn(4)

def create_K_matrix():
    m = 3.1
    Iz = 0.04712
    g = 9.81
    lf = 0.15875  #0.1621 #0.15875         #0.1653
    lr =  0.17145 #0.1425 #0.17145 	      #0.1494

    multiplier_f = ((m * g) / (1 + (lf/lr)))*(3.14/180)

    multiplier_r = ((m*g) - ((m*g) / (1 + (lf/lr))))*(3.14/180)

    Caf =  4.718 * multiplier_f  #4.718 * 0.275  #10.310 * 0.2513  #5.1134
    Car =  5.4562  * multiplier_r  #5.4562 * 0.254 #8.522 * 0.2779 #4.9883
    Vx = 5
    a_1 = np.array([0,1,0,0],dtype=np.float64)
    a_2 = np.hstack((0,-((2*Caf)+(2*Car))/(m*Vx),((2*Caf)+(2*Car))/(m),((-2*Caf*lf)+(2*Car*lr))/(m*Vx)))
    a_3 = np.array([0,0,0,1],dtype=np.float64)
    a_4 = np.hstack((0, -((2*Caf*lf)-(2*Car*lr))/(Iz*Vx), ((2*Caf*lf)-(2*Car*lr))/(Iz), -((2*Caf*(lf**2))+(2*Car*(lr**2)))/(Iz*Vx)))
    A = np.vstack((a_1,a_2,a_3,a_4))
    B1 = np.vstack((
            np.array(0,dtype=np.float64),
            (2*Caf)/(m),
            np.array(0,dtype=np.float64),
            (2*Caf*lf)/(Iz)))
    # Pol = np.array([-5,-4,-7,-10], dtype=np.float64)
    Pol = np.array([-5-3j,-5+3j,-7,-10])
    # A = np.array([[0, 1, 0, 0],
    # 	[-7438, -1128, -543.6, 503.3],
    # 	[0, 0, 0, 1],
    # 	[-16400, -2472, -1199, 1110]])
    # B1 = np.array([[-125.9, -2103, 344.5, -2529]]).T
    K = place_poles(A, B1, Pol).gain_matrix
    A = A
    B1 = B1
    return K

def distance(pt1, pt2):
    return math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))

data = pd.read_csv('/mnt/c/Users/ogpoy/Documents/predictive_methods/racetrack-database/tracks/Spielberg.csv')

# Extract data into numpy arrays
x = data['# x_m'].values
y = data['y_m'].values
w_tr_right = data['w_tr_right_m'].values
w_tr_left = data['w_tr_left_m'].values


delta_x = np.diff(x)
delta_y = np.diff(y)
headings = np.arctan2(delta_y, delta_x)
headings = np.append(headings, headings[-1])
# Constant Velocity 
vel_arr = np.ones_like(x) * np.floor(np.sqrt(delta_x[0]**2+delta_y[0]**2)/dt)
actual_path = np.column_stack((x, y, headings,vel_arr))
measured_path = np.zeros_like(actual_path)

x0 = actual_path[0]

for i,state in enumerate(actual_path):
    measured_path[i] = get_measurement(state, R)
###### FILTERS ############

def kf_predict(X, P, A, Q):
    X_pred = A @ X
    P_pred = A @ P @ A.T + Q
    return X_pred, P_pred

def kf_update(X_pred, P_pred, Z, H, R):
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    X_update = X_pred + K @ (Z - H @ X_pred)
    P_update = (np.eye(len(X_pred)) - K @ H) @ P_pred
    return X_update, P_update



# phi calculations
i = 0
data_timestamp_prev = 0
smoothing_c = 0.001
e_1_dot = 0.0
e_2_dot = 0.0
angle_error_prev = 0.0
angle_error_prev_2 = 0.0 
vel_error_prev = 0.0
xz_prev = np.array([0.0,0.0])
e_1_prev = 0.0
e_2_prev = 0.0
target_angles_list = []
target_vels_list = []
e_list = []
""" 
Refer to https://github.com/jainachin/bayesrace/blob/master/bayes_race/params/f110.py 
for these params.
"""

K = create_K_matrix()
#Find error values
e1 = []
e2 = []
for i in range(len(actual_path)):
    #for e1
    # find ecludian distance between actual and measured
    #small angle approximation dist = dist*cos(theta)
    e1.append(distance(actual_path[i][:2],measured_path[i][:2]))
    e2.append(measured_path[i][2] - actual_path[i][2])


#For e1_dot and e2_dot
e1_diff = np.diff(e1)
e2_diff = np.diff(e2)
e1_diff = np.append(e1_diff, e1_diff[-1])
e2_diff = np.append(e2_diff, e2_diff[-1])

e1_dots = e1_diff/dt
e2_dots = e2_diff/dt

e1_arr = np.array(e1)
e2_arr = np.array(e2)


e_states = np.column_stack((e1_arr,e1_dots,e2_arr,e2_dots))


#find steering angle
phis = np.zeros_like(e1_arr)

for i in range(len(e_states)):
    phis[i] = -K @ e_states[i].T

# Linearized Bicycle Model (A matrix)
A = np.array([
    [1, 0, -dt*x0[3]*np.sin(x0[2]), dt*np.cos(x0[2])],
    [0, 1, dt*x0[3]*np.cos(x0[2]), dt*np.sin(x0[2])],
    [0, 0, 1, dt/L*np.tan(phis[0])],
    [0, 0, 0, 1]
])

# Measurement Matrix
H = np.eye(4)
H[-1,-1] = 1

# KF Initialization
X_kf = x0
P = np.eye(4) * 0.1
kf_path = [X_kf]


for i,z in enumerate(measured_path[1:]):
    A = np.array([
    [1, 0, -dt*z[3]*np.sin(z[2]), dt*np.cos(z[2])],
    [0, 1, dt*z[3]*np.cos(z[2]), dt*np.sin(z[2])],
    [0, 0, 1, dt/L*np.tan(phis[i+1])],
    [0, 0, 0, 1]
])

    X_kf, P = kf_predict(X_kf, P, A, Q)
    X_kf, P = kf_update(X_kf, P, z, H, R)
    kf_path.append(X_kf)



###### EXTENDED KALMAN FILTER ################

def ekf_predict(X, P, Q, dt, L, phi):
    X_pred = bicycle_model(X, dt, L, phi)
    A = np.array([
        [1, 0, -dt*X[3]*np.sin(X[2]), dt*np.cos(X[2])],
        [0, 1, dt*X[3]*np.cos(X[2]), dt*np.sin(X[2])],
        [0, 0, 1, dt/L*np.tan(phi)],
        [0, 0, 0, 1]
    ])
    P_pred = A @ P @ A.T + Q
    return X_pred, P_pred

# EKF Initialization
X_ekf = x0
ekf_path = [X_ekf]

for i,z in enumerate(measured_path[1:]):
    X_ekf, P = ekf_predict(X_ekf, P, Q, dt, L, phis[i+1])
    X_ekf, P = kf_update(X_ekf, P, z, H, R)
    ekf_path.append(X_ekf)


###

# Plotting
actual_path = np.array(actual_path)
kf_path = np.array(kf_path)
ekf_path = np.array(ekf_path)
measured_path = np.array(measured_path)

plt.plot(actual_path[:, 0], actual_path[:, 1], label='Actual Path')
plt.plot(measured_path[:, 0], measured_path[:, 1], '.', label='Measurements')
plt.plot(kf_path[:, 0], kf_path[:, 1], '--', label='KF Estimated Path')
plt.plot(ekf_path[:, 0], ekf_path[:, 1], ':', label='EKF Estimated Path')
plt.legend()
plt.grid(True)
plt.show()


