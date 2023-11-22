import numpy as np
import matplotlib
# import waypoint_creation
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
# import scipy.linalg as sp
import sys, os
import pybullet as p
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from gap_follower import GapFollower, CustomDriver
import utils 
import pandas as pd
import csv
from pynput import keyboard

start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
eps = 1e-12

# for mpc
from cvxpy_mpc.utils import compute_path_from_wp, get_ref_trajectory
from cvxpy_mpc import MPC



class BulletSim():
	def __init__(self):
		self.i = 0
		self.Vx = 2.0
		self.sim_ts = 0.002 # 0.002
		self.controller_freq = 25
		self.sim_num_ts = 101000 # Single lap time 
		self.text_id = None
		self.text_id_2 = None
		self.cam_fov = 60
		self.cam_aspect = 640 / 480
		self.cam_near = 0.01
		self.cam_far = 1000
		self.cam_view_matrix = p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
		self.cam_projection_matrix = p.computeProjectionMatrixFOV(self.cam_fov, self.cam_aspect, self.cam_near, self.cam_far)
		self.distance = 100000
		self.gf = GapFollower()
		self.customDriver = CustomDriver()
		self.firstLapCompleted = False
		self.positions = []
		self.totalNumOfLaps = 1
		self.lap_count = 0
		self.initial_position = [10, 0.204, 0.0]
		self.inInterval = True

		self.gyro_data = []
		self.accel_data = []
		self.sensor_freq = 200 # Hz
		self.vel_data = []
		self.steering_angle_data = []
		self.prev_vel  = [0,0,0]  #Initial Linear Velocities of the Vehicle
		return 
		
	def run_sim(self):
		# Initialize the p GUI and the Visualizer
		p.connect(p.GUI)
		p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,0)
		p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
		p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)

		p.resetSimulation()
		# Load the track sdf
		ground_tuple = p.loadSDF(os.path.join(
					os.path.dirname(__file__), "./tracks/barcelona/barcelona.sdf")) # ,globalScaling=0.75)

		new_orientation = p.getQuaternionFromEuler((-np.pi/2, 0, 0))
		for gt_elem in ground_tuple:
			p.resetBasePositionAndOrientation(gt_elem, (0, 0, 0), new_orientation)

		init_heading_euler = R.from_euler("YZX",[180.0,0.0,-90.0] , degrees=True)
		# init_heading_quat  = init_heading_euler.as_quat()
		
		# Agent Load 
		agent = p.loadURDF(os.path.join(
					os.path.dirname(__file__), "./urdf/uva-f1tenth/uva-f1tenth-model.urdf"),self.initial_position,init_heading_euler.as_quat(), globalScaling=1.25)

		# base_p_orient = p.getBasePositionAndOrientation(agent) 

		p.setGravity(0, -9.81, 0)
		p.setTimeStep(self.sim_ts)
		p.setRealTimeSimulation(0)

		fic = 0.4
		fic_g = 0.9
		for gt_elem in ground_tuple:
			# Make lateralFriction f(t) 
			p.changeDynamics(gt_elem,0,lateralFriction=fic_g,spinningFriction=0.0,rollingFriction=0.0,restitution=0)

		p.changeDynamics(agent,2,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		p.changeDynamics(agent,3,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		p.changeDynamics(agent,5,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		p.changeDynamics(agent,7,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)

		i = 0
		driving = False
		max_velocity = 20
		velocity = 0
		steering_angle = 0
		max_steering_angle = 0.5  # Maximum steering angle
		steering_increment = 0.02
		while(True):
		# Condition on the lap count
			keys = p.getKeyboardEvents()
			p.stepSimulation()
			# time.sleep(0.000001)

			# Start and Stop Condition
			# Start driving with 'D' key
			if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
				driving = True
			# Stop the loop with 'S' key
			if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
				break  # Exit the while loop
			if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
				p.resetBasePositionAndOrientation(agent, self.initial_position, init_heading_euler.as_quat())
				velocity = 0
				steering_angle = 0
			if driving:
				# Control logic based on key presses
				if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
					velocity += 0.01
				elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
					velocity -= 0.01
				else:
					velocity = 0
				if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
					steering_angle += steering_increment
				elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
					steering_angle -= steering_increment
				else:
					# Gradually return to neutral position
					if steering_angle > 0:
						steering_angle = max(0, steering_angle - steering_increment)
					elif steering_angle < 0:
						steering_angle = min(0, steering_angle + steering_increment)

				velocity = max(-max_velocity, min(velocity, max_velocity))
				steering_angle = max(-max_steering_angle, min(steering_angle, max_steering_angle))

				# cg_pos_3d_xyz = np.array(p.getLinkState(agent,8)[0])
				agent_pos, agent_orn = p.getBasePositionAndOrientation(agent)
				self.positions.append(agent_pos)

				cg_pos_3d_xyz = np.array(agent_pos)
				self.veh_2d_world_pos = cg_pos_3d_xyz[[0,2]]
				# self.count_laps()
			
				# cg_heading_3d_quat = R.from_quat(np.array(p.getLinkState(agent,8)[1]))
				cg_heading_3d_quat = R.from_quat(np.array(agent_orn))
				self.veh_2d_world_heading  = -cg_heading_3d_quat.as_euler('YZX', degrees=False)[0]

				# Setup camera positioning.
				xA, yA, zA = agent_pos
				# xA = xA - 0.2
				yA = yA + 0.4
				xB = xA + np.cos(self.veh_2d_world_heading) * self.distance
				zB = zA + np.sin(self.veh_2d_world_heading) * self.distance
				yB = yA

				view_matrix = p.computeViewMatrix(
									cameraEyePosition=[xA, yA, zA],
									cameraTargetPosition=[xB, yB, zB],
									cameraUpVector=[0.0, 1.0, 0.0]
								)

				projection_matrix = p.computeProjectionMatrixFOV(
							fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)
				if not (i%((1/self.sim_ts))):
					print("Current Velocity Command:", velocity, "Steering Angle:", steering_angle)
					print("Current Velocity Reading", self.prev_vel)

				if not (i%self.controller_freq):
					imgs = p.getCameraImage(640, 480,
								view_matrix,
								projection_matrix, shadow=True,
								renderer=p.ER_BULLET_HARDWARE_OPENGL)
				

					# self.target_steering_angle = np.deg2rad(self.linear_Cntrl(self.K,e_state))
					self.target_steering_angle = steering_angle
					# velocity = 1.5*velocity
					# velocity = 18.0
						
					p.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=4,
						controlMode=p.POSITION_CONTROL,
						targetPosition = self.target_steering_angle)
					p.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=6,
						controlMode=p.POSITION_CONTROL,
						targetPosition = self.target_steering_angle)
					
					self.steering_angle_data.append(self.target_steering_angle)

				if not (i%int((1/self.controller_freq) * (1/self.sim_ts))):
					# Controlling the four wheel of the car  
					p.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=2,
						controlMode=p.VELOCITY_CONTROL,
						targetVelocity = velocity)
					p.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=3,
						controlMode=p.VELOCITY_CONTROL,
						targetVelocity = velocity)
					
					p.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=5,
						controlMode=p.VELOCITY_CONTROL,
						targetVelocity = velocity)
					p.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=7,
						controlMode=p.VELOCITY_CONTROL,
						targetVelocity = velocity)
					# Collect the gyro and accel val	
					self.vel_data.append(velocity)
					
				# input
				if not (i%np.ceil((1/self.sensor_freq)*(1/self.sim_ts))):
					gyro_reading = self.gyro(agent)
					self.gyro_data.append(gyro_reading)

					accel_reading = self.accelerometer(agent, self.prev_vel, self.sim_ts)
					self.accel_data.append(accel_reading)
					self.prev_vel = p.getBaseVelocity(agent)[0]
				# Check for the whether we finished a lap or not
				i += 1

		self.gyro_data = np.array(self.gyro_data)
		self.accel_data = np.array(self.accel_data)

		return 
	
	def get_true_depth_values(self, input):
		return self.cam_far * self.cam_near / (self.cam_far - (self.cam_far - self.cam_near)*input)


	

	def plot_positions(self):
		# Extract X and Y coordinates
		x_coords = [pos[0] for pos in self.positions]
		y_coords = [pos[2] for pos in self.positions]
		plt.figure("traj")
		ax = plt.gca()
		ax.set_aspect('equal', adjustable='box')
		plt.plot(x_coords, y_coords, label='Car trajectory')

		# cmap = plt.cm.viridis
		# norm = mcolors.Normalize(vmin=0, vmax=len(x_coords))
		# for i in range(len(x_coords) - 1):
		# 	ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=cmap(norm(i)), linewidth=2)

		plt.xlabel('X position')
		plt.ylabel('Y position')
		plt.title('Car Position Over Time')
		plt.legend()
		plt.grid(True)
		plt.axis("equal")

		# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		# sm.set_array([])
		# plt.colorbar(sm, ticks=np.linspace(0, len(x_coords), num=5))  # Adjust number of ticks if needed


		plt.savefig('./data/trajectories/car_trajectory_open_world.pdf')  # Save the plot as a PDF
		plt.close()

	
		# Save positions as csv file
		filename = './data/friction/friction_open_world.csv'
		with open(filename, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(self.positions)

		print(f"Data has been written to {filename}")
		return
	
	def change_friction(self,ground_tuple,agent):
		# Define a set of different friction values
		# friction_val_list = [0s.1, 0.44, 0.92]
		friction_val_list = [0.92]
		friction_val = np.random.choice(friction_val_list)

		# for gt_elem in ground_tuple:
		# 	# Make lateralFriction f(t) 
		# 	p.changeDynamics(gt_elem,0,lateralFriction=friction_val,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
			
		p.changeDynamics(agent,2,lateralFriction=friction_val,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		p.changeDynamics(agent,3,lateralFriction=friction_val,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		p.changeDynamics(agent,5,lateralFriction=friction_val,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		p.changeDynamics(agent,7,lateralFriction=friction_val,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		
		return friction_val
	
	# Rotational Velocity roll,pitch,yaw
	def gyro(self, agent, noise_std=0.01):
		_, angular_velocity = p.getBaseVelocity(agent)
		noise = np.random.normal(0, noise_std, len(angular_velocity))
		angular_velocity_with_noise = angular_velocity + noise
		return angular_velocity_with_noise
	
	# Linear acceleration for x,y,z
	def accelerometer(self, agent, pre_vel, time_step, noise_std=0.01):
		linear_velocity, _ = p.getBaseVelocity(agent)
		acceleration = [(v - pv) / time_step for v, pv in zip(linear_velocity, pre_vel)]
		noise = np.random.normal(0, noise_std, len(acceleration))
		acceleration_with_noise = acceleration + noise
		return acceleration_with_noise


def plot_gyro_accel(gyro_data, accel_data):
	# Extract X and Y coordinates
	plt.figure(figsize=(10, 8))
	titles = ['Gyroscope X-axis', 'Gyroscope Y-axis', 'Gyroscope Z-axis']
	for i in range(3):
		plt.subplot(3, 1, i+1)
		plt.plot(gyro_data[:, i])
		plt.title(titles[i])
		plt.xlabel('Time Step')
		plt.ylabel('Angular Velocity (rad/s)')
		plt.grid()

	plt.tight_layout()
	
	plt.savefig('./data/sensors/gyro_plots_open_world.pdf')  # Save the plot as a PDF
	plt.close()


	# Save positions as csv file
	filename = './data/sensors/gyro_values_open_world.csv'
	with open(filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(gyro_data)

	print(f"Gyro Data has been written to {filename}")

	plt.figure(figsize=(10, 8))
	titles = ['Accelerometer X-axis', 'Accelerometer Y-axis', 'Accelerometer Z-axis']
	for i in range(3):
		plt.subplot(3, 1, i+1)
		plt.plot(accel_data[:, i])
		plt.title(titles[i])
		plt.xlabel('Time Step')
		plt.ylabel('Acceleration (m/s^2)')
		plt.grid()

	plt.tight_layout()
	plt.savefig('./data/sensors/accel_plots_open_world.pdf')  # Save the plot as a PDF
	plt.close()

	#Save positions as csv file
	filename = './data/sensors/accel_values_flat.csv'
	with open(filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(accel_data)

	print(f"Accel Data has been written to {filename}")

	


if __name__ == "__main__":
	bs = BulletSim()
	bs.run_sim()
	# print(bs.lap_count)
	plot_gyro_accel(bs.gyro_data, bs.accel_data)
	bs.plot_positions()
	quit()