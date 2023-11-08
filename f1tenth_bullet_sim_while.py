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
import pybullet
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from gap_follower import GapFollower, CustomDriver
import utils 
import pandas as pd
import csv
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
		self.sim_ts = 0.01 # 0.002
		self.controller_freq = 25
		self.sim_num_ts = 101000 # Single lap time 
		self.text_id = None
		self.text_id_2 = None
		self.cam_fov = 60
		self.cam_aspect = 640 / 480
		self.cam_near = 0.01
		self.cam_far = 1000
		self.cam_view_matrix = pybullet.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
		self.cam_projection_matrix = pybullet.computeProjectionMatrixFOV(self.cam_fov, self.cam_aspect, self.cam_near, self.cam_far)
		self.distance = 100000
		self.gf = GapFollower()
		self.customDriver = CustomDriver()
		self.firstLapCompleted = False
		self.positions = []
		self.totalNumOfLaps = 1
		self.lap_count = 0
		self.initial_position = [0.0, 0.204, 0.0]
		self.inInterval = True
		return 
		
	def run_sim(self):
		# Initialize the pybullet GUI and the Visualizer
		pybullet.connect(pybullet.GUI)
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME,0)
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,1)
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_Y_AXIS_UP,1)

		pybullet.resetSimulation()
		# Load the track sdf
		ground_tuple = pybullet.loadSDF(os.path.join(
					os.path.dirname(__file__), "./tracks/barcelona/barcelona.sdf"))

		new_orientation = pybullet.getQuaternionFromEuler((-np.pi/2, 0, 0))
		for gt_elem in ground_tuple:
			pybullet.resetBasePositionAndOrientation(gt_elem, (0, 0, 0), new_orientation)

		init_heading_euler = R.from_euler("YZX",[0.0,0.0,-90.0] , degrees=True)
		# init_heading_quat  = init_heading_euler.as_quat()
		
		# Agent Load 
		agent = pybullet.loadURDF(os.path.join(
					os.path.dirname(__file__), "./urdf/uva-f1tenth/uva-f1tenth-model.urdf"),self.initial_position,init_heading_euler.as_quat())

		# base_p_orient = pybullet.getBasePositionAndOrientation(agent) 

		pybullet.setGravity(0, -9.81, 0)
		pybullet.setTimeStep(self.sim_ts)
		pybullet.setRealTimeSimulation(0)

		 
		for gt_elem in ground_tuple:
			# Make lateralFriction f(t) 
			pybullet.changeDynamics(gt_elem,0,lateralFriction=0.92,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		fic = 0.2
		pybullet.changeDynamics(agent,2,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,3,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,5,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,7,lateralFriction=fic,spinningFriction=0.0,rollingFriction=0.0,restitution=0)

		i = 0
		while(self.lap_count != self.totalNumOfLaps):
		# Condition on the lap count
			if (self.firstLapCompleted == False):			
				pybullet.stepSimulation()
				# time.sleep(0.000001)

				# cg_pos_3d_xyz = np.array(pybullet.getLinkState(agent,8)[0])
				agent_pos, agent_orn = pybullet.getBasePositionAndOrientation(agent)
				self.positions.append(agent_pos)
				# if (i % 10000) == 0:
				# 	friction_val = self.change_friction(ground_tuple)
				# 	print("friction value changed to: ", friction_val)	
				cg_pos_3d_xyz = np.array(agent_pos)
				self.veh_2d_world_pos = cg_pos_3d_xyz[[0,2]]
				self.count_laps()
				
				# cg_heading_3d_quat = R.from_quat(np.array(pybullet.getLinkState(agent,8)[1]))
				cg_heading_3d_quat = R.from_quat(np.array(agent_orn))
				self.veh_2d_world_heading  = -cg_heading_3d_quat.as_euler('YZX', degrees=False)[0]

				# Setup camera positioning.
				xA, yA, zA = agent_pos
				yA = yA + 0.3
				xB = xA + np.cos(self.veh_2d_world_heading) * self.distance
				zB = zA + np.sin(self.veh_2d_world_heading) * self.distance
				yB = yA

				view_matrix = pybullet.computeViewMatrix(
									cameraEyePosition=[xA, yA, zA],
									cameraTargetPosition=[xB, yB, zB],
									cameraUpVector=[0.0, 1.0, 0.0]
								)

				projection_matrix = pybullet.computeProjectionMatrixFOV(
							fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

				if not (i%self.controller_freq):
					imgs = pybullet.getCameraImage(640, 480,
								view_matrix,
								projection_matrix, shadow=True,
								renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
					
					true_scan_depths = self.get_true_depth_values(imgs[3][240,:])
					# Creating the controlled imput
					# _, steering_angle = self.gf.process_lidar(true_scan_depths)
					velocity, steering_angle = self.customDriver.process_lidar(true_scan_depths)


					# self.target_steering_angle = np.deg2rad(self.linear_Cntrl(self.K,e_state))
					self.target_steering_angle = -steering_angle
					# velocity = 1.5*velocity
					if not (i%int((1/self.controller_freq) * (1/self.sim_ts))):
						# Controlling the four wheel of the car  
						pybullet.setJointMotorControl2(bodyUniqueId=agent,
							jointIndex=2,
							controlMode=pybullet.VELOCITY_CONTROL,
							targetVelocity = velocity)
						pybullet.setJointMotorControl2(bodyUniqueId=agent,
							jointIndex=3,
							controlMode=pybullet.VELOCITY_CONTROL,
							targetVelocity = velocity)
						
						pybullet.setJointMotorControl2(bodyUniqueId=agent,
							jointIndex=5,
							controlMode=pybullet.VELOCITY_CONTROL,
							targetVelocity = velocity)
						pybullet.setJointMotorControl2(bodyUniqueId=agent,
							jointIndex=7,
							controlMode=pybullet.VELOCITY_CONTROL,
							targetVelocity = velocity)	
					
					pybullet.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=4,
						controlMode=pybullet.POSITION_CONTROL,
						targetPosition = self.target_steering_angle)
					pybullet.setJointMotorControl2(bodyUniqueId=agent,
						jointIndex=6,
						controlMode=pybullet.POSITION_CONTROL,
						targetPosition = self.target_steering_angle)
					
			# Check for the whether we finished a lap or not
			i += 1
		return 
	
	def get_true_depth_values(self, input):
		return self.cam_far * self.cam_near / (self.cam_far - (self.cam_far - self.cam_near)*input)


	def count_laps(self):
		# Create a treshold for interval when we enter it
		eps_x = 0.1
		eps_z = 1.5
		prev_inInterval = self.inInterval
		x_interval = [self.initial_position[0]-eps_x,self.initial_position[0]+eps_x]
		z_interval = [self.initial_position[2]-eps_z,self.initial_position[2]+eps_z]
		if (self.veh_2d_world_pos[0] >= x_interval[0]) and (self.veh_2d_world_pos[0] <= x_interval[1]):
			if (self.veh_2d_world_pos[1] >= z_interval[0]) and (self.veh_2d_world_pos[1] <= z_interval[1]):
				self.inInterval = True
			else:
				self.inInterval = False
		else:
			self.inInterval = False

		if (prev_inInterval != self.inInterval) and self.inInterval:
			self.lap_count += 1
			print("Car took a lap around the track")
		return  
	

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


		plt.savefig('./data/trajectories/car_trajectory_02_2.pdf')  # Save the plot as a PDF
		plt.close()
		
		# Save positions as csv file
		filename = './data/friction/ftg_friction_02_2.csv'
		with open(filename, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(self.positions)

		print(f"Data has been written to {filename}")
		return
	

	def change_friction(self,ground_tuple):
		# Define a set of different friction values
		friction_val_list = [0.05, 0.2, 0.44, 0.67, 0.81, 0.92]
		friction_val = np.random.choice(friction_val_list)
		for gt_elem in ground_tuple:
			# Make lateralFriction f(t) 
			pybullet.changeDynamics(gt_elem,0,lateralFriction=friction_val,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		return friction_val

if __name__ == "__main__":
	bs = BulletSim()
	bs.run_sim()
	print(bs.lap_count)
	bs.plot_positions()
	quit()