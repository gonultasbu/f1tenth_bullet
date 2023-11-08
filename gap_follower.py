import numpy as np
import time
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
# import scipy.linalg as sp
import sys, os, fire, math
import pybullet
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.signal import place_poles
start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
eps = 1e-12


class CustomDriver:
    BUBBLE_RADIUS = 135     # default 160,150,140
    PREPROCESS_CONV_SIZE = 3    # default 3
    BEST_POINT_CONV_SIZE = 113   # default 80,120,117
    MAX_LIDAR_DIST = 3000000    # 3m
    
    ONE_DEGREE = np.pi / 180   # 1 degrees
    TWO_DEGREES = np.pi / 90    # 2 degrees
    FIVE_DEGREES = np.pi / 36   # 5 degrees
    TEN_DEGREES = np.pi / 18    # 10 degrees
    FIFTEEN_DEGREES = np.pi / 12    # 15 degrees
    TWENTY_DEGREES = np.pi / 9  # 20 degrees

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                    'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle
    
    def velocity_table(self, steering_angle, best_point_length):
        """ The speed is determined by two parameters.  
            The higher the best point length, the higher the speed.  
            The larger the steering angle value, the smaller the speed.
            The velocity table is a heuristic.
        Return : speed
        """
        input_angle = abs(steering_angle)
        
        # point의 거리가 25m 이상일 때
        if best_point_length >= 25:

            if input_angle < self.ONE_DEGREE:
                speed = 22
            elif input_angle >= self.ONE_DEGREE and input_angle < self.TWO_DEGREES:
                speed = 17
            elif input_angle >= self.TWO_DEGREES and input_angle < self.FIVE_DEGREES:
                speed = 13
            elif input_angle >= self.FIVE_DEGREES and input_angle < self.TEN_DEGREES:
                speed = 12
            else:
                speed = 8

        # point의 거리가 20m 이상일 때 25m 이내일 때
        elif best_point_length >= 20 and best_point_length < 25:

            if input_angle < self.ONE_DEGREE:
                speed = 17
            elif input_angle >= self.ONE_DEGREE and input_angle < self.TWO_DEGREES:
                speed = 15
            elif input_angle >= self.TWO_DEGREES and input_angle < self.FIVE_DEGREES:
                speed = 10
            elif input_angle >= self.FIVE_DEGREES and input_angle < self.TEN_DEGREES:
                speed = 9
            else:
                speed = 8
        
        # point의 거리가 15m 이상일 때 20m 이내일 때
        elif best_point_length >= 15 and best_point_length < 20:

            if input_angle < self.ONE_DEGREE:
                speed = 15
            elif input_angle >= self.ONE_DEGREE and input_angle < self.TWO_DEGREES:
                speed = 12
            elif input_angle >= self.TWO_DEGREES and input_angle < self.FIVE_DEGREES:
                speed = 10
            elif input_angle >= self.FIVE_DEGREES and input_angle < self.TEN_DEGREES:
                speed = 8
            elif input_angle >= self.TEN_DEGREES and input_angle < self.FIFTEEN_DEGREES:
                speed = 6
            elif input_angle >= self.FIFTEEN_DEGREES and input_angle < self.TWENTY_DEGREES:
                speed = 6
            else:
                speed = 6
        
        # point의 거리가 10m 이상일 때 15m 이내일 때
        elif best_point_length >= 10 and best_point_length < 15:

            if input_angle < self.ONE_DEGREE:
                speed = 14
            elif input_angle >= self.ONE_DEGREE and input_angle < self.TWO_DEGREES:
                speed = 10.5
            elif input_angle >= self.TWO_DEGREES and input_angle < self.FIVE_DEGREES:
                speed = 10
            elif input_angle >= self.FIVE_DEGREES and input_angle < self.TEN_DEGREES:
                speed = 9
            elif input_angle >= self.TEN_DEGREES and input_angle < self.FIFTEEN_DEGREES:
                speed = 7
            elif input_angle >= self.FIFTEEN_DEGREES and input_angle < self.TWENTY_DEGREES:
                speed = 7
            else:
                speed = 7
        
        # point의 거리가 5m 이상일 때 10m 이내일 때
        elif best_point_length >= 5 and best_point_length < 10:

            if input_angle < self.ONE_DEGREE:
                speed = 12
            elif input_angle >= self.ONE_DEGREE and input_angle < self.TWO_DEGREES:
                speed = 11
            elif input_angle >= self.TWO_DEGREES and input_angle < self.FIVE_DEGREES:
                speed = 10.5
            elif input_angle >= self.FIVE_DEGREES and input_angle < self.TEN_DEGREES:
                speed = 9
            elif input_angle >= self.TEN_DEGREES and input_angle < self.FIFTEEN_DEGREES:
                speed = 8.5
            elif input_angle >= self.FIFTEEN_DEGREES and input_angle < self.TWENTY_DEGREES:
                speed = 7.5
            else:
                speed = 7.5
        
        # 5m 이내일 때
        else:

            if input_angle < self.ONE_DEGREE:
                speed = 10
            elif input_angle >= self.ONE_DEGREE and input_angle < self.TWO_DEGREES:
                speed = 9
            elif input_angle >= self.TWO_DEGREES and input_angle < self.FIVE_DEGREES:
                speed = 8
            elif input_angle >= self.FIVE_DEGREES and input_angle < self.TEN_DEGREES:
                speed = 7
            elif input_angle >= self.TEN_DEGREES and input_angle < self.FIFTEEN_DEGREES:
                speed = 6
            elif input_angle >= self.FIFTEEN_DEGREES and input_angle < self.TWENTY_DEGREES:
                speed = 6
            else:
                speed = 6
    
        return speed

    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR 가장 가까운 Lidar 위치
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)
        
        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        speed = self.velocity_table(steering_angle, proc_ranges[best])
        # print('1. best point lengh : ', proc_ranges[best])
        # print('2. speed : ',speed)
        # print('3. Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle



class GapFollower:
    """
    Credits: https://github.com/Taeyoung96/f1tenth-my-own-driver/blob/master/pkg/src/pkg/drivers.py
    """
    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_SPEED = 8.0 # 8.0
    CORNERS_SPEED = 5.0 # 5.0
    STRAIGHTS_STEERING_ANGLE = 0.0  

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        """ 
		Preprocess the LiDAR scan array. Expert implementation includes:    
		1.Setting each value to the mean over some window
        2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges)
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ 
		Return the start index & end index of the max gap in free_space_ranges
        free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """
		Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ 
        Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        """ 
        Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)
        
        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle

if __name__ == "__main__":
	quit()