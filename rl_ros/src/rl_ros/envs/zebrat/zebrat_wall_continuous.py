#!/usr/bin/env python3
"""
Author: Zhaofeng Tian
Email: shoguntian@gmail.com

"""
import rospy
import rospkg
import numpy as np
from gym import spaces
from rl_ros.envs.zebrat import zebrat_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from rl_ros.registration import LoadYamlFileParamsTest
from rl_ros.registration import ROSLauncher
import os
import math

class ZebratWallContinuousEnv(zebrat_env.ZebratEnv):
    def __init__(self):
        """
        This Task Env is designed for having the zebrat in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = "/home/tian/simulation_ws"


        ROSLauncher(rospackage_name="zebrat",
                    launch_file_name="load_env.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="rl_ros",
                               rel_path_from_package_to_file="src/rl_ros/config",
                               yaml_file_name="zebrat_wall.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ZebratWallContinuousEnv, self).__init__(ros_ws_abspath)
        # ***************************************************************************
        # ***************************************************************************
        # ***************************************************************************
        """
        ******************* ACTION SPACE *****************
        | Num | Action | Min  | Max |
        |-----|--------|------|-----|
        | 0   | Linear | -0.6 | 0.6 |
        | 1   | Rotate | -0.6 | 0.6 |


        """
        a_high = np.array([0.6, 0.6])
        a_low = np.array([-0.6, -0.6])
        self.action_space = spaces.Box(a_low.astype(np.float32), a_high.astype(np.float32))
        """
        ******************* OBSERVATION SPACE *****************
        300 * 300 Gray image

        """
        self.observation_space = spaces.Box(low=0, high=6, shape=(32,), dtype=np.float32)




        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        self.index = [0, 6, 12, 19, 22, 24, 28, 32, 38, 46, 58, 72, 90, 109, 118, 136, 180, 224, 242, 252, 270, 288, 302, 314, 322, 328, 332, 336, 338, 342, 348, 354]
        self.collision_ranges = [0.9, 0.905, 0.92, 0.906, 0.817, 0.73, 0.644, 0.562, 0.484, 0.414, 0.355, 0.315, 0.3, 0.302, 0.215, 0.138, 0.1, 0.138, 0.215, 0.315, 0.3, 0.315, 0.355, 0.414, 0.484, 0.562, 0.644, 0.73, 0.817, 0.944, 0.92, 0.905]
        #self.collision_ranges = [1.0, 1.086, 0.566, 0.4, 0.449, 0.2, 0.449, 0.4, 0.566, 1.086]
        #self.offset = [0.1, 0.16 , 0.05, 0.02, 0.085, 0.1, 0.085, 0.02, 0.05, 0.16, 0.1]
        #self.offset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #self.offset = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]
        self.offset = [-0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02]
        #self.offset = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        #number_observations = rospy.get_param('/zebrat/n_observations')
        #self.offset = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/zebrat/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/zebrat/linear_turn_speed')
        self.angular_speed = rospy.get_param('/zebrat/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/zebrat/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/zebrat/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/zebrat/new_ranges')
        self.min_range = rospy.get_param('/zebrat/min_range')
        self.max_laser_value = rospy.get_param('/zebrat/max_laser_value')
        self.min_laser_value = rospy.get_param('/zebrat/min_laser_value')

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/zebrat/desired_pose/x")
        self.desired_point.y = rospy.get_param("/zebrat/desired_pose/y")
        self.desired_point.z = rospy.get_param("/zebrat/desired_pose/z")
        
        # Get GridMap size
        self.gridmap_size_x = rospy.get_param("/zebrat/gridmap_size_x")
        self.gridmap_size_y = rospy.get_param("/zebrat/gridmap_size_y")

        self.center_x_index = math.ceil(self.gridmap_size_x/2)
        self.center_y_index = math.ceil(self.gridmap_size_y/2)
        self.gridmap_resolution = rospy.get_param("/zebrat/gridmap_resolution")
        self.gridmap = np.zeros((self.gridmap_size_x, self.gridmap_size_y))


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        rospy.logdebug("laser_scan len===>" + str(len(laser_scan.ranges)))

        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        #high = np.full((num_laser_readings), self.max_laser_value)
        #low = np.full((num_laser_readings), self.min_laser_value)

        # We only use two integers
        #self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/zebrat/forwards_reward")
        self.turn_reward = rospy.get_param("/zebrat/turn_reward")
        self.end_episode_points = rospy.get_param("/zebrat/end_episode_points")

        self.cumulated_steps = 0.0

        self.discretized_laser_scan= []

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the zebrat
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        # We tell zebrat the linear and angular speed to set to execute
        #self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        print (action[0])
        print (action[1])
        #print(action)
        self.move_base(action[0], action[1], epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        
        """
        #Here we define what sensor data defines our robots observations
        #To know which Variables we have acces to, we need to read the
        #zebratEnv API DOCS
        #:return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()                                                                                                    
        

        # We get the odometry so that SumitXL knows where it is.
        '''odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y'''

        # We round to only two decimals to avoid very big Observation space
        #odometry_array = [round(x_position, 1),round(y_position, 1)]

        # We only want the X and Y position and the Yaw
        #observations = discretized_laser_scan + odometry_array

        self.discretized_laser_scan = self.discretize_observation(laser_scan)
        observations = self.discretized_laser_scan
        '''laser_ranges = [] 
        raw_ranges = laser_scan.ranges
        for i, item in enumerate(raw_ranges):
            if (i%1==0):
                if i < 360:
                    if item == float ('Inf') or np.isinf(item):
                        laser_ranges.append(self.max_laser_value)
                    elif np.isnan(item):
                        laser_ranges.append(self.min_laser_value)
                    else:
                        laser_ranges.append(round(item,3))'''

                #if (self.collision_ranges[i] > item > 0):
                    #rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.collision_ranges[i]))
                    #self._episode_done = True'''


        #observations = laser_ranges
        rospy.logdebug("Observations==>"+str(observations) + '   The length is: '+ str(len(observations)))
        rospy.logdebug("END Get Observation ==>")


        return observations
        
        """
        map = self._get_gridmap()
        return map
        """

    def discretize_observation(self,data):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        #discretized_data = [data.ranges[0],data.ranges[23],data.ranges[45],data.ranges[90], data.ranges[117],data.ranges[180],data.ranges[243],data.ranges[270], data.ranges[315], data.ranges[337]]
        discretized_data= []
        for i in range(len(self.collision_ranges)):
            discretized_data.append(data.ranges[self.index[i]])



        discretized_ranges = []

        rospy.logdebug("data=" + str(data))
        # rospy.logwarn("new_ranges=" + str(new_ranges))
        # rospy.logwarn("mod=" + str(mod))

        for i, item in enumerate(discretized_data):
            #if (i%mod==0):
            if item == float ('Inf') or np.isinf(item):
                discretized_ranges.append(self.max_laser_value)
            elif np.isnan(item):
                discretized_ranges.append(self.min_laser_value)
            else:
                discretized_ranges.append(round(item,3))

            if (self.collision_ranges[i]-self.offset[i] > item > 0):
                rospy.logerr("done Validation >>> item=" + str(i)+': '+ str(item)+" < "+str(self.collision_ranges[i]-self.offset[i]))
                self._episode_done = True
            #else:
                #rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.collision_ranges[i]))


        return discretized_ranges

    def _get_gridmap(self):
        """
        Get the occupancy grid map of nearby enviroment
        """
        gridmap = np.zeros((self.gridmap_size_x, self.gridmap_size_y))
        for i in range(self.gridmap_size_x):
            for j in range(self.gridmap_size_y):
                if (i>=self.center_x_index-1/self.gridmap_resolution) and (i<= self.center_x_index+0.2/self.gridmap_resolution):
                    if (j>=self.center_y_index - 0.4/self.gridmap_resolution) and (j<=self.center_y_index + 0.4/self.gridmap_resolution):
                    #if (j>=60) and (j<=140):
                        gridmap[i][j] = 0.5
        
        scan = self.get_laser_scan()

        for i, item in enumerate(scan.ranges):

            if item == float ('Inf') or np.isinf(item):
                x_index = int(self.center_x_index - (self.max_laser_value*math.cos(i*math.pi/180))/self.gridmap_resolution)
                y_index = int(self.center_y_index + (self.max_laser_value*math.sin(i*math.pi/180))/self.gridmap_resolution)
            elif np.isnan(item):
                x_index = int(self.center_x_index - (self.min_laser_value*math.cos(i*math.pi/180))/self.gridmap_resolution)
                y_index = int(self.center_y_index + (self.min_laser_value*math.sin(i*math.pi/180))/self.gridmap_resolution) 
            else:
                x_index = int(self.center_x_index - (scan.ranges[i] * math.cos(i*math.pi/180))/self.gridmap_resolution)
                y_index = int(self.center_y_index + (scan.ranges[i] * math.sin(i*math.pi/180))/self.gridmap_resolution) 
            if (x_index >= self.gridmap_size_x or x_index <= 0) or (y_index >= self.gridmap_size_y or y_index <= 0):
                continue
            gridmap[x_index][y_index] = 1
        gridmap = gridmap.reshape([-1,300,300,1])
        # print(gridmap.shape)
        
        return gridmap


        
            




    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("zebrat is Too Close to wall==>")
        else:
            rospy.logerr("zebrat didnt crash at least ==>")


            '''current_position = Point()
            current_position.x = observations[-2]
            current_position.y = observations[-1]
            current_position.z = 0.0'''

            current_position = Point()
            odometry = self.get_odom()
            x_position = odometry.pose.pose.position.x
            y_position = odometry.pose.pose.position.y
            current_position.x = x_position
            current_position.y = y_position
            current_position.z = 0.0

            """
            MAX_X = 3.5
            MIN_X = -3.5
            MAX_Y = 3.5
            MIN_Y = -3.5"""

            MAX_X = 50
            MIN_X = -50
            MAX_Y = 50
            MIN_Y = -50

            # We see if we are outside the Learning Space

            if current_position.x <= MAX_X and current_position.x > MIN_X:
                if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                    rospy.logdebug("Zebrat Position is OK ==>["+str(current_position.x)+","+str(current_position.y)+"]")

                    # We see if it got to the desired point
                    #if self.is_in_desired_position(current_position):
                    #obs = self.discretized_laser_scan
                    obs = self.discretize_observation(self.get_laser_scan())
                    #if obs[3]+ obs[4] + obs[5] + obs[6]+ obs[7] >= 5 * self.max_laser_value:
                    if obs[20] + obs[12]  >= 12:
                    #     270   243  180  117  90
                        self._episode_done = True


                else:
                    rospy.logerr("Zebrat to Far in Y Pos ==>"+str(current_position.x))
                    self._episode_done = True
            else:
                rospy.logerr("Zebrat to Far in X Pos ==>"+str(current_position.x))
                self._episode_done = True




        return self._episode_done
    
    def _compute_reward(self, observations, action, done):


        observations = self.discretize_observation(self.get_laser_scan())       
        gap_ranges = []
        crash = False
        reward = 0
        reward1 = 0
        reward2 = 0
        reward3 = 0
        reward4 = 0
        self.crash = 'Not Crash'
        if not done:
            
            for i in range(len(observations)):
            
                gap_ranges.append(observations[i] - self.collision_ranges[i]+ self.offset[i])
                #gap_ranges.append(observations[i] - self.collision_ranges[i])
            for i in range(len(gap_ranges)):   
                if gap_ranges[i] <= 0:
                    reward = -50
                    crash = True
                    self.crash = 'Crash'
                    rospy.logerr

            if crash != True:
                
                # **************************** Part I  Distance ************************************
                gap_ranges.sort()
                decay1 = 0.9
                for i in range(12):
                    reward1 += decay1 * math.log10(gap_ranges[i])
                    decay1 = decay1*decay1
                rospy.logerr("The reward 1 is:  " + str(reward1))
                #reward = math.log10(gap_ranges[0])+ 0.8 * math.log10(gap_ranges[1])+ 0.5 * math.log10(gap_ranges[2])
                #reward = math.log10(0.01*gap_ranges[0])

                # **************************** Part II  Forward ************************************

                decay2 = 0.9
                
                #reward += math.log10(observations[5])+ 0.8*math.log10(observations[4]) + 0.8* math.log10(observations[6])
                reward2 += action[0]*(decay2 *observations[16] + math.pow(decay2,2) * (observations[17] + observations[15])+ math.pow(decay2,3)* (observations[18]+ observations[14])+ math.pow(decay2,4)* (observations[19]+ observations[13])+ math.pow(decay2,5)* (observations[20]+ observations[12])   )                
                reward2 = round(reward2, 3)
                rospy.logerr("The reward 2 is:  " + str(reward2))

                # **************************** Part III  Center Keeping ************************************
                decay3 = 0.9
                reward3 -= abs(observations[20] - observations[12]) + decay3*(abs(observations[19] - observations[13])+abs(observations[21] - observations[11])) + math.pow(decay3,2)*(abs(observations[18] - observations[13])+abs(observations[22] - observations[10]))
                rospy.logerr("The reward 3 is:  " + str(reward3))
                # **************************** Part IV  Time is valuable ************************************
                reward4 -= 1
                rospy.logerr("The reward 4 is:  " + str(reward4))

                reward = reward1 + reward2 + reward3 + reward4
                rospy.logerr("The reward total for this step is: " + str(reward))
        # NOt DONE
        else:

            #if self.is_in_desired_position(current_position):
                #reward = self.end_episode_points
                #reward += self.end_episode_points
            #if observations[20] + observations[12]  >= 2 * self.max_laser_value-6:
            if observations[20] + observations[12]  >= 12.0:
                reward += self.end_episode_points
            else:
                reward = -1*self.end_episode_points
                self.crash = 'crash'
                


        #self.previous_distance_from_des_point = distance_from_des_point



        return reward



    # Internal TaskEnv Methods




    def is_in_desired_position(self,current_position, epsilon=0.35):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False


        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos


    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

