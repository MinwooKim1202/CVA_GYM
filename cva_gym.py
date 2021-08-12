#!/usr/bin/env python2

import os
import threading
import time
import sys
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from std_srvs.srv import Empty

lock = threading.Lock()

state = np.zeros(360)
collision_flag = None
play_state_flag = False
clear_flag = False
reward = 0
robot_x = 0
robot_y = 0
before_robot_y = -8.0
step_i = 0

cmd_vel = Twist()
cmd_vel.linear.x = 0.0
cmd_vel.linear.y = 0.0
cmd_vel.linear.z = 0.0
cmd_vel.angular.x = 0.0
cmd_vel.angular.y = 0.0
cmd_vel.angular.z = 0.0

def make(env_name):
    env = Env(env_name)
    return env

class Env:
    def __init__(self, env_name):
        self.env_name = env_name
        
        try:
            #os.system("gnome-terminal -- roslaunch cva_gym simple_circuit.launch")
            rospy.init_node('cva_gym')
            
            t = Worker()
            t.daemon = True
            t.start()

            print("Environment Init Success!!")
        except Exception as e:
            print(e)
        
    def reset(self):
        global state
        state = np.array(state)
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:            
            sim_reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            sim_reset()
            global collision_flag 
            collision_flag = False

            #print("Reset Success!!")
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            sim_pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            sim_pause()
            play_state_flag = False

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        state = np.array(state)
        #print(type(state))
        return state
        

    def step(self, action):
        global play_state_flag
        global clear_flag
        global collision_flag
        global state
        global cmd_vel
        global step_i
        global before_robot_y
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        
        reward = 0
        
        if robot_y > before_robot_y:
            reward = 0.1
            before_robot_y = robot_y
            #print(before_robot_y)

    
        if play_state_flag == True:
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                sim_pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                sim_pause()
                #print("pause")
                play_state_flag = False

            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        else:
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                sim_unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                sim_unpause()
                #print("unpause")
                play_state_flag = True

            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        if  robot_y > 7.15:
            print("Clear!!!")
            clear_flag = True
            reward = 100
            rospy.wait_for_service('/gazebo/reset_simulation')
            try:            
                sim_reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                sim_reset()
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        time.sleep(0.01)

        if collision_flag == True and clear_flag == False:
            #reward = reward - (step_i / 1000) 
            reward = -10
            before_robot_y = -8.0
            step_i = 0
        
        state = np.array(state)
        step_i = step_i + 1
        #state = state.flatten()
        return state, reward, collision_flag

    def close(self):
        print("close")

class Worker(threading.Thread):
    def __init__(self):
        super(Worker, self).__init__()      

    def scan_callback(self, data):
        global state
        self.scan_data = data.ranges
        state = list(self.scan_data)
        for i in range(len(state)):
            if state[i] == np.inf:
                state[i] = 3.5

    def contact_callback(self, data):
        self.collision_data = data.states
        if len(self.collision_data) == 1:
            global collision_flag 
            collision_flag = True
            
    def groud_truth_callback(self,data):
        global robot_x
        global robot_y
        robot_x = data.pose.pose.position.x
        robot_y = data.pose.pose.position.y

    def run(self):
        global cmd_vel
        groud_truth_sub = rospy.Subscriber("/ground_truth_pose", Odometry, self.groud_truth_callback)
        scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        collision_sub = rospy.Subscriber("/contact", ContactsState, self.contact_callback)
        cmv_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        r = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            cmv_vel_pub.publish(cmd_vel)       
            try:
                r.sleep()
            except rospy.ROSTimeMovedBackwardsException as e:
                pass
