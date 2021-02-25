#!/usr/bin/env python
import turtlebot
import rospy
import cv2

Stage = None

if __name__ == '__main__':

    rospy.init_node('image_converter', anonymous=True)
    robot = turtlebot.Turtlebot()
    rospy.sleep(3)

    while True:
        # To DO, complete the comptetion track using the robot object funtions
        
