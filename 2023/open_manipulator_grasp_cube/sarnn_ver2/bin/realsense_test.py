#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

rospy.init_node('test')

def call_back(ros_data):
    np_arr = np.fromstring(ros_data.data, np.uint8)
    print(np_arr)
    
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('camera/color/image_raw', Image, call_back)
    rospy.spin()

if __name__ == '__main__':
    listener()
    
    
