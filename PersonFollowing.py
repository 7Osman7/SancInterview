#!/usr/bin/env python
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os

import roslib
roslib.load_manifest('faceReco')
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist

# depth_data = None
navigateTo = None
body = 4
tracker = cv2.TrackerKCF_create()
FaceFound = False
BodyFound = False

def change_target(data):
    global navigateTo
    navigateTo = data.data
    global body 
    body = 1

def follow_body(data):
    #global navigateTo
    #navigateTo = data.data
    global body 
    body = 2
	
def stop(data):
    #global navigateTo
    #navigateTo = data.data
    global body 
    body = 0

def callback(data):
    try:
        img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #global body

    if body == 1:
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            global sess
            global graph
            with graph.as_default():
                tf.keras.backend.set_session(sess)
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                arr = cv2.resize(img[y:y+h,x:x+w],(224,224))
                arr = image.img_to_array(arr)
                arr = np.expand_dims(arr,axis=0)
                arr = utils.preprocess_input(arr,version=1)
                prediction = model.predict(arr)
                prediction = prediction.reshape(1,1,2048)
                finalPred = modelLin.predict(prediction)
                if (names[np.argmax(finalPred)] == navigateTo):
                    if((2*x+w)/2 > 2*(img.shape[0]/3)):
                        vel_msg.angular.z = -0.35
                        vel_msg.linear.x = 0.2 #0
                    elif((2*x+w)/2 < (img.shape[0]/3)):
                        vel_msg.angular.z = 0.35
                        vel_msg.linear.x = 0.2 #0
                    else:
                        vel_msg.angular.z = 0
                        vel_msg.linear.x = 0.25
                    vel_pub.publish(vel_msg)
            cv2.putText(img, names[np.argmax(finalPred)], (x+5,y-5), font, 1, (255,255,255), 2)
    elif body == 0:
        vel_msg.angular.z = 0
        vel_msg.linear.x = 0
        vel_pub.publish(vel_msg)

	
    elif body == 2:
        #img = cv2.resize(img, (320, 240))
        if BodyFound == False:
            #gray = cv2.resize(gray, (320, 240))
            #bodies = body_detector.detectMultiScale(gray, 1.3, 5)


            (H, W) = img.shape[:2]
            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)


            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    if classID != HumanClass:
                        continue
                    confidence = scores[classID]
                    
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > CONFIDENCE:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)


            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
                THRESHOLD)
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    # if((2*x+w)/2 > 2*(img.shape[0]/3)):
                    #     vel_msg.angular.z = -0.2
                    #     vel_msg.linear.x = 0.1 #0
                    # elif((2*x+w)/2 < (img.shape[0]/3)):
                    #     vel_msg.angular.z = 0.2
                    #     vel_msg.linear.x = 0.1 #0
                    # else:
                    #     vel_msg.angular.z = 0
                    #     vel_msg.linear.x = 0.3
                global tracker
                tracker = cv2.TrackerKCF_create()
                initBB = (x,y,w,h)
                tracker.init(img, initBB)
                global BodyFound
                BodyFound = True
        else: 
            h, w = img.shape[:2]
            (success, box) = tracker.update(img)
            if success:
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if((2*x+w)/2 > 2*(img.shape[0]/3)):
                    vel_msg.angular.z = -0.25
                    vel_msg.linear.x = 0.1 #0
                elif((2*x+w)/2 < (img.shape[0]/3)):
                    vel_msg.angular.z = 0.25
                    vel_msg.linear.x = 0.1 #0
                else:
                    vel_msg.angular.z = 0
                    vel_msg.linear.x = 0.25
                #x_center = (2*x + w)/2
                #msg = rospy.wait_for_message("/camera/depth_registered/image", Image)
                #depth_data = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                #if depth_data[y,x_center] > 500:
                vel_pub.publish(vel_msg)
            else:
		        global BodyFound
		        BodyFound = False


    cv2.imshow('image', img)
    cv2.waitKey(3)

sess = tf.Session()
graph = tf.get_default_graph()

tf.keras.backend.set_session(sess)
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')
modelLin = tf.keras.models.load_model("/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/pouet.model")

name_file = open("/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/names","rb")
names = pickle.load(name_file)
name_file.close()
print(names)

bridge = CvBridge()
face_detector = cv2.CascadeClassifier('/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/haarcascade_frontalface_default.xml')
body_detector = cv2.CascadeClassifier('/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/haarcascade_lowerbody.xml')
image_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle",Image,callback)
name_sub = rospy.Subscriber("/orders/find",String,change_target)
follow_sub = rospy.Subscriber("/orders/follow",String,follow_body)
stop_sub = rospy.Subscriber("/orders/stop",String,stop)
#depth_sub = rospy.Subscriber("/camera/depth_registered/image_throttle", Image, depth_callback)

vel_pub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=10)
vel_msg = Twist()

#navigateTo = "Osman"
#body = False

vel_msg.linear.x = 0
vel_msg.linear.y = 0
vel_msg.linear.z = 0
vel_msg.angular.x = 0
vel_msg.angular.y = 0
vel_msg.angular.z = 0


THRESHOLD = 0.3
CONFIDENCE = 0.5
HumanClass = 0

# load the COCO class labels our YOLO model was trained on
labelsPath = "/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/darknet/data/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = "/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/darknet/yolov3.weights"
configPath =  "/home/Osman/comp_ws/src/FaceReco/faceReco/scripts/darknet/cfg/yolov3.cfg" 

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

font = cv2.FONT_HERSHEY_SIMPLEX

if os.path.isfile("features_file.pickle"):
    prev_feat = open("features_file.pickle","rb")
    feats = pickle.load(prev_feat)
    prev_feat.close()

rospy.init_node('image_converter', anonymous=True)
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()
