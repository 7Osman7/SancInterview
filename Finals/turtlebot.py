#!/usr/bin/env python


import rospy
from tensorflow import keras
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from config import *
import cv2
import tensorflow as tf
import pickle
import os
from keras.preprocessing import image
import numpy as np

class Turtlebot:
    def __init__(self):
        self.__face_detector = cv2.CascadeClassifier(FACE_CASCADE)
        self.img = []
        self.img_depth = []

        # Yolo attributes
        self.person_coordinates = []

        # Tracker attributes 
        self.tracker_coordinates = []
        self.tracker_status = False
        self.__tracker = cv2.TrackerKCF_create()

        #Face Train/Detect attributes
        self.face_coordinates = []
        self.found_face = False
        self.data_collected = False

        # Speech Attributes
        self.listening = False
        self.latest_voice_msg = ""

        #YOLO Initializer
        LABELS = open(LABELS_PATH).read().strip().split("\n")
        self.__net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGTHS_PATH)

        #VGG16 Initializer
        self.__sess = tf.Session()
        self.__graph = tf.get_default_graph()
        tf.keras.backend.set_session(self.__sess)
        self.__model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')
        self.__modelLin = tf.keras.models.load_model(MODEL_PATH)
        name_file = open(NAMES_FILE,"rb")
        self.__names = pickle.load(name_file)
        name_file.close()
        self.__font = cv2.FONT_HERSHEY_SIMPLEX

        # RGB Image Subscriber 
        rospy.Subscriber(IMG_TOPIC,Image,self.__getImage)

        # Voice Subsciber 
        #rospy.Subscriber(,,self.__getVcommand)

        # Arm Publisher
        self.__armPub = rospy.Publisher('arm_pose', String, queue_size=3)

    def __getVcommand(self,data):
        if self.listening:
            self.latest_voice_msg = data.data


    def __getImage(self,data):
        try:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.img = cv_image
        except CvBridgeError as e:
            print(e)
    
    def track_init(self,InitBB):
        self.__tracker = cv2.TrackerKCF_create()
        self.__tracker.init(self.img, InitBB)
    
    def track(self):
        (success, box) = self.__tracker.update(self.img)
        self.tracker_coordinates = []
        self.tracker_status = success
        self.tracker_coordinates.append(box)

    def find_face_of(self,name):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.__face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                with self.__graph.as_default():
                    tf.keras.backend.set_session(self.__sess)
                    arr = cv2.resize(self.img[y:y+h,x:x+w],(224,224))
                    arr = image.img_to_array(arr)
                    arr = np.expand_dims(arr,axis=0)
                    arr = utils.preprocess_input(arr,version=1)
                    prediction = self.__model.predict(arr)
                    prediction = prediction.reshape(1,1,2048)
                    finalPred = self.__modelLin.predict(prediction)
                    if self.__names[np.argmax(finalPred)] == name:
                        self.face_coordinates = []
                        self.face_coordinates.append((x,y,w,h))
                        self.found_face = True
                        break 
                    else:
                        self.found_face = False                 
        else:
            self.found_face = False


    def collect_features(self,name):
        path = DATASET_PATH + name
        if not os.path.isdir(path):
            os.mkdir(path)
        count = len(os.listdir(path))
        if count < IMG_DATASET:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            face = self.__face_detector.detectMultiScale(gray, 1.3, 5)
            if len(face) > 0:
                print("YAY")
                x = face[0][0]
                y = face[0][1]
                w = face[0][2]
                h = face[0][3]
                cv2.imwrite(path+"/"+str(count)+".jpg",self.img[y:y+h,x:x+w])
            self.data_collected = False
        else:
            self.data_collected = True
        return self.data_collected

    def train_face(self,name):
        features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')
        fileName = FEATURES_PATH + name
        if not os.path.isfile(fileName):
            im_features = []
            for im in os.listdir(DATASET_PATH+str(name)):
                img = image.load_img(DATASET_PATH+str(name)+"/"+str(im),target_size=(224,224))
                x = image.img_to_array(img)
                x = np.expand_dims(x,axis=0)
                x = utils.preprocess_input(x,version=1)
                im_features.append(features.predict(x))
            file_out = open(fileName,"wb")
            pickle.dump(im_features,file_out)
            file_out.close()
            self.__train_model()

    def __train_model(self):
        people = [person for person in os.listdir(FEATURES_PATH)]
        test = []
        test_labels = []
        train = []
        train_labels = []

        names = {}

        split = 0.3

        count = -1
        for feat in people:
            count += 1
            names[count] = str(feat)
            fileName = FEATURES_PATH + str(feat)
            file_in = open(fileName,"rb")
            fullArr = pickle.load(file_in)
            nbFeat = -1
            for arr in fullArr:
                nbFeat += 1
                if nbFeat/len(fullArr) < 1 - split:
                    train.append(arr)
                    train_labels.append(count)
                else:
                    test.append(arr)
                    test_labels.append(count)

        names_file = open("names","wb")
        pickle.dump(names,names_file)
        names_file.close()

        train = np.asarray(train)
        test = np.asarray(test)

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1,2048)),
            keras.layers.Dense(7,activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(train, train_labels, epochs=10)\
        self.__model = model
        model.save(MODEL_PATH)


    def find_person(self):
        self.person_coordinates = []
        (H, W) = self.img.shape[:2]
        ln = self.__net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(self.img, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.__net.setInput(blob)
        layerOutputs = self.__net.forward(ln)

        boxes = []
        confidences = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                if classID != HUMAN_CLASS:
                    continue
                confidence = scores[classID]
                    
                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
            THRESHOLD)
    
        if len(idxs) > 0:
            for i in idxs.flatten():
                self.person_coordinates.append(boxes[i])

    def arm_pose(self, pose = "resting"):
        self.__armPub.publish(pose)

        

