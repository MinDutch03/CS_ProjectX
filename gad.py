import cv2
import math
import time
import matplotlib.pyplot as plt
import os
from logging import exception
import  sys
#from  Adafruit_IO import  MQTTClient
#from sqlalchemy import except_


# define stuff for Adafruit.
AIO_FEED_ID = ""
AIO_USERNAME = "namelessbtw"
AIO_KEY = ""
'''
def  connected(client):
    print("Service connected")
    client.subscribe(AIO_FEED_ID)

def  subscribe(client , userdata , mid , granted_qos):
    print("Subscribed")

def  disconnected(client):
    print("Disconnected!!!")
    sys.exit (1)

def  message(client , feed_id , payload):
    print("Data received " + payload)

client = MQTTClient(AIO_USERNAME , AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()
'''


# get models and defining categories and statistics
faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"

ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) # model value

ageList = [
    '(0-6)', '(6-13)', '(14-19)', '(20-29)',
    '(30- 39)', '(40-55)', '(56-60)', '(61-70)', '(71-100)'
]
genderList = ["Male", "Female"]

# for storing values.
predicted_age = []
predicted_gender = []
confidenceAge = []
confidenceGender = []

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
padding = 20



# defining function
def getFaceBox(net, frame, conf_threshold=0.7):
    ''' 
    Detect the face of the person inside the image and Output a bounding box 
    encapsulating the face detected.
    '''

    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    
    return frameOpencvDnn, bboxes


def age_gender_detector(frame):
    '''
    Get face of the person inside inputted `frame` and predict their `gender` and `age`.

    - Age is categorised into several groups, which are determined based on the 
    distiction of the face in some age periods throughout their life. 
    All group age: `(0-6)`, `(6-13)`, `(14-19)`, `(20-29)`, `(30- 39)`, `(40-55)`, `(56-60)`, `(61-70)`, `(71-100)`.
    
    - Gender is categorised into only `Male` and `Female`.

    ------------------------------------------------------------------------------------------
    Output:
    - Print out the predicted age, gender and the confidence of the prediction for each instance. 
    '''

    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)

    if not bboxes:
        print("No face detected")

    for bbox in bboxes:
        # print(bbox)
        face = frame[
            max(0, bbox[1] - padding): min(bbox[3] + padding, frame.shape[0] - 1),
            max(0, bbox[0] - padding): min(bbox[2] + padding, frame.shape[1] - 1),
        ]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        print("Gender : {}, conf = {:.3f}".format(
            gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        
        age = ageList[agePreds[0].argmax()]

        #print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        # store values for iot
        
        print("Update Age:", age) # check age name
        # client.publish("Age", age)

        print("Update Gender:", gender) # check age name
        # client.publish("Gender", gender)
        
        conf_age = agePreds[0].max()
        print("Update Age Confidence:", conf_age) # check age name
        # client.publish("Age_Confidence", conf_age)
        
        conf_gender = genderPreds[0].max()
        print("Update Confidence:", conf_gender) # check age name
        # client.publish("Gender_confidence", conf_gender)
        

        label = "{} , {}".format(gender, age)
        cv2.putText(
            img= frameFace,
            text= label,
            org= (bbox[0], bbox[1] - 10),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX,
            fontScale= 0.8,
            color= (0, 255, 255),
            thickness= 2,
            # cv2. LINE_AA gives anti-aliased line.
            lineType= cv2.LINE_AA,
        )
    return frameFace


def show_results(folder):
    # load all images in the folder img
    images = []

    # listdir() returns a list containing the names of the entries in the directory given by path
    for filename in os.listdir(folder):

        # cv2.imread: load an image from the specified file
        # os.path.join: combine path names into one complete path
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

        # results
        output = age_gender_detector(img)
        rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(rgb_output)
        # plt.show()


# show_results("./celeba-dataset/img_align_celeba/img_align_celeba")
show_results("./img")

#print(predicted_age)
#print(predicted_gender)
#print(confidenceAge)
#print(confidenceGender)
