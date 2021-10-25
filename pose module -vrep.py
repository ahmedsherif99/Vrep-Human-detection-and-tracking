import cv2
import sim
import mediapipe as mp
import time
import math
import numpy as np
print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',20000,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # Get the object Handles from Vrep scene
    # We will use the handles of the motors and the camera
    errorCode, leftMotor = sim.simxGetObjectHandle(clientID, 'blackleft', sim.simx_opmode_blocking)
    errorCode, rightMotor = sim.simxGetObjectHandle(clientID, 'backright', sim.simx_opmode_blocking)
    errorCode, leftMotorf = sim.simxGetObjectHandle(clientID, 'frontleft', sim.simx_opmode_blocking)
    errorCode, rightMotorf = sim.simxGetObjectHandle(clientID, 'frontright', sim.simx_opmode_blocking)
    #errorCode, cam = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_blocking)
    errorCode, cam = sim.simxGetObjectHandle(clientID, 'Vision_sensorh', sim.simx_opmode_blocking)

    # initialize the streaming variables
    errorCode, resolution, image = sim.simxGetVisionSensorImage(clientID, cam, 0, sim.simx_opmode_streaming)

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    #cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        errorCode, resolution, image = sim.simxGetVisionSensorImage(clientID, cam, 0, sim.simx_opmode_buffer)
        # errorCode, resolution, capture = sim.simxGetVisionSensorImage(clientID, cam, 0, sim.simx_opmode_buffer)
        # The image is sent as 1D array we will need to reshape it to HxWx3 for RGB images
        # First we convert it to a numpy array
        img = np.array(image, dtype=np.uint8)

        # Check if the image is read successfully
        if resolution:
            # Reshape the numpy array
            img = img.reshape([resolution[0], resolution[1], 3])
            # openCV deals with images as BGR by default and the sent image is in RGB
            # Here we convert it to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # The image is inverted as the orgin is in the button left corner in the Vrep
            # While it's on the top left corner in openCV so we have to flip the image
            img = cv2.flip(img, 0)
            #success, img = cap.read()
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                print(lmList[14])
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()