import requests
import math
import cv2
import numpy as np

video = cv2.VideoCapture("Road traffic video for object recognition.mp4")

def getNextServer():
	file = open("NextServer.txt", "r")
	return file.read()

#force 640x480 webcam resolution
video.set(3,640)
video.set(4,480)

for i in range(0,20):
    (grabbed, frame) = video.read()

while True:
    (grabbed, frame) = video.read()
    height = np.size(frame,0)
    width = np.size(frame,1)

    #if cannot grab a frame, this program ends here.
    if not grabbed:
        break

    r = requests.post("http://" + getNextServer() + "/imageProcessing", data = {"image":frame})
    print(r)
    break