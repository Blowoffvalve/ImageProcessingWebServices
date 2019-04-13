import requests
import math
import cv2
import numpy as np
import time

frameCount = 1
peakFPS = 0
minFPS = 10000
averageFPS =0
FPS = []
startTime = time.time()
video = cv2.VideoCapture("Road traffic video for object recognition.mp4")

def getNextServer():
	file = open("NextServer.txt", "r")
	return file.read()[:-1]

uri = "http://" + getNextServer() + "/frameProcessing"
#force 640x480 webcam resolution
video.set(3,640)
video.set(4,480)

for i in range(0,20):
    (grabbed, frame) = video.read()

while True:
	frameStartTime = time.time()
	frameCount+=1
	(grabbed, frame) = video.read()
	height = np.size(frame,0)
	width = np.size(frame,1)

    #if cannot grab a frame, this program ends here.
	if not grabbed:
		break
	cv2.imwrite("Frame.jpg", frame)

	r = requests.post(uri, files = {'image' : open("Frame.jpg", "rb")})
	currentFPS = 1.0/(time.time() - frameStartTime)
	FPS.append(currentFPS)
	print(r, round(np.mean(FPS), 3))
	if r == "<Response [500]>":
		break
print("Average FPS = {}".format(round(np.mean(FPS), 3)))
print("RunTimeInSeconds = {}".format(round(frameStartTime - startTime, 3)))