from flask import Flask, request, url_for, send_file, Response
from flask_restful import Resource, Api
import cv2
import requests
import os
import io
import random
import numpy as np
import time

#global variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
minContourArea = 300  #Adjust ths value according to tweak the size of the moving object found
binarizationThreshold = 30  #Adjust ths value to tweak the binarization
offsetEntranceLine = 30  #offset of the entrance line above the center of the image
offsetExitLine = 60
app = Flask(__name__)
api = Api(app)


@app.route("/")
def index():
	return "Image processing functional decomposition"

@app.route("/imageProcessing", methods = ["POST"])
def imageProcessing():
	"""
	Convert an image from RGB to Grayscale
	"""
	#receive the image from the request.
	file = request.files['image']
	file.save("./temp_image.jpg")
	frame = cv2.imread("./temp_image.jpg")
	#os.remove("./temp_image.jpg")

	#gray-scale conversion and Gaussian blur filter applying
	grayFrame = greyScaleConversion(frame)
	blurredFrame = gaussianBlurring(grayFrame)

	#Check if a frame has been previously processed and set it as the previous frame.
	referenceFrame = blurredFrame
	if os.path.exists("./previousImage.jpg"):
		referenceFrame = cv2.imread("./previousImage.jpg")[:,:,0]
	#Background subtraction and image binarization
	frameDelta = getImageDiff(referenceFrame, blurredFrame)
	cv2.imwrite("previousImage.jpg", blurredFrame)
	frameThresh = thresholdImage(frameDelta, binarizationThreshold)

	#Dilate image and find all the contours
	dilatedFrame = dilateImage(frameThresh)
	cv2.imwrite("dilatedFrame.jpg", dilatedFrame)
	cnts = getContours(dilatedFrame.copy())
	print(len(cnts))

	for c in cnts:
		print("x")
		if cv2.contourArea(c) < minContourArea:
			continue
		(x, y, w, h) = getContourBound(c)
		#grab an area 2 times larger than the contour.
		cntImage  = frame[y:y+int(1.2*w), x:x+int(1.2*h)]
		headers = {"enctype" : "multipart/form-data"}
		i = random.randint(1,1000)
		cv2.imwrite("ContourImages/contour"+str(i)+".jpg", cntImage)
		files = {"image":open("ContourImages/contour"+str(i)+".jpg", "rb")}
		r = requests.post("http://" + getNextServer() + "/objectClassifier", headers = headers, files = files )

	
	return Response(status=200)

@app.route("/objectClassifier", methods = ["POST"])
def classifier():
	"""
	Classify an object as either a car or a pedestrian.
	"""
	print(1)
	#initialize important variables
	minConfidence = 0.5
	thresholdValue = 0.3
	
	file = request.files['image']
	file.save("./classifier_image.jpg")
	frame = cv2.imread("./classifier_image.jpg")
	
	#Load YOLO weights from disk
	yolodir = "./yolo/"
	labelsPath = os.path.sep.join([yolodir, "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
	weightsPath = os.path.sep.join([yolodir, "yolov3.weights"])
	configPath = os.path.sep.join([yolodir, "yolov3.cfg"])
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	
	#Get Image dimensions
	image = cv2.copyMakeBorder(frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
	(H, W) = image.shape[:2]
	
	#Get the output layers parameters
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	
	#Create a blob to do a forward pass
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	
	boxes = []
	confidences = []
	classIDs = []
	for output in layerOutputs:
		#loop over each detection
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > minConfidence:
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
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfidence, thresholdValue)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			print(LABELS[classIDs[i]])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return LABELS[classIDs[i]]
	else:
		return Response(status=200)

def getNextServer():
	file = open("NextServer.txt", "r")
	return file.read()

def getContours(frame):
    """
    Get the contours in the frame 
    @return: contours
    """
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getContourBound(contour):
    """
    Returns a rectangle that is a bound around the contour.
    @return: contour Bounds
    """
    (x,y,w,h) = cv2.boundingRect(contour)
    return (x,y,w,h)

def thresholdImage(frame, binarizationThreshold=30):
    """
    Threshold the image to make it black and white. values above binarizationThreshold are made white and the rest
    is black
    """
    return cv2.threshold(frame, binarizationThreshold, 255, cv2.THRESH_BINARY)[1]

def getImageDiff(referenceFrame, frame):
    """
    Get the difference between 2 frames to isolate and retrieve only the moving object
    """
    return cv2.absdiff(referenceFrame, frame)

def gaussianBlurring(frame):
    """
    Preprocess the image by applying a gaussian blur
    """
    return cv2.GaussianBlur(frame, ksize =(11, 11), sigmaX = 0)

def greyScaleConversion(frame):
    """
    Convert the image from 3 channels to greyscale to reduce the compute required to run it.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def dilateImage(frame, interations=2 ):
    """
    Dilate the image to prevent spots that are black inside an image from being counted as individual objects
    """
    return cv2.dilate(frame, None, iterations=2)

if __name__ == "__main__":
	app.run(debug=True)