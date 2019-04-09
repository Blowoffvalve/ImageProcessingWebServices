from flask import Flask, request, url_for, send_file, Response
from flask_restful import Resource, Api
import cv2
import requests
import os
import io
import random

#global variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
minContourArea = 600  #Adjust ths value according to tweak the size of the moving object found
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

	
	return send_file("dilatedFrame.jpg", mimetype = "image/jpg", as_attachment = True, attachment_filename="as.jpg")

@app.route("/objectClassifier", methods = ["POST"])
def classifier():
	"""
	Classify an object as either a car or a pedestrian.
	"""
	file = request.files['image']
	file.save("./classifier_image.jpg")
	frame = cv2.imread("./classifier_image.jpg")
	os.remove("./classifier_image.jpg")
	cv2.imwrite("image.jpg", frame)
	resp= Response("True")
	return resp

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