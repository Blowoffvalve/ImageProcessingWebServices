from flask import Flask, request, url_for, send_file
from flask_restful import Resource, Api
import cv2

app = Flask(__name__)
api = Api(app)

@app.route("/")
def index():
	return "Image processing functional decomposition"

@app.route("/grayScale", methods = ["POST"])
def grayScale():
	"""
	Convert an image from RGB to Grayscale
	"""
	file = request.files['image']
	file.save("./test_image.jpg")
	img = cv2.imread("./test_image.jpg")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("output_image.jpg", img)
	return send_file("output_image.jpg", mimetype = "image/jpg", as_attachment = True, attachment_filename="as.jpg")
	
if __name__ == "__main__":
	app.run(debug=True)
#api.add_resource(GrayScale, "/grayScale")
#api.add_resource(Binarizer, "/binarize")