# ImageProcessingWebServices

## Object Detection
[server.py](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/server.py) implements a RESTFUL webservice for object detection.
The following endpoints currently exist.

1. /frameProcessing: This uses openCv methods to find the objects that have changed from a preceding frame. It sends each object to /classifier.
2. /objectClassifier: This is a deep learning model(Yolo-V3 trained on COCO) that attempts to classify an image into a fixed set of classes. It increments the [counter](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/output.txt) for that object class.
3. /init: This initializes all counters to 0
4. /getCounts: This retrieves the current value of the counters.

[client.py](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/client.py) reads a video and sends frames to server.py.

[Dockerfile](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/Dockerfile) for starting up a container with the server already running.

### Setup
1. Install your virtual environment to have the packages specified in [requirements.txt](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/requirements.txt).
2. Start the server by the flask app [server.py](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/server.py).

### Test
1. Unzip [testImages.rar](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/testImages.rar) to the same directory as server.py
2. Send Frame.jpg(it is inside testImages.rar) to a newly setup instance of the app.
