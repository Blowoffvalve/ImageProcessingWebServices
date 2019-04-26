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
2. Clone the repository.
2. Start the server by the flask app [server.py](https://github.com/Blowoffvalve/ImageProcessingWebServices/blob/master/server.py).
3. Download the YOLO weights from [here](https://t.dripemail2.com/c/eyJhY2NvdW50X2lkIjoiNDc2ODQyOSIsImRlbGl2ZXJ5X2lkIjoiNjA5MjA5NTM2NCIsInVybCI6Imh0dHBzOi8vczMtdXMtd2VzdC0yLmFtYXpvbmF3cy5jb20vc3RhdGljLnB5aW1hZ2VzZWFyY2guY29tL29wZW5jdi15b2xvL3lvbG8tb2JqZWN0LWRldGVjdGlvbi56aXA_X19zPXFhZWJ1cHdpeGlzbjdmb2JqZnMzIn0) and save them to YOLO/

### Test
1. Run server.py : `python server.py`
2. Edit NextServer.txt to contain the IP and port(Default is 5000) of the server that is running server.py. If you are running it on the same server, the default value `localhost:5000` should suffice
3. Run client.py : `python client.py`
