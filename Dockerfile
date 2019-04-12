FROM valian/docker-python-opencv-ffmpeg
MAINTAINER Opeoluwa Osunkoya "Osunkoyaopeoluwa@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential unzip
RUN wget -O yolo_weights.zip https://t.dripemail2.com/c/eyJhY2NvdW50X2lkIjoiNDc2ODQyOSIsImRlbGl2ZXJ5X2lkIjoiNjA5MjA5NTM2NCIsInVybCI6Imh0dHBzOi8vczMtdXMtd2VzdC0yLmFtYXpvbmF3cy5jb20vc3RhdGljLnB5aW1hZ2VzZWFyY2guY29tL29wZW5jdi15b2xvL3lvbG8tb2JqZWN0LWRldGVjdGlvbi56aXA_X19zPXFhZWJ1cHdpeGlzbjdmb2JqZnMzIn0
RUN unzip yolo_weights.zip
RUN git clone https://github.com/Blowoffvalve/ImageProcessingWebServices.git
WORKDIR ImageProcessingWebServices
RUN cp ../yolo-object-detection/yolo-coco/yolov3.weights ./YOLO/
RUN pip install flask flask_restful opencv_contrib_python numpy requests
ENTRYPOINT ["python"]
CMD ["server.py"]