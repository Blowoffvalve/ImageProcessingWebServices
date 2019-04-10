FROM valian/docker-python-opencv-ffmpeg
MAINTAINER Opeoluwa Osunkoya "Osunkoyaopeoluwa@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential
RUN git clone https://github.com/Blowoffvalve/ImageProcessingWebServices.git
WORKDIR ImageProcessingWebServices
COPY . ../app
WORKDIR ../../app
RUN pip install flask flask_restful opencv_contrib_python numpy requests
ENTRYPOINT ["python"]
CMD ["server.py"]