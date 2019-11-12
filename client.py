import math
import cv2
import numpy as np
import time
from http3_client import HttpClient, save_session_ticket

from aioquic.asyncio.client import connect
from aioquic.h3.connection import H3_ALPN
from aioquic.quic.configuration import QuicConfiguration
from aioquic.h3.events import DataReceived, H3Event, HeadersReceived
from typing import cast
import asyncio
import argparse
import sys
from urllib.parse import urlparse

def getNextServer():
    file = open("NextServer.txt", "r")
    return file.read()[:-1]

def save_session_ticket(ticket):
    """
    Callback which is invoked by the TLS engine when a new session ticket
    is received.
    """
    logger.info("New session ticket received")
    with open('session_ticket', "wb") as fp:
        pickle.dump(ticket, fp)

async def run(args):
    frameCount = 1
    peakFPS = 0
    minFPS = 10000
    averageFPS =0
    FPS = []
    startTime = time.time()
    video = cv2.VideoCapture("Road traffic video for object recognition.mp4")

    uri = "http://" + getNextServer() + "/frameProcessing"
    parsed = urlparse(uri)
    host, port_str = parsed.netloc.split(":")
    port = int(port_str)
    #force 640x480 webcam resolution
    video.set(3,640)
    video.set(4,480)

    for i in range(0,20):
        (grabbed, frame) = video.read()
    configuration = QuicConfiguration(
        is_client=True, alpn_protocols=H3_ALPN
    )
    configuration.load_verify_locations(args.ca_certs)
    async with connect(
            host,
            port,
            configuration=configuration,
            create_protocol=HttpClient,
            session_ticket_handler=save_session_ticket,
        ) as client:
            client = cast(HttpClient, client)
            while True:
                frameStartTime = time.time()
                frameCount+=1
                (grabbed, frame) = video.read()
                height = np.size(frame,0)
                width = np.size(frame,1)
                #if cannot grab a frame, this program ends here.
                if not grabbed:
                    break
                #cv2.imwrite("Frame.jpg", frame)
                #print(frame.shape)
                data = {"Frame":frame.tolist()}

                http_events = await client.post(
                    u"/frameProcessing",
                    data=str(data).encode('utf-8'),
                )
                currentFPS = 1.0/(time.time() - frameStartTime)
                FPS.append(currentFPS)
                for http_event in http_events:
                    if isinstance(http_event, HeadersReceived):
                        headers = b""
                        for k, v in http_event.headers:
                            headers += k + b": " + v + b"\r\n"
                        if headers:
                            sys.stderr.buffer.write(headers + b"\r\n")
                            sys.stderr.buffer.flush()
                        print("frame = {}, fps = {} ".format(frameCount, round(currentFPS, 3) ))

            print("Average FPS = {}".format(round(np.mean(FPS), 3)))
            print("RunTimeInSeconds = {}".format(round(frameStartTime - startTime, 3)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HTTP/3 client")
    parser.add_argument(
        "--ca-certs", type=str, help="load CA certificates from the specified file"
    )
    parser.add_argument(
        "-s",
        "--session-ticket",
        type=str,
        help="read and write session ticket from the specified file",
    )
    args = parser.parse_args()
    asyncio.run(run(args))
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(
    #     run(args)
    # )
