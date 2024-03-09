# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging
import sys
from os import path
import time
import cv2

import grpc
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), 'grpc_files')))
from grpc_files import image_pb2 as pb2
from grpc_files import image_pb2_grpc as pb2_grpc

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = pb2_grpc.ImageServiceStub(channel)
        captureAndSendFrames(stub)

def captureAndSendFrames(stub):
    # initialize the video capture object
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # serialize the frame
        _, frame_bytes = cv2.imencode('.jpg', frame)
        frame_bytes = frame_bytes.tobytes()
        # send the frame
        #time.sleep(0)
        stub.Image(pb2.FrameSend(frame=frame_bytes))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig()
    run()
