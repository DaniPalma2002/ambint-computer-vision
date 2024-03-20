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
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
import sys
from os import path
import cv2

import grpc
import numpy as np
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), 'grpc_files')))
from grpc_files import image_pb2 as pb2
from grpc_files import image_pb2_grpc as pb2_grpc
from google.protobuf import empty_pb2

import head_hand_detection

identify_student = True

class ImageService(pb2_grpc.ImageServiceServicer):
    def Image(self, request, context):
        #print("Received frame")
        if not imageHandler(request.frame):
            return pb2.FrameResponse(camera_off=True)
        return pb2.FrameResponse(camera_off=False)

def imageHandler(frame_bytes):
    global identify_student
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ret = head_hand_detection.processingFrame(img_np, identify_student)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        identify_student = not identify_student
        print(f'IDENTIFY_STUDENT: {identify_student}')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

    if ret == 'off':
        return False
    return True
    

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ImageServiceServicer_to_server(ImageService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
