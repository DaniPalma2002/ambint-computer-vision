import sys
from os import path
import cv2
import grpc
import threading
from paho.mqtt import client as mqtt_client
import random

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), 'grpc_files')))
from grpc_files import image_pb2 as pb2
from grpc_files import image_pb2_grpc as pb2_grpc

movement_detected = False
movement_lock = threading.Lock()

SERVER_IP = "192.168.37.55"

broker = '192.168.37.199'
port = 1883
topic = "ambint/sensor"
client_id = f'python-mqtt-{random.randint(0, 1000)}'

camera_off = True

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        payload = msg.payload.decode()
        print(f"Received `{payload}` from `{msg.topic}` topic")
        if payload == "Movement":
            set_movement_detected()
    client.subscribe(topic)
    client.on_message = on_message

def set_movement_detected():
    global movement_detected
    with movement_lock:
        movement_detected = True

def run():
    mqtt_client = connect_mqtt()
    subscribe_thread = threading.Thread(target=subscribe, args=(mqtt_client,))
    subscribe_thread.start()
    mqtt_client.loop_start()

    with grpc.insecure_channel(f"{SERVER_IP}:50051") as channel:
        stub = pb2_grpc.ImageServiceStub(channel)
        captureAndSendFrames(stub)

def captureAndSendFrames(stub):
    global movement_detected, camera_off
    # initialize the video capture object
    cap = cv2.VideoCapture(0)
    while True:
        if camera_off:
            if movement_detected:
                print("Movement detected!")
                camera_off = False
            else:
                print("camera is still off")
                continue
        
        movement_detected = False

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
    
        # serialize the frame
        _, frame_bytes = cv2.imencode('.jpg', frame)
        frame_bytes = frame_bytes.tobytes()
        # send the frame
        response = stub.Image(pb2.FrameSend(frame=frame_bytes))
        if response.camera_off:
            print("Camera is off")
            camera_off = True

    cap.release()

if __name__ == "__main__":
    try:
        run()
    except grpc.RpcError as e:
        print(f"Caught an RpcError: {e}")
        exit()
