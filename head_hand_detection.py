import math
from os import path
import sys
import time
import cv2
import yolov5
import requests
import json
import numpy as np
import pandas as pd

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), 'models')))
from yolo import YOLO

headModel = yolov5.load('models/crowdhuman_yolov5m.pt')
headModel.conf = 0.5  # NMS confidence threshold
headModel.iou = 0.45  # NMS IoU threshold
headModel.agnostic = True  # NMS class-agnostic
headModel.multi_label = False  # NMS multiple labels per box
headModel.classes = 1  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
headModel.max_det = 1000  # maximum number of detections per image
headModel.amp = False  # Automatic Mixed Precision (AMP) inference
headModel.line_thickness = 1  # bounding box thickness (pixels)
headModel.device = 'cuda'

handModel = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
handModel.confidence = 0.2
handModel.size = 416
handModel.device = 'cuda'

# ====Global variables==========================================================
head_count = 0
hand_count = 0

head_median = []

head_detection_flag = 0
hand_detection_flag = 0

head_matrix = np.zeros([3, 2], dtype = int)
# ==============================================================================

def processingFrame(frame):
    global head_count, hand_count, head_median, head_detection_flag, hand_detection_flag

    hand_detection_flag += 1
    head_detection_flag += 1

    # divide frame in sections
    height, width, _ = frame.shape

    # Divide the frame into a 3x3 grid
    section_width = width // 2
    section_height = height // 3

    # Draw the grid on the frame
    for i in range(1, 3):
        cv2.line(frame, (0, i * section_height), (width, i * section_height), (0, 255, 0), 1)

    for i in range(1, 2):    
        cv2.line(frame, (i * section_width, 0), (i * section_width, height), (0, 255, 0), 1)


    # head detection ===========================================================
    if head_detection_flag > 5:
        head_detection_flag = 0
        head_count = headDetection(frame)
        head_median.append(head_count)

    # hand detection ===========================================================
    if hand_detection_flag > 5:
        hand_detection_flag = 0
        hand_count = handDetection(frame)

    # number of hands and heads 
    cv2.putText(frame, f'Heads: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Hands: {hand_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # send request to server with head count
    if len(head_median) == 15:
        # remove outliers
        for i in range(3):
            head_median.remove(np.max(head_median))
        res = np.max(head_median)
        head_median = []
        postRequest(int(res))

    if (head_count > 0):
        print(f'Heads: {head_count} | Hands: {hand_count}')

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)


def headDetection(frame):
    results = headModel(frame)
    table = results.pandas().xyxy[0]
    head_count = len(table)
    results.render()

    for index, row in table.iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        # get centre of the head
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        print(f'Head: ({cx},{cy})')

    return head_count

def handDetection(frame):
    width, height, inference_time, results = handModel.inference(frame)
    print(f'hand Inference time: {inference_time}')
    # how many hands
    hand_count = len(results)

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    return hand_count


def headHandCameraDetection():
    # initialize the video capture object
    cap = cv2.VideoCapture(0)

    while True:
        processingFrame(cap.read()[1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def postRequest(heads):
    # The URL to which you are sending the POST request
    url = 'https://ami-dashboard-vercel-v2.vercel.app/data'

    # The data you want to send in JSON format
    data = {
        'info': heads,
    }

    # Convert the Python dictionary to a JSON string
    json_data = json.dumps(data)

    # Set the appropriate headers for JSON - this is important!
    headers = {'Content-Type': 'application/json'}

    # Send the POST request
    response = requests.post(url, data=json_data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        print('Success!')
        # If response is JSON, you can parse it into a Python dictionary
        response_data = response.json()
        print(response_data)
    else:
        print('Failed to send POST request')
        print('Status code:', response.status_code)
        print('Response:', response.text)


if __name__ == "__main__":
    headHandCameraDetection()