import math
from os import path
import sys
import time
import cv2
import yolov5
import requests
import json
import numpy as np

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), 'models')))
from yolo import YOLO

headModel = yolov5.load('models/crowdhuman_yolov5m.pt')
headModel.conf = 0.5  # NMS confidence threshold
headModel.iou = 0.45  # NMS IoU threshold
#headModel.agnostic = True  # NMS class-agnostic
#headModel.multi_label = False  # NMS multiple labels per box
headModel.classes = 1  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#headModel.max_det = 1000  # maximum number of detections per image
#headModel.amp = False  # Automatic Mixed Precision (AMP) inference
#headModel.line_thickness = 1  # bounding box thickness (pixels)
#headModel.device = 'cuda'

handModel = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
handModel.confidence = 0.2
handModel.size = 416
handModel.device = 'cuda'

head_median = []

def processingFrame(frame):
    global head_median
    # Record the time before processing the frame
    start_time = time.time()

    # head detection ===========================================================
    results = headModel(frame)
    table = results.pandas().xyxy[0]
    head_count = len(table)

    # show detection bounding boxes on image
    results.render()

    # hand detection ===========================================================
    width, height, inference_time, results = handModel.inference(frame)

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

    # number of hands and heads 
    cv2.putText(frame, f'Heads: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Hands: {hand_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Calculate the FPS
    fps = 1.0 / (time.time() - start_time)

    print(f'Heads: {head_count} | Hands: {hand_count} | FPS: {fps:.2f}')

    head_median.append(head_count)
    if (len(head_median) == 11):
        res = np.median(head_median)
        head_median = []
        postReq(int(res))

    # Display the FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)


def headHandCameraDetection():
    # initialize the video capture object
    cap = cv2.VideoCapture(1)

    while True:
        processingFrame(cap.read()[1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def postReq(heads):
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