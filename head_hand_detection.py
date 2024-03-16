import asyncio
from os import path
import sys
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
headModel.agnostic = True  # NMS class-agnostic
headModel.multi_label = False  # NMS multiple labels per box
headModel.classes = 1  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
headModel.max_det = 1000  # maximum number of detections per image
headModel.amp = False  # Automatic Mixed Precision (AMP) inference
headModel.line_thickness = 1  # bounding box thickness (pixels)
headModel.device = 'cuda'

handModel = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
handModel.confidence = 0.2
handModel.size = 416
handModel.device = 'cuda'

# ====Global variables==========================================================
head_count = 0
hand_count = 0

head_list = []

head_detection_flag = 0
hand_detection_flag = 0

SECTION_WIDTH = 0
SECTION_HEIGHT = 0
height = 0
width = 0

HEAD_FLAG_SIZE = 10
HAND_FLAG_SIZE = 50000
HEAD_LIST_SIZE = 15
# ==============================================================================

def headHandCameraDetection():
    global SECTION_HEIGHT, SECTION_WIDTH, height, width
    # initialize the video capture object
    cap = cv2.VideoCapture(1)

    # divide frame in sections
    height, width, _ = cap.read()[1].shape
    
    # Divide the frame into a 2x3 grid
    SECTION_WIDTH = width // 2
    SECTION_HEIGHT = height // 3

    while True:
        processingFrame(cap.read()[1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def processingFrame(frame):
    global head_count, hand_count, head_list, head_detection_flag, hand_detection_flag

    hand_detection_flag += 1
    head_detection_flag += 1

    # Draw the grid on the frame
    for i in range(1, 3):
        cv2.line(frame, (0, i * SECTION_HEIGHT), (width, i * SECTION_HEIGHT), (0, 255, 0), 1)

    for i in range(1, 2):    
        cv2.line(frame, (i * SECTION_WIDTH, 0), (i * SECTION_WIDTH, height), (0, 255, 0), 1)


    # head detection ===========================================================
    if head_detection_flag > HEAD_FLAG_SIZE:
        head_detection_flag = 0
        head = headDetection(frame)
        head_count = head[0]
        head_list.append(head)

    # hand detection ===========================================================
    if hand_detection_flag > HAND_FLAG_SIZE:
        hand_detection_flag = 0
        hand_count = handDetection(frame)

    # number of hands and heads 
    cv2.putText(frame, f'Heads: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Hands: {hand_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # send request to server with head count
    if len(head_list) >= HEAD_LIST_SIZE:
        # remove outliers
        # for i in range(HEAD_LIST_SIZE//5):
        #     head_list.remove(np.max([x[0] for x in head_list]))
        #res = np.max(head_list)
        res = max(head_list[::-1], key=lambda x: x[0])
        matrix = storeHeadPositionsMatrix(res[1])
        print(matrix)
        head_list = []
        postRequest(int(res[0]))
        # TODO send matrix to server

    if (head_count > 0 or hand_count > 0):
        print(f'Heads: {head_count} | Hands: {hand_count}')

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)


def headDetection(frame):
    results = headModel(frame)
    table = results.pandas().xyxy[0]
    head_count = len(table)
    results.render()

    return (head_count, table)


def storeHeadPositionsMatrix(table):
    head_matrix = np.zeros([3, 2], dtype = int)

    for index, row in table.iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        # get centre of the head
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        print(f'Head: ({cx},{cy})')

        # store head position in matrix
        if (cy < SECTION_HEIGHT):
            if (cx < SECTION_WIDTH):
                head_matrix[0][0] += 1
            else:
                head_matrix[0][1] += 1
        elif (cy < 2 * SECTION_HEIGHT):
            if (cx < SECTION_WIDTH):
                head_matrix[1][0] += 1
            else:
                head_matrix[1][1] += 1
        else:
            if (cx < SECTION_WIDTH):
                head_matrix[2][0] += 1
            else:
                head_matrix[2][1] += 1

    return head_matrix


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



def postRequest(heads):
    # The URL to which you are sending the POST request
    url = 'https://smart-engagement-room.vercel.app/data'

    # The data you want to send in JSON format
    data = {
        'totalNumber': heads,
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
        if response.text:
            response_data = response.text
            print(response_data)
    else:
        print('Failed to send POST request')
        print('Status code:', response.status_code)
        print('Response:', response.text)


if __name__ == "__main__":
    headHandCameraDetection()