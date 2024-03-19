from os import path
import sys
import time
import cv2
import yolov5
import requests
import json
import numpy as np
import ambint_compreface

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

head_list = []

head_detection_flag = 0
hand_detection_flag = 0
student_identification_flag = 0

SECTION_WIDTH = 0
SECTION_HEIGHT = 0
height = 0
width = 0

HEAD_FLAG_SIZE = 4
HAND_FLAG_SIZE = 50
HEAD_LIST_SIZE = 10

STUDENT_FLAG_SIZE = 150
# ==============================================================================

def headHandCameraDetection():
    global SECTION_HEIGHT, SECTION_WIDTH, height, width
    # initialize the video capture object
    cap = cv2.VideoCapture(0)

    # divide frame in sections
    height, width, _ = cap.read()[1].shape
    
    # Divide the frame into a 2x3 grid
    SECTION_WIDTH = width // 2
    SECTION_HEIGHT = height // 3

    identify_student = True
    while True:
        processingFrame(cap.read()[1], identify_student)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            identify_student = not identify_student
            print(f'IDENTIFY_STUDENT: {identify_student}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def processingFrame(frame, identify_student):
    global head_count, hand_count, head_list, head_detection_flag, hand_detection_flag, student_identification_flag

    hand_detection_flag += 1
    head_detection_flag += 1
    student_identification_flag += 1

    # write in image if student identification is on
    cv2.putText(frame, f'Student identification: {identify_student}', (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # student identification
    if identify_student:
        if student_identification_flag > STUDENT_FLAG_SIZE:
            student_identification_flag = 0
            # identify student
            print('Identifying student')
            cv2.imwrite('images/frame.jpg', frame)
            students = ambint_compreface.identify_students()
            students = list(set(students))
            print('Students identified: ', '|'.join(students))
            studentPostReq(students)
        # show the frame to our screen
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        return

    # Draw the grid on the frame
    for i in range(1, 3):
        cv2.line(frame, (0, i * SECTION_HEIGHT), (width, i * SECTION_HEIGHT), (0, 255, 0), 1)

    for i in range(1, 2):    
        cv2.line(frame, (i * SECTION_WIDTH, 0), (i * SECTION_WIDTH, height), (0, 255, 0), 1)


    # head detection ===========================================================
    if head_detection_flag > HEAD_FLAG_SIZE:
        head_detection_flag = 0
        head_count, _ = headDetection(frame)
        print(f'Heads: {head_count} | Hands: {hand_count}')
        head_list.append(head_count)


    # hand detection ===========================================================
    if hand_detection_flag > HAND_FLAG_SIZE:
        hand_detection_flag = 0
        hand_count, hand_results = handDetection(frame)
        print(f'Heads: {head_count} | Hands: {hand_count}')
        head_count, head_table = headDetection(frame)
        matrix = storeHeadPositionsMatrix(frame, head_table, hand_results)
        matrixPostReq(matrix)
            

    # number of hands and heads 
    cv2.putText(frame, f'Heads: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Hands: {hand_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # send request to server with head count
    if len(head_list) >= HEAD_LIST_SIZE:
        # remove outliers
        for i in range(HEAD_LIST_SIZE//5):
            head_list.remove(np.max(head_list))
        res = np.max(head_list)
        head_list = []
        headCountPostReq(int(res))

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)



def storeHeadPositionsMatrix(frame, head_table, hand_results):
    matrix = np.zeros([3, 2], dtype=object)
    for i in range(3):
        for j in range(2):
            matrix[i][j] = [0,0]

    for _, row in head_table.iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        # get centre of the head
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2

        # store head position in matrix
        sectionDetection(matrix, cx, cy, 0)

    # display hands
    for detection in hand_results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        sectionDetection(matrix, cx, cy, 1)

    return matrix


def sectionDetection(matrix, cx, cy, flag):
    if (cy < SECTION_HEIGHT):
        if (cx < SECTION_WIDTH):
            matrix[0][0][flag] += 1
        else:
            matrix[0][1][flag] += 1
    elif (cy < 2 * SECTION_HEIGHT):
        if (cx < SECTION_WIDTH):
            matrix[1][0][flag] += 1
        else:
            matrix[1][1][flag] += 1
    else:
        if (cx < SECTION_WIDTH):
            matrix[2][0][flag] += 1
        else:
            matrix[2][1][flag] += 1


def headDetection(frame):
    results = headModel(frame)
    table = results.pandas().xyxy[0]
    head_count = len(table)
    print(f'Heads: {head_count}')
    results.render()

    return head_count, table

def handDetection(frame):
    _, _, inference_time, results = handModel.inference(frame)
    print(f'hand Inference time: {inference_time}')
    # how many hands
    hand_count = len(results)

    return hand_count, results


def headCountPostReq(heads):
    # The URL to which you are sending the POST request
    url = 'https://smart-engagement-room.vercel.app/data'

    # The data you want to send in JSON format
    data = {
        'totalNumber': heads,
    }

    sendPostRequest(url, data)

def matrixPostReq(matrix: np.ndarray):
    url = 'https://smart-engagement-room.vercel.app/regions'

    data = {
        'regions': [],
    }

    i = 1
    for val in matrix.flatten():
        data['regions'].append({
            'id': i,
            'number': val[0],
            'question': True if val[1] > 0 else False
        })
        i += 1

    sendPostRequest(url, data)

def studentPostReq(students):
    url = 'https://smart-engagement-room.vercel.app/api/attendance'

    for student in students:
        data = {
            "ist-number": student,
            "attendance": True,
        }
        sendPostRequest(url, data)

# Send the POST request
def sendPostRequest(url, data):
    # Convert the Python dictionary to a JSON string
    json_data = json.dumps(data)

    print(json_data)

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