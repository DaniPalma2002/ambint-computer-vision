import time
import cv2
import threading
from requests import head
import yolov5
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

handModel = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
handModel.confidence = 0.2
handModel.size = 412

cap = cv2.VideoCapture(0)

def detect_heads(frame, results):
    results[0] = headModel(frame)

def detect_hands(frame, results):
    _, _, _, results[1] = handModel.inference(frame)

while True:
    start_time = time.time()

    #ret, frame = cap.read()
    frame = cv2.imread('images/3.jpg')


    results = [None, None]

    # Run the head and hand detection in separate threads
    t1 = threading.Thread(target=detect_heads, args=(frame, results))
    t2 = threading.Thread(target=detect_hands, args=(frame, results))
    t1.start()
    t2.start()

    # # Wait for both threads to finish
    # while None in results:
    #     time.sleep(0.01)
    t1.join()
    t2.join()

    # Process the results...

    table = results[0].pandas().xyxy[0]

    # show detection bounding boxes on image
    results[0].render()

    hand_count = len(results[1])

    # display hands
    for detection in results[1][:hand_count]:
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
    cv2.putText(frame, f'Heads: {len(table)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Hands: {hand_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()