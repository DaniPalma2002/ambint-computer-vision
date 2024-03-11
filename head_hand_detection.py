from os import path
import sys
import time
import cv2
import yolov5

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


def processingFrame(frame):
    # Record the time before processing the frame
    start_time = time.time()

    # head detection ===========================================================
    results = headModel(frame)
    table = results.pandas().xyxy[0]
    print(table)

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
    cv2.putText(frame, f'Heads: {len(table)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Hands: {hand_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Calculate the FPS
    fps = 1.0 / (time.time() - start_time)

    print(f'Heads: {len(table)} | Hands: {hand_count} | FPS: {fps:.2f}')

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


if __name__ == "__main__":
    headHandCameraDetection()