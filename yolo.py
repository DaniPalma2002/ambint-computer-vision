import cv2
import yolov5

# initialize the video capture object
cap = cv2.VideoCapture(0)

model = yolov5.load('crowdhuman_yolov5m.pt')
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = True  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = 1  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference
model.line_thickness = 1  # bounding box thickness (pixels)

while True:
    ret, frame = cap.read()
    #img = cv2.imread('3.jpg')

    # run the YOLO model on the frame
    results: yolov5.models.common.Detections = model(frame)

    table = results.pandas().xyxy[0]

    print(table)
    print('-'*50)

    # show detection bounding boxes on image
    results.render()

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
