import cv2
import torch
import yolov5
#from helper import create_video_writer


# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# initialize the video capture object
cap = cv2.VideoCapture(0)

model = yolov5.load('crowdhuman_yolov5m.pt')

while True:
    # start time to compute the fps
    # start = datetime.datetime.now()

    ret, frame = cap.read()

    # run the YOLO model on the frame
    results = model(frame)

    print('results:', results.boxes.data.tolist())

    # show detection bounding boxes on image
    results.render()

    # # loop over the detections
    # for data in detections.boxes.data.tolist():
    #     # extract the confidence (i.e., probability) associated with the detection
    #     confidence = data[4]

    #     # filter out weak detections by ensuring the 
    #     # confidence is greater than the minimum confidence
    #     if float(confidence) < CONFIDENCE_THRESHOLD:
    #         continue

    #     # if the confidence is greater than the minimum confidence,
    #     # draw the bounding box on the frame
    #     xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    #     cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

    #     # end time to compute the fps
    # end = datetime.datetime.now()
    # # show the time it took to process 1 frame
    # total = (end - start).total_seconds()
    # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # # calculate the frame per second and draw it on the frame
    # fps = f"FPS: {1 / total:.2f}"
    # cv2.putText(frame, fps, (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()