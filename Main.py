import cv2

#img = cv2.imread('Screenshot 2024-04-21 211221.png')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
import os

script_dir = os.path.dirname(__file__)  # Get the directory of your script
classFile = os.path.join(script_dir, 'coco.names')
configPath = os.path.join(script_dir, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(script_dir, 'frozen_inference_graph.pb')

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame from the camera.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    print("Detected Objects:")
    print(classIds, confs, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1], (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0),2)

    cv2.imshow("OUTPUT", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
