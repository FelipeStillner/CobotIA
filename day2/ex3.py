# Student Task:
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROTOTXT = HERE / "MobileNetSSD_deploy.prototxt"
MODEL = HERE / "MobileNetSSD_deploy.caffemodel"
SNAPSHOT = HERE / "snapshot.jpg"

net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL))
classes = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
           'cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

img = cv2.imread(str(SNAPSHOT))
if img is None:
    raise FileNotFoundError(
        f"Could not read {SNAPSHOT}. Run day2/ex1.py first and press 'q' to save a snapshot."
    )
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 0.007843, (300,300), 127.5)
net.setInput(blob)
detections = net.forward()
for i in range(detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.4:
        idx = int(detections[0,0,i,1])
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype('int')
        label = f"{classes[idx]}: {confidence*100:.1f}%"
        cv2.rectangle(img, (startX, startY), (endX, endY), (0,255,0), 2)
        cv2.putText(img, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2.imshow('Detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()