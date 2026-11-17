# Student Task:
from pathlib import Path

import cv2

SNAPSHOT = Path(__file__).resolve().parent / "snapshot.jpg"

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('UR5 Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(str(SNAPSHOT), frame)
        print(f"Saved {SNAPSHOT}")
        break
cap.release()
cv2.destroyAllWindows()