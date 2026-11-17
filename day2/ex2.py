# Student Task:
from pathlib import Path

import cv2
import numpy as np

SNAPSHOT = Path(__file__).resolve().parent / "snapshot.jpg"

img = cv2.imread(str(SNAPSHOT))
if img is None:
    raise FileNotFoundError(
        f"Could not read {SNAPSHOT}. Run day2/ex1.py first and press 'q' to save a snapshot."
    )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()