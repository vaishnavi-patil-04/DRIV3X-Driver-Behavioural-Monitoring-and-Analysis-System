import cv2
from core.pipeline import run_pipeline

cap = cv2.VideoCapture("testing/video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = run_pipeline(frame)

    print(output)  # later → alerts, logging, report
