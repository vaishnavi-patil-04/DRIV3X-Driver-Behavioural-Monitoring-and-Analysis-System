import cv2
from core.pipeline import run_pipeline
from behaviour.tracker import ObjectTracker
from behaviour.tailgating import TailgatingDetector
from behaviour.lead_vehicle import select_lead_vehicle


cap = cv2.VideoCapture("testing/test4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30

tracker = ObjectTracker()
tailgating = TailgatingDetector()

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    frame_h = frame.shape[0]

    output = run_pipeline(frame)

    # IMPORTANT: objects is now YOLO Results
    tracked_objects = tracker.update(output["objects"])

    lead_vehicle = select_lead_vehicle(
        tracked_objects,
        output["lanes"]
    )

    event = tailgating.update(
        lead_vehicle,
        frame_h,
        timestamp
    )

    if event:
        print("⚠️ EVENT:", event)

    frame_idx += 1
