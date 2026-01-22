"""
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
"""

"""
import cv2
from core.pipeline import run_pipeline
from behaviour.tracker import ObjectTracker
from behaviour.tailgating import TailgatingDetector
from behaviour.lead_vehicle import select_lead_vehicle
from behaviour.lane_departure import LaneDepartureDetector  # new import

# Video
cap = cv2.VideoCapture("testing/test4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Detectors
tracker = ObjectTracker()
tailgating = TailgatingDetector()
lane_departure = LaneDepartureDetector(offset_threshold=50)  # new detector

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    frame_h = frame.shape[0]

    # Run detection pipeline
    output = run_pipeline(frame)

    # Update tracker
    tracked_objects = tracker.update(output["objects"])

    # Identify lead vehicle
    lead_vehicle = select_lead_vehicle(tracked_objects, output["lanes"])

    # Tailgating detection
    tg_event = tailgating.update(lead_vehicle, frame_h, timestamp)
    if tg_event:
        print("⚠️ EVENT:", tg_event)

    # Lane departure detection
    ld_event = lane_departure.update(output["lanes"]["offset_px"], timestamp)
    if ld_event:
        print("⚠️ EVENT:", ld_event)

    frame_idx += 1
"""

import cv2
import json
from core.pipeline import run_pipeline
from behaviour.tracker import ObjectTracker
from behaviour.tailgating import TailgatingDetector
from behaviour.lead_vehicle import select_lead_vehicle
from behaviour.lane_departure import LaneDepartureDetector

# Video path
VIDEO_PATH = "testing/test4.mp4"
OUTPUT_EVENTS_JSON = "events.json"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Initialize detectors
tracker = ObjectTracker()
tailgating = TailgatingDetector()
lane_departure = LaneDepartureDetector(offset_threshold=50)

frame_idx = 0
all_events = []  # List to store all events

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    frame_h = frame.shape[0]

    # Run detection pipeline
    output = run_pipeline(frame)

    # Update tracker
    tracked_objects = tracker.update(output["objects"])

    # Identify lead vehicle
    lead_vehicle = select_lead_vehicle(tracked_objects, output["lanes"])

    # Tailgating detection
    tg_event = tailgating.update(lead_vehicle, frame_h, timestamp)
    if tg_event:
        print("⚠️ EVENT:", tg_event)
        all_events.append(tg_event)

    # Lane departure detection
    ld_event = lane_departure.update(output["lanes"]["offset_px"], timestamp)
    if ld_event:
        # Convert numpy int to native Python int for JSON serialization
        ld_event["offset_px"] = int(ld_event["offset_px"])
        print("⚠️ EVENT:", ld_event)
        all_events.append(ld_event)

    frame_idx += 1

# Save all events to JSON
with open(OUTPUT_EVENTS_JSON, "w") as f:
    json.dump(all_events, f, indent=2)

print(f"\n✅ All events saved to {OUTPUT_EVENTS_JSON}")

cap.release()
cv2.destroyAllWindows()

