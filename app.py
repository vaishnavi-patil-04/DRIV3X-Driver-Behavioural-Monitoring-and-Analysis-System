import streamlit as st
import cv2
import json
import tempfile
import pandas as pd
import numpy as np

from core.pipeline import run_pipeline
from behaviour.tracker import ObjectTracker
from behaviour.tailgating import TailgatingDetector
from behaviour.lead_vehicle import select_lead_vehicle
from behaviour.lane_departure import LaneDepartureDetector
from behaviour.sign_violation import SignViolationDetector


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="DRIV3X",
    layout="wide"
)

# -------------------- TITLE & ABOUT --------------------
st.title("🚗 DRIV3X – Driver Behaviour Monitoring & Analysis System")

st.markdown("""
The main objectives of the Driver Behavioural Monitoring and Analysis System
are as follows:\n
• To develop asystem capable of detecting vehicles, pedestrians, and cyclists from
dashcam video feeds.\n
• To monitor driving behaviour and identify potential risks based on object distance,
reaction time, and environmental factors.\n
• To generate a post-drive risk score analysis report summarizing the driver’s
performance and safety level.\n
• To assist new drivers in improving their driving skills and provide reliable data to
insurance companies for behavioural analysis.\n

\nUpload a driving video to generate a behavioral safety report.
""")

st.divider()

# -------------------- VIDEO UPLOAD --------------------
uploaded_video = st.file_uploader(
    "📤 Upload a driving video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is None:
    st.info("Please upload a video to start analysis.")
    st.stop()

# Save uploaded video
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded_video.read())
    video_path = tmp.name

st.video(video_path)

# -------------------- DRIVING SCORE FUNCTION --------------------
def calculate_driving_score(events):
    base_score = 100

    severity_penalty = {
        "low": 1.0,
        "medium": 1.5,
        "high": 2.0
    }

    max_penalty = {
        "lane_departure": 15,
        "tailgating": 20,
        "sign_violation": 25
    }

    penalty_tracker = {
        "lane_departure": 0.0,
        "tailgating": 0.0,
        "sign_violation": 0.0
    }

    for e in events:
        event_type = e.get("event")
        severity = str(e.get("severity", "low")).lower()

        if event_type in penalty_tracker:
            penalty = severity_penalty.get(severity, 0.5)
            penalty_tracker[event_type] += penalty

    # Apply caps
    total_penalty = sum(
        min(penalty_tracker[k], max_penalty[k])
        for k in penalty_tracker
    )

    return round(max(base_score - total_penalty, 0), 1)


# -------------------- PROCESS BUTTON --------------------
if st.button("▶️ Analyze Video"):
    st.info("Processing video… please wait ⏳")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    tracker = ObjectTracker()
    tailgating = TailgatingDetector()
    lane_departure = LaneDepartureDetector(offset_threshold=50)
    sign_violation = SignViolationDetector()

    frame_idx = 0
    all_events = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        h, w = frame.shape[:2]

        output = run_pipeline(frame)
        tracked_objects = tracker.update(output["objects"])
        lead_vehicle = select_lead_vehicle(tracked_objects, output["lanes"])

        tg_event = tailgating.update(lead_vehicle, h, timestamp)
        if tg_event:
            all_events.append(tg_event)

        ld_event = lane_departure.update(
            output["lanes"]["offset_px"], timestamp
        )
        if ld_event:
            ld_event["offset_px"] = int(ld_event["offset_px"])
            all_events.append(ld_event)

        signs = output.get("signs", [])
        if signs:
            for s in signs:
                x1, y1, x2, y2 = s["bbox"]
                ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
                sign_violation.update(s["class"], ratio, s["confidence"])
        else:
            sign_violation.update("none", 0, 0)

        sign_violation.update_states()

        sv_event = sign_violation.check(0, timestamp)
        if sv_event:
            all_events.append(sv_event)

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()

    # -------------------- JSON SAFETY --------------------
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    all_events = make_json_safe(all_events)

    with open("events.json", "w") as f:
        json.dump(all_events, f, indent=2)

    st.success("✅ Analysis completed!")

    # -------------------- RESULTS --------------------
    if all_events:
        df = pd.DataFrame(all_events)

        driving_score = calculate_driving_score(all_events)

        st.subheader("🧮 Driving Safety Score")

        if driving_score >= 90:
            st.success(f"🏆 Score: {driving_score} / 100 (Excellent)")
        elif driving_score >= 75:
            st.info(f"✅ Score: {driving_score} / 100 (Good)")
        elif driving_score >= 60:
            st.warning(f"⚠️ Score: {driving_score} / 100 (Moderate Risk)")
        else:
            st.error(f"🚨 Score: {driving_score} / 100 (High Risk)")

        st.subheader("📊 Event Summary")
        summary = df["event"].value_counts().reset_index()
        summary.columns = ["Event Type", "Count"]
        st.table(summary)

        st.subheader("📋 Detailed Event Log")
        st.dataframe(df, use_container_width=True)

        with open("events.json", "r") as f:
            st.download_button(
                "📥 Download Full Report",
                f.read(),
                "events_report.json",
                "application/json"
            )
    else:
        st.info("No risky driving events detected.")
