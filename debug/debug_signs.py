import cv2
from vision.signs import detect_signs

# Use 0 for default webcam, 1 or 2 if you have multiple cameras
video_path = r"testing/test4.mp4"   # raw string
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_signs(frame)
    print(detections)  # debug print
 
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
