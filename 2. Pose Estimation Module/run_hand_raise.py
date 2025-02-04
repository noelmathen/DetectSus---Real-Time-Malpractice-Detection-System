from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")  # Use yolov8s-pose.pt for better accuracy

# Load video file
video_path = "Top - Corner multiple scenes.mp4"
cap = cv2.VideoCapture(video_path)

# Set resolution
frame_width, frame_height = 1280, 720

# Save output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

def is_turning_back(keypoints):
    """
    Detect if a student is turning back based on keypoint positions.
    """
    if keypoints is None or len(keypoints) < 7:
        return False  # Not enough keypoints detected

    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder = keypoints[:7]

    if nose is None or left_eye is None or right_eye is None or left_ear is None or right_ear is None:
        return False  # Missing keypoints

    # Calculate horizontal distances
    eye_dist = abs(left_eye[0] - right_eye[0])
    shoulder_dist = abs(left_shoulder[0] - right_shoulder[0])
    
    # If the eye distance is small but both ears are visible, the head is turned
    if eye_dist < 0.4 * shoulder_dist and left_ear[0] > left_eye[0] and right_ear[0] < right_eye[0]:
        return True

    return False

def is_hand_raised(keypoints):
    """
    Detect if a student is raising their hand based on keypoint positions.
    """
    if keypoints is None or len(keypoints) < 11:
        return False
    
    left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist = keypoints[5:11]
    
    #if None in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]:
    #    return False
    
    if left_shoulder is None or right_shoulder is None or left_elbow is None or right_elbow is None or left_wrist is None or right_wrist is None:
        return False 
    
    threshold = min(left_shoulder[1], right_shoulder[1]) + 30  # Shoulder height
    #print("Threhsold:",threshold)
    #print("left_wrist:",left_wrist[1])
    #print("right_wrist:",right_wrist[1])
    if left_wrist[1] < threshold  or right_wrist[1] < threshold:  # If either wrist is above shoulders
        #print("hand raise")
        return True
    
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (frame_width, frame_height))
    results = model(frame)
    
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

        for kp in keypoints:
            turning_back = False
            hand_raised = False
            if is_turning_back(kp):
                cv2.putText(frame, "Turning Back!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for x, y in kp[:6]:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                turning_back = True
            if is_hand_raised(kp):
                cv2.putText(frame, "Hand Raised!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                for x, y in kp[6:11]:
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                hand_raised = True
            if not turning_back:
                for x, y in kp[:6]:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            if not hand_raised:
                for x, y in kp[6:11]:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            for x, y in kp[11:]:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    out.write(frame)
    cv2.imshow("Exam Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
