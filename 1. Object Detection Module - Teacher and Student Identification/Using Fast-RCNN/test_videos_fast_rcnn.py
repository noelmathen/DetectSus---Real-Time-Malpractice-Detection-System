# File: test_videos_fast_rcnn.py

import cv2
import os
import torch
import torchvision.transforms as T
import numpy as np

from get_fast_rcnn_model import get_fast_rcnn_model

# Suppose your classes are:
# background=0, teacher=1, student=2
CLASS_NAMES = {
    1: "teacher",
    2: "student"
}

def load_fast_rcnn(weights_path, num_classes=3, device="cuda"):
    model = get_fast_rcnn_model(num_classes)
    model.load_state_dict(torch.load("fast_rcnn_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def run_inference_on_video(model, video_path, output_path, device="cuda", score_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    transform = T.ToTensor()  # convert frame to tensor [C,H,W], in [0..1]

    frame_count = 0  # Counter for processed frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(rgb_frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(inp)[0]  # first item in batch

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            if label == 0:
                # background
                continue

            x1, y1, x2, y2 = box.astype(int)
            class_name = CLASS_NAMES.get(label, f"cls {label}")
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # teacher=green, student=red

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        out.write(frame)

        # Print progress
        percentage_complete = (frame_count / total_frames) * 100
        print(f"\rProcessing frame {frame_count}/{total_frames} ({percentage_complete:.2f}%)", end="")

    cap.release()
    out.release()
    print(f"\nSaved annotated video to {output_path}")
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_fast_rcnn("fast_rcnn_model.pth", num_classes=3, device=device)

    video_folder = "exam_footages"
    output_folder = "output_videos_fast_rcnn"
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    for vf in video_files:
        video_path = os.path.join(video_folder, vf)
        output_path = os.path.join(output_folder, f"processed_{vf}")
        print(f"Processing {video_path} -> {output_path}")
        run_inference_on_video(model, video_path, output_path, device=device)

if __name__ == "__main__":
    main()
