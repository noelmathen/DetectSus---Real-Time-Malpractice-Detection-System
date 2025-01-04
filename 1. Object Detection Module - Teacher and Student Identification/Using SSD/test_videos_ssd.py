import cv2
import os
import torch
import torchvision.transforms as T

from get_ssd_model import get_ssd_model

# If you used the label shifting approach:
CLASS_NAMES = {
    1: "teacher",
    2: "student"
}

def load_ssd_model(weights_path, num_classes=3, device="cuda"):
    model = get_ssd_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference_on_video(model, video_path, output_path, device="cuda", score_threshold=0.5):
    """
    Runs inference on a single video, writes output to a new video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Prepare VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    transform = T.ToTensor()  # For converting frames to [C,H,W] float in [0..1]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(rgb_frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(inp)[0]  # first (and only) element in the batch
            
        boxes = preds['boxes'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            
            # SSD background label = 0, so skip that
            if label == 0:
                continue
            
            x1, y1, x2, y2 = box.astype(int)
            class_name = CLASS_NAMES.get(label, f"cls {label}")
            
            # Draw bounding box
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            
            text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Write the frame to the output video
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "ssd_model.pth"  # your trained SSD weights
    model = load_ssd_model(model_path, num_classes=3, device=device)
    
    video_folder = "exam_footages"
    video_files = [vf for vf in os.listdir(video_folder) if vf.endswith(('.mp4', '.avi', '.mov'))]
    
    for vf in video_files:
        video_path = os.path.join(video_folder, vf)
        output_path = os.path.join("output_videos", f"processed_{vf}")
        os.makedirs("output_videos", exist_ok=True)
        print(f"Processing {video_path} -> {output_path}")
        run_inference_on_video(model, video_path, output_path, device=device)

if __name__ == "__main__":
    main()
