import cv2
import os
import torch
import torchvision.transforms as T
from model import get_faster_rcnn_model  # your custom get_faster_rcnn_model(num_classes) function

# Adjust this dictionary as needed for your classes
CLASS_NAMES = {
    0: 'teacher',
    1: 'student'
}

def load_model(model_path, num_classes=3, device='cuda'):
    """
    Loads the model (Faster R-CNN) with the specified number of classes 
    and weights from model_path.
    """
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference_on_video(model, video_path, device='cuda', score_threshold=0.5, show_video=True):
    """
    Runs inference on a single video, draws bounding boxes and labels, and optionally shows video frames.
    Press 'q' to quit the video early.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Prepare transform to convert frames to tensors
    transform = T.ToTensor()  # simply converts numpy [H,W,C] -> tensor [C,H,W], scaled [0..1].
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB, then to tensor
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(rgb_frame).unsqueeze(0).to(device)  # shape: [1, 3, H, W]
        
        # Perform inference
        with torch.no_grad():
            prediction = model(inp)[0]
        
        # Extract boxes, scores, labels
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Loop through detected boxes and draw them
        for box, score, label in zip(boxes, scores, labels):
            if score < score_threshold:
                continue
            
            x1, y1, x2, y2 = box.astype(int)
            class_name = CLASS_NAMES.get(label, f"cls {label}")
            
            # Draw rectangle
            color = (0, 255, 0) if label == 0 else (0, 0, 255)  # green for teacher, red for student
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            
            # Put text (class name + score)
            text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1, cv2.LINE_AA)
        
        if show_video:
            cv2.imshow('Video Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if show_video:
        cv2.destroyAllWindows()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "fasterrcnn_model.pth"  # path to your trained weights
    
    # If you have exactly 2 classes + background = 3
    # (class 0=teacher, class 1=student, class 2=background in PyTorch's internal indexing)
    model = load_model(model_path, num_classes=3, device=device)
    
    # Path to the folder containing your videos
    video_folder = "exam_footages"
    
    # List all videos in that folder (assuming .mp4, .avi, etc.)
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    for vf in video_files:
        video_path = os.path.join(video_folder, vf)
        print(f"Processing video: {video_path}")
        run_inference_on_video(model, video_path, device=device, score_threshold=0.5, show_video=True)

if __name__ == "__main__":
    main()
