# Using EfficientDET/test_efficientdet_videos.py
import cv2
import os
import torch
from train_efficientdet import create_model
from evaluate_metrics import box_iou  # if you need or not
import torchvision.transforms as T

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "efficientdet_d0.pth"
    
    # 1. Load model
    model = create_model(num_classes=2, compound_coef=0)  # same as training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. For each video in exam_footages
    video_dir = "exam_footages"
    video_files = [vf for vf in os.listdir(video_dir) if vf.lower().endswith(('.mp4','.avi','.mov'))]

    transform = T.ToTensor()

    # Class mapping: if offset=1 => class 1=teacher, class 2=student
    class_names = {
        1: "teacher",
        2: "student"
    }

    for vid_file in video_files:
        video_path = os.path.join(video_dir, vid_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video {video_path}")
            continue

        print(f"Running inference on: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            orig = frame.copy()

            # Convert BGR->RGB->Tensor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = transform(rgb).unsqueeze(0).to(device)  # [1,3,H,W]

            # Inference
            with torch.no_grad():
                preds = model(x, torch.jit.annotate(list, []))  # returns list[dict] with 'detections'
            
            detections = preds[0]['detections'].cpu().numpy()
            # shape [N,6]: [xmin,ymin,xmax,ymax,score,class]
            for det in detections:
                x1, y1, x2, y2, score, cls = det
                cls = int(cls)
                if score < 0.5 or cls == 0:
                    # skip background or low scores
                    continue
                cv2.rectangle(orig, (int(x1), int(int(y1))), (int(x2), int(y2)), (0,255,0), 2)
                class_name = class_names.get(cls, f"cls{cls}")
                text = f"{class_name}:{score:.2f}"
                cv2.putText(orig, text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            cv2.imshow("EfficientDet Inference", orig)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
