Based on your project details and the repository structure, here's a suggested **README.md** for your GitHub repository:

---

# DetectSus - Real-Time Malpractice Detection System

DetectSus is a computer vision-based project designed to enhance examination integrity by leveraging object detection and behavior analysis. The system aims to detect suspicious behaviors and activities in classroom exam settings, using advanced machine learning models for real-time monitoring and alerting.

---

## Features

- **Multi-Model Object Detection**: Implements YOLOv8, Faster R-CNN, and SSD for detecting teachers and students in examination environments.
- **Real-Time Monitoring**: Tracks objects, gestures, and interactions to identify suspicious behaviors.
- **Dataset Support**: Utilizes a custom annotated dataset combining teacher and student behaviors in exam scenarios.
- **Comprehensive Evaluation**: Provides precision, recall, F1-score, and other metrics to compare model performances.
- **Scalable and Modular**: Designed to work across diverse classroom environments with easy model switching.

---

## Repository Structure

```
DetectSus - Real Time Malpractice Detection System/
├── .gitignore
├── Object Detection Module - Teacher and Student Identification/
│   ├── Using Fast R-CNN/
│   │   ├── convert_yolo_to_frcnn.py
│   │   ├── dataset_frcnn.py
│   │   ├── train_frcnn.py
│   │   ├── evaluate.py
│   │   ├── test_videos.py
│   │   └── ...
│   ├── Using SSD/
│   │   ├── get_ssd_model.py
│   │   ├── dataset_ssd.py
│   │   ├── train_ssd.py
│   │   ├── evaluate.py
│   │   ├── test_videos_ssd.py
│   │   └── ...
│   ├── Using YOLOv8/
│   │   ├── best.pt
│   │   ├── train_yolov8.py
│   │   ├── evaluate.py
│   │   ├── test_videos_yolov8.py
│   │   └── ...
│   └── runs/
│       └── detect/
│           ├── exp/
│           │   ├── labels/
│           │   ├── images/
│           │   └── weights/
├── README.md
├── requirements.txt
└── .vscode/
```

---

## Models Implemented

### 1. **YOLOv8**
- **Purpose**: Fast, real-time object detection.
- **Training**: `train_yolov8.py`
- **Testing**: `test_videos_yolov8.py`
- **Evaluation**: `evaluate.py`

### 2. **Faster R-CNN**
- **Purpose**: High-accuracy object detection, suitable for detailed analysis.
- **Training**: `train_frcnn.py`
- **Testing**: `test_videos.py`
- **Evaluation**: `evaluate.py`

### 3. **SSD (Single Shot MultiBox Detector)**
- **Purpose**: Lightweight and balanced object detection.
- **Training**: `train_ssd.py`
- **Testing**: `test_videos_ssd.py`
- **Evaluation**: `evaluate.py`

---

## Dataset

The repository uses a custom combined dataset for teachers and students:
- **Structure**:
  ```
  combined_dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── annotations/
  ├── train_anns.json
  └── val_anns.json
  ```
- **Annotations**: Converted to suit YOLO, SSD, and Faster R-CNN formats.
- **Preprocessing**: Scripts for dataset conversion and augmentation provided.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/noelmathen/DetectSus---Real-Time-Malpractice-Detection-System.git
   cd DetectSus---Real-Time-Malpractice-Detection-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that dataset is placed in the `combined_dataset/` directory.

---

## Usage

### Training
- **YOLOv8**:
  ```bash
  python train_yolov8.py
  ```
- **Faster R-CNN**:
  ```bash
  python train_frcnn.py
  ```
- **SSD**:
  ```bash
  python train_ssd.py
  ```

### Testing
- Use test scripts in respective directories to evaluate models on video footage.

---

## Evaluation Metrics

For each model, the following metrics are calculated:
- **Precision**
- **Recall**
- **F1-Score**
- **Inference Time**

Run the evaluation script:
```bash
python evaluate.py
```

---

## Contributing

Contributions are welcome! Please submit issues or pull requests for bug fixes or feature additions.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Guide**: Dr. Divya James
- **Team Members**: Allen Prince, Dea Elizabeth Varghese, Noel Mathen Eldho, Shruti Maria Shibu
- **Institution**: Rajagiri School of Engineering & Technology

---

Let me know if you need additional details or modifications!
