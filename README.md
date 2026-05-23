# Design and Evaluation of a Deep Learning-Based Vehicle Detection and Classification System Using EfficientNet-B0 and YOLOv8
---

## Overview

This project implements a complete deep learning pipeline for automated vehicle detection and classification in real-world traffic imagery. Six vehicle categories are targeted — **bus, car, microbus, motorbike, pickup-van, and truck** — across a six-phase pipeline covering dataset engineering, model training, optimization, deployment, and critical analysis.

Two architectures are implemented and compared:
- **EfficientNet-B0** — transfer learning for patch-level vehicle classification
- **YOLOv8n** — single-stage object detection on full traffic scenes

| Metric | EfficientNet-B0 | YOLOv8n |
|---|---|---|
| Task | Classification | Object Detection |
| Accuracy / mAP@50 | **80.94%** | **72.0%** |
| Inference Time (GPU) | 36.8ms | 36.6ms |
| Parameters | 5.3M | 3.0M |
| Weakest Class | Microbus (F1: 0.407) | Motorbike (AP@50: 0.633) |

---

## Repository Structure

```
.
├── vehicle_project.ipynb               # Main notebook (all 6 phases)
├── report.docx                         # Final project report
└── vehicle_project/
    ├── efficientnet_best.pt            # Best EfficientNet-B0 checkpoint
    ├── efficientnet_history.json       # Per-epoch train/val loss and accuracy
    ├── benchmark_results.json          # GPU inference benchmark (Phase 3)
    ├── phase3_results.json             # YOLOv8 test evaluation metrics
    ├── phase4_results.json             # Optimizer experiment results (Phase 4)
    ├── efficientnet_curves.png         # Training loss & accuracy curves
    ├── efficientnet_confusion.png      # EfficientNet confusion matrix
    ├── class_distribution.png          # Dataset class imbalance visualization
    ├── augmentation_examples.png       # Augmentation pipeline examples
    ├── architecture_comparison.png     # EfficientNet vs YOLOv8 comparison chart
    ├── phase4_comparison.png           # Optimizer experiment comparison chart
    ├── phase5_analysis.png             # Inference latency & throughput analysis
    └── yolov8_vehicle-2/               # YOLOv8 training run output
        ├── weights/
        │   ├── best.pt                 # Best YOLOv8n checkpoint
        │   └── last.pt
        ├── results.csv                 # Per-epoch training metrics
        ├── results.png                 # Training curves (mAP, loss, P, R)
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── BoxP_curve.png
        ├── BoxR_curve.png
        ├── BoxF1_curve.png
        ├── BoxPR_curve.png
        ├── args.yaml                   # Training configuration
        ├── labels.jpg
        ├── train_batch{0,1,2}.jpg      # Training batch visualizations
        └── val_batch{0,1,2}_{labels,pred}.jpg
```

---

## Dataset

- **Source:** [Roboflow — Vehicle Detection Dataset v3](https://universe.roboflow.com/lynkeus03/vehicle-detection-by9xs/dataset/3) (CC BY 4.0)
- **Total images:** 9,211 annotated traffic images
- **Classes:** bus, car, microbus, motorbike, pickup-van, truck
- **Split:** 70% train (6,449) / 15% val (1,381) / 15% test (1,381) — fixed seed 42

The dataset is **not included** in this repository. Download it via Roboflow inside the notebook (API key required) or manually from the link above.

---

## Setup

### Requirements

```bash
pip install roboflow ultralytics torchvision gradio fastapi uvicorn python-multipart pillow
```

The notebook was developed and trained on **Google Colab** with a Tesla T4 GPU (CUDA 13.0, PyTorch 2.10.0+cu128). All phases can be reproduced by running the notebook top-to-bottom after mounting Google Drive.

### Pretrained Weights

| Model | File | Location |
|---|---|---|
| EfficientNet-B0 | `efficientnet_best.pt` | `vehicle_project/` |
| YOLOv8n (best) | `best.pt` | `vehicle_project/yolov8_vehicle-2/weights/` |

Both checkpoints are included in this repository. To load them:

```python
# EfficientNet-B0
import torch
from torchvision import models
import torch.nn as nn

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 6)
)
model.load_state_dict(torch.load('vehicle_project/efficientnet_best.pt'))
model.eval()

# YOLOv8n
from ultralytics import YOLO
yolo = YOLO('vehicle_project/yolov8_vehicle-2/weights/best.pt')
```

---

## Project Phases

### Phase 1 — Dataset Engineering
- Downloaded from Roboflow in YOLOv8 format
- 70/15/15 train/val/test split
- Augmentation pipeline: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomGrayscale`, `Normalize`
- Custom crop dataset for EfficientNet: 40,466 train / 8,611 val / 9,245 test crops

### Phase 2 — Baseline Model (EfficientNet-B0)
- ImageNet pretrained backbone, frozen; classifier head fine-tuned (7,686 trainable params)
- 10 epochs, AdamW (lr=1e-4), CosineAnnealingLR, CrossEntropyLoss, batch size 32
- **Best val accuracy: 80.94%** | Test accuracy: 79.8%

### Phase 3 — Advanced Architecture (YOLOv8n)
- Single-stage detector, 73 layers, 3.0M params, 8.1 GFLOPs
- Fine-tuned from COCO pretrained weights at 640×640
- 10 epochs, batch 16, early stopping patience 5
- **mAP@50: 0.720** | mAP@50-95: 0.406 | Training time: ~21.6 min

### Phase 4 — Training Optimization
Three optimizer configurations compared (1 epoch each):

| Config | Val Acc | Overfit Gap |
|---|---|---|
| Baseline: AdamW + Cosine + drop=0.3 | 80.94% | — |
| SGD + StepLR + drop=0.3 | **81.92%** | -7.93% |
| Adam + CosineAnnealing + drop=0.3 | 80.72% | -7.26% |
| AdamW + CosineAnnealing + drop=0.5 | 80.72% | **-9.74%** |

### Phase 5 — Deployment
- **Gradio** interface deployed on Google Colab with public shareable link
- Accepts a single traffic image; runs both models in parallel
- Outputs: YOLOv8 annotated bounding box image + EfficientNet top-1 class + confidence + latency

### Phase 6 — Engineering Analysis
Critical evaluation of failure cases, dataset bias (class imbalance, geographic and viewpoint bias), model limitations (no temporal reasoning, single-frame only), computational constraints, real-world deployment challenges, and ethical considerations (surveillance risk, fairness, accountability).

---

## Results

### EfficientNet-B0 — Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| bus | 0.839 | 0.764 | 0.800 | 998 |
| car | 0.785 | 0.906 | 0.841 | 4295 |
| microbus | 0.767 | 0.277 | 0.407 | 606 |
| motorbike | 0.837 | 0.907 | 0.871 | 2391 |
| pickup-van | 0.659 | 0.368 | 0.472 | 726 |
| truck | 0.722 | 0.533 | 0.613 | 229 |
| **weighted avg** | **0.792** | **0.798** | **0.781** | **9245** |

### YOLOv8n — Per-Class AP@50

| Class | AP@50 | Precision | Recall |
|---|---|---|---|
| bus | 0.865 | 0.740 | 0.856 |
| car | 0.725 | 0.620 | 0.725 |
| microbus | 0.698 | 0.641 | 0.679 |
| motorbike | 0.633 | 0.659 | 0.558 |
| pickup-van | 0.640 | 0.661 | 0.562 |
| truck | 0.760 | 0.725 | 0.668 |
| **all** | **0.720** | **0.674** | **0.675** |

---

## Inference Latency

Benchmarked on 100 test images (CPU mode, Phase 5):

| Metric | EfficientNet-B0 | YOLOv8n |
|---|---|---|
| Avg Latency | 77.3ms | 298.8ms |
| Throughput | 12.9 FPS | 3.3 FPS |
| GPU Latency (T4) | ~36.8ms | ~36.6ms |
| GPU Memory | ~84MB | ~170MB |

---

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Roboflow — Vehicle Detection Dataset v3)  
Code: MIT
