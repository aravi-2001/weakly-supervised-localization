# Weakly Supervised Object Localization — WSDDN

Implements the **Weakly Supervised Deep Detection Network (WSDDN)** for object detection and localization using only image-level class labels — no bounding box annotations at training time.

---

## Overview

Standard object detection requires expensive bounding box annotations. WSDDN instead uses **weak supervision**: only a yes/no label per image class is given, and the network learns to localize objects by itself.

The model takes region proposals from the image, processes them through a shared CNN, and produces class-specific attention scores that implicitly learn which regions contain each object.

---

## Approach

### Architecture

Built on an **AlexNet** backbone pretrained on ImageNet:

```
Image
  ↓
AlexNet (conv1–conv5)
  ↓
ROI Pooling (per region proposal)
  ↓
FC6 → FC7 (shared)
  ↓        ↓
FC_cls   FC_det      (two separate branches)
  ↓        ↓
Softmax  Softmax
  ↓        ↓
  ×  (element-wise multiply)
  ↓
Sum over regions → class scores
```

### Dual-Branch Attention

The key insight is the multiplicative combination of two softmax outputs:
- **Classification stream** (`σ_cls`): scores each region for each class
- **Detection stream** (`σ_det`): scores each class across regions (which region is most likely to contain the object)

Their product, summed over all regions, gives the final image-level class score used for training with BCE loss.

### Region Proposals

Selective Search generates ~2000 candidate bounding boxes per image, which are fed as ROI proposals to the network.

### Dataset — Pascal VOC (20 Classes)

```
aeroplane  bicycle   bird      boat       bottle
bus        car       cat       chair      cow
diningtable  dog     horse     motorbike  person
pottedplant  sheep   sofa      train      tvmonitor
```

---

## File Structure

```
CODE/
├── wsddn.py          # WSDDN model (AlexNet + ROI pooling + dual-branch head)
├── AlexNet.py        # AlexNet backbone definition
├── voc_dataset.py    # Pascal VOC dataset loader with selective search proposals
├── task_1.py         # Training and evaluation — task 1
├── task_2.py         # Training and evaluation — task 2
├── test_plot.py      # Visualize predicted bounding boxes on test images
├── utils.py          # mAP, IoU, and other evaluation utilities
└── task_0.ipynb      # Setup and data exploration notebook
```

---

## How to Run

```bash
pip install torch torchvision numpy matplotlib

cd CODE

# Train and evaluate
python task_1.py
python task_2.py

# Visualize detections
python test_plot.py
```

---

## Key Concepts

| Concept | Details |
|---------|---------|
| Supervision level | Image-level labels only (no bounding boxes) |
| Backbone | AlexNet (pretrained on ImageNet) |
| Region proposals | Selective Search (~2000 proposals/image) |
| ROI pooling | Aligns variable-size proposals to fixed 6×6 feature maps |
| Loss | Binary Cross-Entropy over 20-class image labels |
| Evaluation | mAP at IoU = 0.5 |

---

## Dependencies

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- `selective_search` or OpenCV (for region proposals)
