# Pneumonia Screening on Chest X-Rays

Binary pneumonia screening from pediatric chest X-ray images using **PneumoniaMNIST** (MedMNIST v2). The project compares a small CNN trained from scratch, a pretrained ResNet-18, and a class-weighted ResNet-18 with a validation-selected threshold, and uses Grad-CAM to inspect the final model's predictions.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or patient-care decisions.

## Results

All numbers are on the official test set (624 images).

| Metric      | Baseline CNN |   ResNet-18 | Weighted ResNet-18 (final) |
| ----------- | -----------: | ----------: | -------------------------: |
| Accuracy    |       89.26% |      84.62% |                 **91.19%** |
| Sensitivity |       96.41% | **100.00%** |                     98.97% |
| Specificity |       77.35% |      58.97% |                 **78.21%** |
| ROC-AUC     |       0.9338 |      0.9639 |                 **0.9771** |
| Threshold   |         0.50 |        0.50 |                       0.80 |

Confusion matrices (rows = actual, columns = predicted; order Normal, Pneumonia):

| Model              |  TN |  FP |  FN |  TP |
| ------------------ | --: | --: | --: | --: |
| Baseline CNN       | 181 |  53 |  14 | 376 |
| ResNet-18          | 138 |  96 |   0 | 390 |
| Weighted ResNet-18 | 183 |  51 |   4 | 386 |

The weighted ResNet-18 is the final model: it has the best accuracy, specificity, and ROC-AUC, while missing only 4 pneumonia cases. The unweighted ResNet-18 catches every pneumonia case at the default threshold but produces 96 false positives.

## Dataset

PneumoniaMNIST contains 5,856 pediatric chest X-rays at **28×28 grayscale**, labeled `0` (Normal) or `1` (Pneumonia). The official splits are used as provided, with no re-splitting:

| Split      | Images |
| ---------- | -----: |
| Train      |  4,708 |
| Validation |    524 |
| Test       |    624 |

## Models

| Model              | Loss                   | Optimizer |   LR | Batch | Epochs |
| ------------------ | ---------------------- | --------- | ---: | ----: | -----: |
| Baseline CNN       | Cross-Entropy          | Adam      | 1e-3 |    64 |     20 |
| ResNet-18          | Cross-Entropy          | Adam      | 1e-4 |    32 |     10 |
| Weighted ResNet-18 | Weighted Cross-Entropy | Adam      | 1e-4 |    32 |     10 |

**Baseline CNN:** `Conv(1→16) → ReLU → MaxPool → Conv(16→32) → ReLU → MaxPool → Flatten → Linear(1568→64) → ReLU → Linear(64→2)`, with 3×3 kernels and padding 1.

**ResNet-18:** pretrained weights, with the final layer replaced by `nn.Linear(model.fc.in_features, 2)`. Inputs are resized from 28×28 to 224×224 and converted from 1 to 3 channels.

**Weighted ResNet-18:** a fresh pretrained ResNet-18 trained with class-weighted loss to handle class imbalance. The decision threshold (**0.80**) was chosen by evaluating several thresholds on the **validation set only**; the test set was never used for tuning.

## Grad-CAM

Grad-CAM heatmaps are generated from `weighted_model.layer4[-1].conv2` for the pneumonia class. The notebook selects true-positive and false-positive test cases at the 0.80 threshold and overlays the heatmaps on the X-rays, to check whether attention falls on plausible chest regions or on borders and artifacts. Grad-CAM is an explainability aid, not evidence of medically correct reasoning.

## Repository Structure

```text
pneumonia-screening-chest-xray/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # load data, class distribution, sample images
│   ├── 02_cnn_baseline.ipynb              # baseline CNN training and evaluation
│   ├── 03_resnet18_transfer_learning.ipynb # ResNet-18, weighted loss, threshold selection
│   └── 04_gradcam.ipynb                   # Grad-CAM on TP / FP test cases
├── plots/                                 # figures (ROC curve, class distribution, Grad-CAM)
├── results/                               # saved model weights (.pth)
├── README.md/
├── Report.md/
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt
```

Run the notebooks in numerical order. Notebook 04 loads the weights saved by notebook 03.

## Limitations

- Images are only 28×28, far less detail than real clinical X-rays.
- The dataset is small and there is no external validation set.
- The threshold of 0.80 is specific to this validation set; validation accuracy was much higher than test accuracy for the ResNet models (97.71% vs 84.62%), so the validation set may not match the test distribution well.
- The gains from class weighting and from threshold selection were not evaluated separately.
- Grad-CAM does not establish causal or clinically correct reasoning.

## Future Work

- Compare 28×28, 64×64, and 128×128 inputs (ROC-AUC vs training time).
- Add data augmentation and systematic hyperparameter tuning.
- Try additional architectures and analyze more Grad-CAM error cases.
- Evaluate probability calibration and validate on an external dataset.

## Educational Disclaimer

This project is for educational and research practice only. It is not a medical device, does not provide medical advice, and must not be used to diagnose pneumonia or make patient-care decisions.
