# Pneumonia Screening on Chest X-Rays

**Machine Learning Project Report**

*PneumoniaMNIST • CNN • ResNet-18 • Grad-CAM*

## Abstract

This project builds an educational binary classifier that screens pediatric chest X-rays for pneumonia using the PneumoniaMNIST dataset (MedMNIST v2). Models are compared against a trivial baseline: a small CNN trained from scratch, a pretrained ResNet-18, and a class-weighted ResNet-18 whose decision threshold is selected on the validation set. The final weighted ResNet-18 reaches 91.19% test accuracy, 98.97% sensitivity, 78.21% specificity, and 0.9771 ROC-AUC at a threshold of 0.80. Grad-CAM is used to inspect true-positive and false-positive predictions. The work is for learning purposes only and is not a clinical diagnostic system.

## 1. Introduction

The goal is to classify pediatric chest X-rays as normal or pneumonia. The project progresses from a simple CNN baseline to transfer learning, then to class-weighted training with a carefully chosen operating threshold. This progression shows not only how performance changes, but why different training and decision strategies trade sensitivity against specificity. The final model is analyzed with Grad-CAM to see which image regions drive its predictions.

## 2. Dataset

PneumoniaMNIST (MedMNIST v2) contains 5,856 pediatric chest X-ray images at 28×28 grayscale resolution, labeled 0 (Normal) or 1 (Pneumonia). The official splits were used directly, with no manual re-splitting. The validation set is used for model-selection decisions such as threshold choice, and the test set is reserved for final evaluation.

| **Split**  | **Images** | **Normal** | **Pneumonia** | **Pneumonia share** |
|------------|------------|------------|---------------|---------------------|
| Training   | 4,708      | 1,214      | 3,494         | 74.2%               |
| Validation | 524        | –          | –             | –                   |
| Test       | 624        | 234        | 390           | 62.5%               |

Training counts are from the data-exploration notebook; test counts follow from the confusion matrices. MedMNIST forms the validation set by a 9:1 split of the source training set, while the test set is a separate source split, so validation and test can differ in distribution.

## 3. Methods

### 3.1 Baseline CNN

A deliberately small CNN, so the full forward pass is easy to follow:

```text
Conv(1→16, 3×3) → ReLU → MaxPool(2) → Conv(16→32, 3×3) → ReLU → MaxPool(2) → Flatten → Linear(1568→64) → ReLU → Linear(64→2)
```

### 3.2 ResNet-18 Transfer Learning

A pretrained ResNet-18 reuses visual features learned during pretraining instead of learning every filter from random initialization. Its residual connections ease optimization, and it is lightweight compared with deeper networks. Two adaptations were needed:

- **Input:** the 1×28×28 images are resized to 224×224 and converted to 3 channels, matching what the network expects.
- **Classifier:** the final layer is replaced with a two-output layer: `nn.Linear(model.fc.in_features, 2)`.

### 3.3 Class Weighting and Threshold Selection

To address class imbalance, a fresh pretrained ResNet-18 was trained with class-weighted Cross-Entropy Loss. Weights follow n_samples / (2 × class_count), giving 1.939 for Normal and 0.674 for Pneumonia.

The default rule of predicting pneumonia when its probability is at least 0.5 is not necessarily optimal. Thresholds from 0.10 to 0.90 (step 0.05) were scanned on the validation set, and 0.80 was picked from that table, where sensitivity and specificity are nearly equal (96.92% and 97.04%). The test set was not used to tune the threshold.

### 3.4 Training Configuration

| **Model**          | **Loss**      | **LR** | **Batch** | **Epochs** | **Optim.** |
|--------------------|---------------|--------|-----------|------------|------------|
| Baseline CNN       | Cross-Entropy | 0.001  | 64        | 20         | Adam       |
| ResNet-18          | Cross-Entropy | 0.0001 | 32        | 10         | Adam       |
| Weighted ResNet-18 | Weighted CE   | 0.0001 | 32        | 10         | Adam       |

### 3.5 Evaluation Metrics

Models are evaluated on the test set with accuracy, sensitivity, specificity, the confusion matrix, and ROC-AUC. Sensitivity = TP / (TP + FN) is the share of pneumonia cases found, which matters most in a screening setting because false negatives are missed cases. Specificity = TN / (TN + FP) is the share of normal cases correctly cleared and reflects the false-positive burden. ROC-AUC measures how well the model ranks positives above negatives across all thresholds, whereas accuracy is computed at one threshold.

## 4. Results

All numbers are from the test set (624 images) and come from a single training run per model; no random seeds were fixed, so no error bars are available.

| **Model**                  | **Thr.** | **Accuracy** | **Sensitivity** | **Specificity** | **ROC-AUC** |
|----------------------------|----------|--------------|-----------------|-----------------|-------------|
| Trivial (always pneumonia) | –        | 62.50%       | 100.00%         | 0.00%           | 0.5000      |
| Baseline CNN               | 0.50     | 89.26%       | 96.41%          | 77.35%          | 0.9338      |
| ResNet-18                  | 0.50     | 84.62%       | 100.00%         | 58.97%          | 0.9639      |
| \+ class-weighted loss     | 0.50     | 88.30%       | 99.74%          | 69.23%          | 0.9771      |
| \+ threshold (final)       | 0.80     | 91.19%       | 98.97%          | 78.21%          | 0.9771      |

The trivial row is computed from the test class counts, not trained.

| **Confusion matrix**    | **TN** | **FP** | **FN** | **TP** |
|-------------------------|--------|--------|--------|--------|
| Baseline CNN            | 181    | 53     | 14     | 376    |
| ResNet-18               | 138    | 96     | 0      | 390    |
| \+ weighted loss (0.50) | 162    | 72     | 1      | 389    |
| \+ threshold 0.80       | 183    | 51     | 4      | 386    |

### 4.1 Discussion

**Baseline CNN.** A strong small baseline: 89.26% accuracy and 0.9338 ROC-AUC, with 96.41% sensitivity and 77.35% specificity.

**ResNet-18.** Pretraining raises ROC-AUC to 0.9639 and catches every pneumonia case, but at 0.5 it flags 96 normal images, dropping specificity to 58.97%.

**Class-weighted loss.** Weighting alone cuts false positives from 96 to 72 and lifts specificity to 69.23%, while only one pneumonia case is missed.

**Threshold 0.80.** Raising the threshold cuts false positives to 51 (specificity 78.21%) at the cost of three more missed cases (4 in total). ROC-AUC is unchanged because the model is the same, so the 0.9639 to 0.9771 gain comes from the weighted training run; with a single seed, run-to-run noise cannot be ruled out.

**Final model.** Weighted ResNet-18 at 0.80 gives the best accuracy, specificity and ROC-AUC of the models compared.

## 5. Grad-CAM and Error Analysis

Grad-CAM uses activations and gradients from `weighted_model.layer4[-1].conv2` to build a heatmap for the pneumonia class. It was run on the first five true positives (top row) and first five false positives (bottom row) at threshold 0.80.


(Figure: Grad-CAM examples, see plots/gradcam/)
**What the examples show.** In three of the five false positives (test images 28, 30, 44) the strongest activation sits at the top edge, above the lungs. In four of the five true positives and two of the five false positives the peak is at the same lower-right edge location. The same hot spot on correct and wrong predictions suggests the model may rely on a position or border cue, not only on lung appearance.

This is a small, hand-inspected sample, so it is a hypothesis, not proof. A heatmap shows which regions influenced a prediction, not that those regions contain pneumonia.

## 6. Limitations
- Single run per model, no fixed seeds, so there is no mean ± std.
- At 0.5 the weighted model has 94.81% specificity on validation but 69.23% on test. Validation comes from the training source and test from a separate one, so the 0.80 threshold may not transfer.
- The threshold was applied to the final model only; the CNN and plain ResNet-18 were scored at 0.5.
- No ImageNet mean/std normalization and no data augmentation were used.
- The 28×28 images carry far less detail than real X-rays; the dataset is small; there is no external validation.
- Grad-CAM does not establish causal or medically correct reasoning, and the model is not clinically validated.

## 7. Future Work
- Repeat training over at least three seeds and report mean ± std.
- Apply the same validation-chosen threshold to every model for a like-for-like comparison.
- Compare 28×28, 64×64 and 128×128 inputs, reporting ROC-AUC against training time.
- Add data augmentation and ImageNet normalization; check calibration; validate on an external dataset.

## 8. Conclusion

This project walks through a complete workflow for binary chest X-ray classification: data exploration, a CNN baseline, transfer learning, class weighting, validation-based threshold selection, and Grad-CAM analysis. The final weighted ResNet-18 achieves 91.19% accuracy, 98.97% sensitivity, 78.21% specificity, and 0.9771 ROC-AUC at a threshold of 0.80, the strongest result among the models compared.

## 9. References
- Yang, J. et al. (2023). MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10, 41.
- Kermany, D. S. et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5), 1122–1131.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. CVPR, 770–778.
- Selvaraju, R. R. et al. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV, 618–626.

**Educational disclaimer:** *This project is for educational and research practice only. It is not a medical device, does not provide medical advice, and must not be used to diagnose pneumonia or make patient-care decisions.*