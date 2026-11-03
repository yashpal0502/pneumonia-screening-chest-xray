# Pneumonia Screening on Chest X-Rays

**Machine Learning Project Report**

*PneumoniaMNIST • CNN • ResNet-18 • Grad-CAM*

## Abstract

This project builds an educational binary classifier that screens pediatric chest X-rays for pneumonia using the PneumoniaMNIST dataset (MedMNIST v2). Three models are compared: a small CNN trained from scratch, a pretrained ResNet-18, and a class-weighted ResNet-18 whose decision threshold is selected on the validation set. The final weighted ResNet-18 reaches 91.19% test accuracy, 98.97% sensitivity, 78.21% specificity, and 0.9771 ROC-AUC at a threshold of 0.80. Grad-CAM is used to inspect true-positive and false-positive predictions. The work is for learning purposes only and is not a clinical diagnostic system.

## 1. Introduction

The goal is to classify pediatric chest X-rays as normal or pneumonia. The project progresses from a simple CNN baseline to transfer learning, then to class-weighted training with a carefully chosen operating threshold. This progression shows not only how performance changes, but why different training and decision strategies trade sensitivity against specificity. The final model is analyzed with Grad-CAM to see which image regions drive its predictions.

## 2. Dataset

PneumoniaMNIST contains 5,856 pediatric chest X-ray images at 28×28 grayscale resolution, labeled 0 (Normal) or 1 (Pneumonia). The official splits were used directly, with no manual re-splitting. The validation set is used for model-selection decisions such as threshold choice, and the test set is reserved for final evaluation.

| **Split**  | **Images** | **Use**             |
|------------|------------|---------------------|
| Training   | 4,708      | Model fitting       |
| Validation | 524        | Threshold selection |
| Test       | 624        | Final evaluation    |
| Total      | 5,856      |                     |

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

To address class imbalance, a fresh pretrained ResNet-18 was trained with class-weighted Cross-Entropy Loss, which gives the less-represented class more influence on the training loss.

The default rule of predicting pneumonia when its probability is at least 0.5 is not necessarily optimal. Several thresholds were therefore evaluated on the validation set, and 0.80 was selected. The test set was not used to tune the threshold.

### 3.4 Training Configuration

| **Model**          | **Loss**      | **LR** | **Batch** | **Epochs** | **Optim.** |
|--------------------|---------------|--------|-----------|------------|------------|
| Baseline CNN       | Cross-Entropy | 0.001  | 64        | 20         | Adam       |
| ResNet-18          | Cross-Entropy | 0.0001 | 32        | 10         | Adam       |
| Weighted ResNet-18 | Weighted CE   | 0.0001 | 32        | 10         | Adam       |

### 3.5 Evaluation Metrics

Models are evaluated on the test set with accuracy, sensitivity, specificity, the confusion matrix, and ROC-AUC. Sensitivity = TP / (TP + FN) is the share of pneumonia cases found, which matters most in a screening setting because false negatives are missed cases. Specificity = TN / (TN + FP) is the share of normal cases correctly cleared and reflects the false-positive burden. ROC-AUC measures how well the model ranks positives above negatives across all thresholds, whereas accuracy is computed at one threshold.

## 4. Results

| **Test metric** | **Baseline CNN** | **ResNet-18** | **Weighted ResNet-18** |
|-----------------|------------------|---------------|------------------------|
| Accuracy        | 89.26%           | 84.62%        | **91.19%**             |
| Sensitivity     | 96.41%           | 100.00%       | **98.97%**             |
| Specificity     | 77.35%           | 58.97%        | **78.21%**             |
| ROC-AUC         | 0.9338           | 0.9639        | **0.9771**             |
| Threshold       | 0.50             | 0.50          | **0.80**               |

| **Confusion matrix** | **TN** | **FP** | **FN** | **TP** |
|----------------------|--------|--------|--------|--------|
| Baseline CNN         | 181    | 53     | 14     | 376    |
| ResNet-18            | 138    | 96     | 0      | 390    |
| Weighted ResNet-18   | 183    | 51     | 4      | 386    |

### 4.1 Discussion

**Baseline CNN.** The small CNN is a strong baseline at 89.26% accuracy and 0.9338 ROC-AUC, with 96.41% sensitivity and 77.35% specificity.

**ResNet-18.** Transfer learning raises ROC-AUC to 0.9639 and catches every pneumonia case, but at the default threshold it produces 96 false positives, dropping specificity to 58.97% and accuracy to 84.62%. Good ranking ability does not guarantee a good operating point, which is why accuracy alone can hide important differences between models.

**Weighted ResNet-18.** Class weighting with the validation-selected 0.80 threshold gives the best overall result: ROC-AUC 0.9771, 91.19% accuracy, and 78.21% specificity, with only 4 false negatives and false positives cut to 51. It is therefore selected as the final model.

## 5. Grad-CAM Explainability

Grad-CAM uses activations and gradients from a convolutional layer to produce a spatial heatmap of the regions that most influenced a prediction. The target layer is `weighted_model.layer4[-1].conv2`, a late layer that keeps spatial structure while encoding high-level features.

The notebook identifies test cases at the 0.80 threshold: true positives (actual pneumonia, predicted pneumonia) and false positives (actual normal, predicted pneumonia). Heatmaps are computed for the pneumonia class and overlaid on the X-ray, then inspected:

- For true positives, check whether attention concentrates on plausible chest and lung regions.
- For false positives, check whether the model is responding to borders, positioning, artifacts, or other irrelevant patterns.

A heatmap shows which regions influenced a prediction, not that those regions contain pneumonia. It is an explainability aid, not a clinical validation method.

## 6. Limitations
- The 28×28 images carry far less detail than real clinical chest X-rays.
- The dataset is small, and no external dataset was used for independent validation.
- The 0.80 threshold is specific to this validation set. ResNet validation accuracy (97.71%) was much higher than test accuracy (84.62%), suggesting the validation set may not match the test distribution well.
- The effects of class weighting and threshold selection were not evaluated separately.
- Grad-CAM does not establish causal or medically correct reasoning.
- The model has not been clinically validated, and PneumoniaMNIST performance is not clinical diagnostic performance.

## 7. Future Work
- Compare 28×28, 64×64, and 128×128 inputs, reporting ROC-AUC against training time.
- Add data augmentation and systematic hyperparameter tuning.
- Evaluate additional architectures and larger Grad-CAM error-analysis sets.
- Assess probability calibration and validate on an independent external dataset.

## 8. Conclusion

This project walks through a complete workflow for binary chest X-ray classification: data exploration, a CNN baseline, transfer learning, class weighting, validation-based threshold selection, and Grad-CAM analysis. The final weighted ResNet-18 achieves 91.19% accuracy, 98.97% sensitivity, 78.21% specificity, and 0.9771 ROC-AUC at a threshold of 0.80, the strongest result among the three models tested.

**Educational disclaimer:** *This project is for educational and research practice only. It is not a medical device, does not provide medical advice, and must not be used to diagnose pneumonia or make patient-care decisions.*
