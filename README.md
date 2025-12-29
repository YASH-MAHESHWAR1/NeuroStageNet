# NeuroStageNet :An Explainable Deep Learning Framework for Alzheimer’s Disease Stage Classification from MRI Scans

## Abstract
This project builds a supervised deep learning pipeline to classify brain MRI images into **four Alzheimer’s disease stages**. The dataset was provided as **Parquet** files containing **binary-encoded MRI images** and labels. Images are decoded into **grayscale pixel arrays**, checked for class balance and resolution consistency, and used to train a **transfer-learning** model based on **EfficientNetB0**. To strengthen MRI-specific representation learning, a **Residual Block with Squeeze-and-Excitation (SE) channel attention** is added after the backbone. Training incorporates **data augmentation**, **label smoothing**, and **adaptive learning-rate scheduling**. Evaluation on an unseen test set uses **accuracy**, **per-class precision/recall/F1**, **confusion matrix**, and **multi-class AUC-ROC (OvR)**. The model achieves approximately **0.98 test accuracy** and **0.9975 AUC-ROC**, with most errors occurring between adjacent stages. **Grad-CAM** heatmaps provide interpretability by visualizing regions that influence predictions.

---

## 1. Introduction and Motivation
Alzheimer’s disease is a progressive neurodegenerative disorder where severity is associated with structural changes visible on MRI. Accurate staging supports early clinical attention, monitoring progression, and consistent decision-making. Manual interpretation can be slow and subject to inter-rater variation, especially between neighboring stages.  
The goal of this project is to build an automated system that learns discriminative MRI patterns, generalizes to unseen scans, and provides explainability so predictions can be inspected rather than treated as a black box.

---

## 2. Dataset and Data Handling

### 2.1 Dataset format
The dataset was provided as `train.parquet` and `test.parquet`. Each record includes:
- **Image** stored as binary bytes (sometimes wrapped inside a dictionary structure)
- **Label** in one of **four classes (0–3)**

Parquet is used due to efficient storage and fast column-based loading.

### 2.2 Decoding and preprocessing
Binary payloads are robustly extracted (handling dict-wrapped formats) and decoded with PIL into **grayscale** images, then converted to NumPy arrays. This ensures consistent conversion from storage format to model-ready tensors.

### 2.3 Exploratory checks (EDA)
Before training, the pipeline verifies:
- dataset shapes (sanity check),
- label distribution (imbalance awareness),
- image resolution statistics (min/mean/max) to justify fixed-size modeling.

---

## 3. Methodology

### 3.1 Data split
The training set is split into **train/validation (80/20)** using **stratification** to preserve class proportions and stabilize validation metrics.

### 3.2 Label encoding
Labels are converted to integers using `LabelEncoder` and then to **one-hot vectors** using `to_categorical` for multi-class learning.

### 3.3 Input pipeline (`tf.data`)
Images are stacked and expanded to include a channel dimension:  

$$(N, H, W) \rightarrow (N, H, W, 1)$$  
`tf.data` batching and prefetching improve training throughput.

### 3.4 Data augmentation
Augmentation is applied only to training data (flip, small rotation/zoom, contrast/brightness). This simulates realistic acquisition variability and reduces overfitting.

---

## 4. Model Architecture (Structure + Intuition)

### 4.1 Design philosophy
The model follows a standard medical-imaging strategy:
1. **Pretrained backbone** for strong feature extraction with limited data  
2. **Task-specific refinement** to adapt features to MRI staging cues  
3. **Regularized classifier head** to ensure generalization  

### 4.2 Architecture overview (layer flow)
- **Input:** $(128,128,1)$ grayscale MRI  
- **Channel projection:** `Conv2D(3, 3×3, same)`  
  - *Reason:* EfficientNet expects 3 channels; a learnable mapping is more flexible than replication.
- **Backbone:** `EfficientNetB0(include_top=False, ImageNet weights)`  
  - *Reason:* transfers robust hierarchical features; early layers capture general patterns.
  - *Partial fine-tuning:* `fine_tune_at=150` freezes early layers and trains later layers for MRI adaptation.
- **Residual refinement (filters=256) + SE attention:**  
  - 2× `SeparableConv2D` + BN + Swish + residual skip  
  - SE attention reweights channels using global context  
  - *Reason:* lightweight “adapter” that refines high-level backbone features; residuals stabilize learning; SE emphasizes informative channels.
- **Feature aggregation:** `GlobalAveragePooling` + `GlobalMaxPooling` → concatenate  
  - *Reason:* combines global distributed evidence and strongest localized evidence.
- **Classifier head:**  
  - Dense(1024) + BN + Dropout(0.5)  
  - Dense(512) + BN + Dropout(0.3) + L2  
  - Output Dense(4) + Softmax  
  - *Reason:* two-stage head learns non-linear class boundaries while regularization prevents overfitting.

---

## 5. Training Configuration
- **Optimizer:** Adam, $lr = 5 \times 10^{-4}$  
- **Loss:** Categorical cross-entropy with **label smoothing = 0.15**  
  - *Reason:* reduces overconfidence and improves robustness in borderline stage cases.
- **Callbacks:**  
  - EarlyStopping (monitor `val_auc`, restore best weights)  
  - ReduceLROnPlateau (monitor `val_loss`)  
  - ModelCheckpoint (save best by `val_auc`)

---

## 6. Evaluation Protocol and Results

### 6.1 Metrics
Evaluation uses:
- accuracy,
- per-class precision/recall/F1,
- confusion matrix,
- multi-class AUC-ROC (one-vs-rest).

### 6.2 Reported test performance
- **Accuracy:** $\approx 0.98$ (1280 samples)  
- **AUC-ROC (OvR):** $\approx 0.9975$  
- **Macro / Weighted F1:** $\approx 0.98$

The confusion matrix is strongly diagonal, with most errors between **adjacent stages (notably 2 vs 3)**, which is expected given gradual disease progression.

### 6.3 Why performance is strong
Performance is supported by:
- transfer learning initialization,
- controlled fine-tuning,
- residual + SE feature refinement,
- augmentation, label smoothing, and regularization,
- model selection based on validation AUC with adaptive LR scheduling.

---

## 7. Explainability (Grad-CAM)
Grad-CAM computes gradients of the predicted class score with respect to a chosen convolutional layer to generate a heatmap highlighting influential regions. Overlays support qualitative validation that predictions rely on meaningful spatial evidence.

---

## 8. Conclusion
This project delivers an end-to-end MRI-based Alzheimer stage classifier using Parquet decoding, EfficientNetB0 transfer learning with partial fine-tuning, residual SE refinement, and Grad-CAM interpretability. The model achieves **~0.98 accuracy** and **~0.9975 AUC-ROC** on the provided test set, with minimal confusion primarily between neighboring stages.  
**Limitations:** Results are specific to the provided dataset and labeling scheme, and very small class supports can make some per-class scores appear overly optimistic.
