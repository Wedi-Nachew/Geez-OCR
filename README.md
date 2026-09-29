# Ge'ez OCR

A deep-learning classification project for recognizing handwritten Ge'ez (ግዕዝ) characters. The project experiments with multiple ANN and CNN architectures, selects the most promising model through stratified cross-validation, and trains it for final evaluation.

## Overview

Ge'ez is an ancient South Semitic script originating from the Horn of Africa (northern Ethiopia and southern Eritrea). It is the historical root of modern languages such as Amharic, Tigrinya, and Tigré, and preserves centuries of historical chronicles, philosophical works, traditional medicine, and religious texts including the Ge'ez Bible.

The Ge'ez script (Fidel) is an abugida in which each symbol represents a consonant–vowel combination. It consists of 34 base consonants, each expanded into 7 vowel forms (orders), plus a set of irregular characters whose shapes do not follow the standard modification patterns. This structural complexity — 287 visually distinct characters, many of which differ only by small strokes or attachments — makes optical character recognition particularly challenging.

This project trains and evaluates deep-learning models to classify isolated Ge'ez character images across all 287 classes.

## Objectives

1. Load and preprocess a Ge'ez character image dataset for classification.
2. Implement and compare multiple ANN and CNN architectures of increasing complexity.
3. Use stratified 5-fold cross-validation to estimate generalization performance.
4. Select the model with the highest mean cross-validation accuracy.
5. Train, evaluate, and save the selected model.
6. Analyze errors, misclassifications, and the performance difference between regular and irregular fidels.

## Dataset

The project uses the [Yaredoffice/geez-characters](https://huggingface.co/datasets/Yaredoffice/geez-characters) dataset, downloaded locally and organized into a directory structure for label inference.

| Property              | Value                          |
| --------------------- | ------------------------------ |
| Total classes          | 287                            |
| Training images        | 13,776 (48 per class)          |
| Test images            | 1,435 (5 per class)            |
| Image dimensions       | 32 × 32 × 1 (grayscale)       |
| Label mode             | Integer (inferred from folder) |

The dataset is organized as:

```
dataset/
├── train/
│   ├── 0/    (48 images)
│   ├── 1/    (48 images)
│   └── ...   (287 class folders)
└── test/
    ├── 0/    (5 images)
    ├── 1/    (5 images)
    └── ...   (287 class folders)
```

Folder names are numeric (0–286). The mapping from folder index to Ge'ez character is defined in `constants.py`, which lists all 287 characters organized by their 7 vowel orders (34 consonants × 7 orders = 238 regular characters) plus 49 irregular characters.

Images are loaded using `tf.keras.utils.image_dataset_from_directory` with explicit numeric `class_names` ordering to prevent alphabetical sorting issues (e.g., "0", "1", "10" …). Loading is done with `shuffle=False` and a large batch size (1024) for fast extraction into NumPy arrays.

## Project Structure

```
geez_ocr/
├── dataset/
│   ├── train/              # 287 subdirectories, 48 images each
│   └── test/               # 287 subdirectories, 5 images each
├── models/
│   ├── ann_models.py       # 3 ANN model definitions
│   └── cnn_models.py       # 3 CNN model definitions
├── constants.py            # GEEZ_CHARACTERS list (287 characters)
├── utils.py                # Data loading, cross-validation, visualization utilities
├── model_selection.ipynb   # Cross-validation of all 6 models
├── geez_ocr.ipynb          # Final model training, evaluation, and analysis
├── final_geez_cnn.keras    # Saved final trained model
├── .gitignore
└── README.md
```

### Key files

- **`constants.py`** — Defines the `GEEZ_CHARACTERS` list mapping integer indices to Ge'ez characters. Characters are organized into 7 vowel orders of 34 consonants each (238 regular fidels) followed by 49 irregular fidels.
- **`utils.py`** — Contains dataset loading (`load_data`), stratified 5-fold cross-validation (`cross_validate_model`), results summarization (`summarize_results`), and image visualization utilities (`show_images`, `show_images_with_prediction`, `show_misclassifications`).
- **`models/ann_models.py`** — Defines three ANN architectures: `build_simpler_ann`, `build_deeper_ann`, `build_regularized_deeper_ann`.
- **`models/cnn_models.py`** — Defines three CNN architectures: `build_simpler_cnn`, `build_deeper_cnn`, `build_regularized_deeper_cnn`.
- **`model_selection.ipynb`** — Trains and cross-validates all six models, then compares their performance.
- **`geez_ocr.ipynb`** — Further cross-validates the selected CNN with enhanced callbacks, trains it on the full training set, evaluates on the test set, and performs error analysis.
- **`final_geez_cnn.keras`** — The saved final trained model (~6.3 MB).

## Methodology

The project follows this workflow:

1. **Data loading** — Images are loaded from the `dataset/` directory using TensorFlow's `image_dataset_from_directory`, converted to grayscale NumPy arrays of shape `(32, 32, 1)`.

2. **Preprocessing** — All models include a `Rescaling` layer that normalizes pixel values from `[0, 255]` to `[-1.0, 1.0]` using `scale=1/127.5, offset=-1.0`. This normalization is embedded in the model graph rather than applied separately.

3. **Model experimentation** — Six architectures (3 ANN, 3 CNN) are defined with increasing complexity and regularization.

4. **Stratified 5-fold cross-validation** — Each model is evaluated using `StratifiedKFold` (5 splits, `shuffle=True`, `random_state=42`). For each fold, the model is rebuilt from scratch, trained, and evaluated on both the training fold and the validation fold. Metrics recorded per fold: accuracy, top-5 accuracy, and train–CV gap.

5. **Model comparison** — The mean and standard deviation of training accuracy, CV accuracy, and gap are computed across the 5 folds for each model.

6. **Model selection** — The regularized deeper CNN achieved the highest mean cross-validation accuracy among the six evaluated models and was selected for further training.

7. **Further cross-validation** — The selected model is cross-validated again in the final notebook with enhanced callbacks (early stopping monitoring `val_accuracy` with `patience=8`, and learning-rate reduction on plateau).

8. **Final training** — The training data is split 90/10 (stratified, `random_state=42`) into development and validation sets. The model is trained for up to 100 epochs with early stopping and learning-rate reduction.

9. **Evaluation** — The trained model is evaluated on the development set, validation set, full training set, and the held-out test set. Classification reports and visualizations are generated.

10. **Error analysis** — Misclassified examples are visualized. Regular and irregular fidels are evaluated separately on the training set.

## Models

### ANN Models

#### Simpler ANN (`Basic_ANN`)

A straightforward fully connected network:

- **Input:** 32 × 32 × 1
- **Rescaling** to [-1, 1]
- **Flatten** → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(287, linear)
- **Optimizer:** Adam (lr=0.0005)
- **Parameters:** 726,047

#### Deeper ANN (`Deeper_ANN`)

A deeper network with LayerNormalization and He Normal initialization:

- **Input:** 32 × 32 × 1
- **Rescaling** → **Flatten**
- Dense(1024) → LayerNorm → ReLU → Dense(512) → LayerNorm → ReLU → Dense(256) → LayerNorm → ReLU → Dense(128) → LayerNorm → ReLU → Dense(287)
- **Optimizer:** Adam (lr=0.0005)
- **Parameters:** 1,779,487

#### Regularized Deeper ANN (`Regularized_Deeper_ANN`)

The deeper ANN with added Dropout:

- Same structure as the Deeper ANN, with Dropout inserted after specific layers:
  - Dense(1024) → LayerNorm → ReLU
  - Dense(512) → LayerNorm → ReLU → **Dropout(0.2)**
  - Dense(256) → LayerNorm → ReLU → **Dropout(0.3)**
  - Dense(128) → LayerNorm → ReLU → **Dropout(0.3)**
  - Dense(287, linear)
- **Optimizer:** Adam (lr=0.0005)
- **Parameters:** 1,779,487

### CNN Models

#### Simpler CNN (`Basic_CNN`)

A minimal CNN with one convolutional block:

- **Input:** 32 × 32 × 1
- **Rescaling** → Conv2D(64, 3×3, ReLU, same) → MaxPool(2×2) → Flatten → Dense(128, ReLU) → Dense(287, linear)
- **Optimizer:** Adam (lr=0.0005)

#### Deeper CNN (`Deeper_CNN_Model`)

Three convolutional blocks without regularization:

- **Input:** 32 × 32 × 1
- **Rescaling**
- Conv2D(64, 3×3, ReLU, same) → MaxPool(2×2)
- Conv2D(64, 3×3, ReLU, same) → MaxPool(2×2)
- Conv2D(32, 3×3, ReLU, same) → MaxPool(2×2)
- Flatten → Dense(128, ReLU) → Dense(287, linear)
- **Optimizer:** Adam (lr=0.0005)

#### Regularized Deeper CNN (`Regularized_Deeper_Model`)

A deep CNN with data augmentation, batch normalization, L2 regularization, spatial dropout, and global average pooling:

- **Input:** 32 × 32 × 1
- **Data augmentation:** RandomRotation(0.02), RandomZoom(0.05), RandomTranslation(height=0.05, width=0.05)
- **Rescaling** to [-1, 1]
- **Block 1:** Conv2D(32, 3×3, same) → BatchNorm → ReLU → Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2) → SpatialDropout2D(0.10)
- **Block 2:** Conv2D(64, 3×3, same) → BatchNorm → ReLU → Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2) → SpatialDropout2D(0.15)
- **Block 3:** Conv2D(128, 3×3, same) → BatchNorm → ReLU → Conv2D(128, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2) → SpatialDropout2D(0.20)
- **Block 4:** Conv2D(128, 3×3, same) → BatchNorm → ReLU
- **GlobalAveragePooling2D** → Dense(128, ReLU) → Dropout(0.4) → Dense(287, linear)
- **Regularization:** L2 (1e-3) on all Conv2D and the Dense(128) kernels; He Normal initialization
- **Optimizer:** Adam (lr=2.5e-4)
- **Loss:** SparseCategoricalCrossentropy (from logits)
- **Metrics:** Accuracy, Top-5 accuracy

## Model Selection

All six models were trained for up to 300 epochs per fold with early stopping (`monitor="val_loss"`, `patience=15`, `restore_best_weights=True`) using stratified 5-fold cross-validation.

### Cross-Validation Results

| Model                        | Mean Train Acc | Mean CV Acc | STD Train Acc | STD CV Acc | Mean Gap |
| ---------------------------- | -------------: | ----------: | ------------: | ---------: | -------: |
| Simpler ANN                  |        18.30%  |      7.67%  |        0.0985 |     0.0393 |  10.63%  |
| Deeper ANN                   |        56.66%  |     19.35%  |        0.0491 |     0.0049 |  37.30%  |
| Regularized Deeper ANN       |        60.83%  |     24.86%  |        0.0153 |     0.0078 |  35.96%  |
| Simpler CNN                  |        51.92%  |     20.66%  |        0.0441 |     0.0096 |  31.26%  |
| Deeper CNN                   |        68.14%  |     38.07%  |        0.0346 |     0.0089 |  30.07%  |
| Regularized Deeper CNN       |        85.70%  |     51.52%  |        0.0112 |     0.0122 |  34.18%  |

The **Regularized Deeper CNN** achieved the highest mean cross-validation accuracy (51.52%) among the six evaluated models and was selected for further training.

Key observations from the model-selection stage:
- CNN architectures consistently outperformed ANN architectures.
- Adding depth and regularization improved cross-validation accuracy at each step.
- The regularized deeper CNN had the lowest standard deviation on training accuracy (0.0112), indicating stable training.
- All models showed a substantial train–CV gap, reflecting the difficulty of the 287-class problem with only 48 examples per class.

## Final Model

### Architecture

The selected model is the **Regularized Deeper CNN** described above, consisting of four convolutional blocks (7 Conv2D layers total), global average pooling, and a single dense hidden layer before the output.

### Training Configuration

In the final notebook (`geez_ocr.ipynb`), the selected model was further cross-validated with enhanced callbacks, then trained as follows:

- **Data split:** Stratified `train_test_split` — 90% development, 10% validation (`random_state=42`)
- **Epochs:** Up to 100
- **Callbacks:**
  - **EarlyStopping** — `monitor="val_accuracy"`, `mode="max"`, `patience=8`, `restore_best_weights=True`, `min_delta=1e-3`
  - **ReduceLROnPlateau** — `monitor="val_loss"`, `mode="min"`, `factor=0.5`, `patience=5`, `min_lr=1e-5`
- **Optimizer:** Adam (lr=2.5e-4)
- **Loss:** SparseCategoricalCrossentropy (from logits)
- **Metrics:** Accuracy, Top-5 accuracy

Training stopped at epoch 54 via early stopping. The best weights were restored from **epoch 46**.

## Data Augmentation and Regularization

The following techniques are implemented in the final model:

| Technique                | Details                                          |
| ------------------------ | ------------------------------------------------ |
| Random rotation          | ±0.02 (fraction of 2π)                           |
| Random zoom              | ±0.05                                            |
| Random translation       | ±0.05 (height and width)                         |
| Batch normalization      | After every Conv2D layer                         |
| L2 weight regularization | 1e-3 on all Conv2D and Dense(128) kernels        |
| Spatial dropout          | 0.10, 0.15, 0.20 after blocks 1, 2, 3           |
| Dense dropout            | 0.40 before the output layer                     |
| Early stopping           | Monitors val_accuracy, patience=8                |
| Learning-rate reduction  | Halves LR on val_loss plateau, patience=5        |
| Pixel rescaling          | [0, 255] → [-1.0, 1.0]                          |
| He Normal initialization | Applied to Conv2D and Dense kernel initializers  |

## Results

### Development/Validation Results

| Metric                 | Development Set | Validation Set |
| ---------------------- | --------------: | -------------: |
| Accuracy               |         90.14%  |        85.27%  |
| Top-5 accuracy         |         99.48%  |        97.82%  |
| Gap (Train – Val)      |                 |   4.88 pp      |

### Full Training Set Evaluation

| Metric             |   Value |
| ------------------ | ------: |
| Training accuracy  | 89.66%  |
| Training top-5 acc | 99.31%  |

### Test Set Evaluation

| Metric          |   Value |
| --------------- | ------: |
| Test accuracy   | 76.79%  |
| Test top-5 acc  | 94.98%  |
| Test loss       | 1.2571  |

The model achieves approximately 95% top-5 accuracy on the test set, meaning the correct character is among the model's top 5 predictions for the vast majority of test samples.

### Generalization Gap

There is a notable difference between validation accuracy (~85.27%) and test accuracy (~76.79%). Several factors may contribute to this gap:

- **Limited training data** — With only 48 images per class, the model has limited examples to learn the full variation within each character class.
- **Distribution differences** — The validation set is drawn from the same distribution as the training set (via stratified split), while the test set may contain different visual styles, noise levels, or writing variations.
- **Visually similar characters** — Many Ge'ez characters differ by small strokes or attachments. On noisy or degraded test images, these distinctions may be lost.
- **Test set difficulty** — One possible explanation for the lower test-set accuracy is the difficulty and visual noise present in some test examples. Several samples appear challenging even for human identification. This remains an observation/hypothesis rather than a quantitatively verified explanation, since the project does not currently include a human-agreement study or a formal measure of image quality/noise.

## Regular and Irregular Fidels

The project separately evaluates the model's performance on regular and irregular Ge'ez fidels using the training dataset.

Regular fidels follow consistent vowel-modification patterns across the 34 base consonants and 7 vowel orders (238 classes, 11,424 images). Irregular fidels have unique modifications due to the shapes of their base characters (49 classes, 2,352 images).

| Subset          | Accuracy | Top-5 Accuracy |
| --------------- | -------: | -------------: |
| Regular fidels  |  89.64%  |        99.29%  |
| Irregular fidels|  89.71%  |        99.40%  |

An interesting observation is that the irregular fidels did not perform noticeably worse than the regular fidels in this evaluation.

**Important limitation:** These measurements were performed on the **training dataset**, not on a separate held-out set. They indicate that the model has learned both regular and irregular character patterns from the training data, but they should not be interpreted as a held-out generalization benchmark for irregular characters.

## Error Analysis

The project generates classification reports (via `sklearn.metrics.classification_report`) on both the training and test sets, reporting per-class precision, recall, and F1-score across all 287 classes.

Misclassification visualizations display images with their actual and predicted labels side-by-side using the Ebrima font for Ge'ez character rendering. These visualizations are generated for both the training and test sets.

Common error patterns include confusion between characters that share similar structural elements — particularly characters from different vowel orders of the same consonant, or characters from different consonants that happen to look similar at 32×32 resolution.

## Limitations

1. **Small dataset** — 48 training images and 5 test images per class limit the model's ability to learn robust representations across all character variations.
2. **Validation vs. test gap** — The ~8.5 percentage-point drop from validation to test accuracy suggests distribution differences or greater difficulty in the test set that have not been fully characterized.
3. **Irregular fidel evaluation** — The regular/irregular fidel comparison is performed on the training data and therefore does not provide a generalization benchmark.
4. **No human baseline** — The project does not include a human accuracy study, making it difficult to contextualize the model's error rate.
5. **Image quality variation** — At 32×32 resolution, fine details that distinguish similar characters may be lost, especially in noisy or degraded samples.
6. **Single dataset source** — All data comes from a single dataset source, which may not represent the full diversity of Ge'ez handwriting or printing styles.

## Future Work

The planned next stage focuses on improving the dataset to strengthen model generalization:

- **Font-based augmentation** — Generate additional Ge'ez character images using multiple Ge'ez fonts to increase visual diversity.
- **Expanded data collection** — Collect more real handwritten/printed examples, particularly for visually similar and difficult characters.
- **Increased variation** — Introduce greater variation in character appearance (size, stroke width, noise, style) to build more robust representations.
- **Targeted improvement** — Improve representation of difficult and visually similar character pairs that currently cause the most confusion.
- **Retraining and evaluation** — Retrain the model on the expanded dataset and evaluate on genuinely unseen examples to measure improvement.

The goal is to improve generalization and robustness rather than simply increase training accuracy. Future experiments should evaluate newly collected data separately so that improvements can be measured on genuinely unseen examples, avoiding potential data leakage between training and evaluation.

Additional directions that may be explored:

- Higher-resolution input images to preserve fine character details.
- Transfer learning from pre-trained feature extractors.
- Confusion-focused training strategies for commonly confused character pairs.
- A formal human-agreement study to establish an upper bound on classification accuracy for this dataset.

## Installation

The project requires Python 3 and the following packages:

- **TensorFlow / Keras** — Model building, training, data loading, and augmentation
- **NumPy** — Array operations
- **scikit-learn** — Stratified K-fold cross-validation, train/test splitting, classification reports
- **Matplotlib** — Visualization and plotting

No `requirements.txt` file is currently included in the repository. Install the dependencies manually:

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

> **Note:** Ensure your TensorFlow version supports the Keras 3 API used in the code (TensorFlow ≥ 2.16 or standalone Keras 3). The saved model uses the `.keras` format.

## Usage

### 1. Obtain the dataset

Download the [Yaredoffice/geez-characters](https://huggingface.co/datasets/Yaredoffice/geez-characters) dataset and organize it into the `dataset/train/` and `dataset/test/` directories with numeric subfolder names (0–286).

### 2. Run model selection (optional)

Open and run `model_selection.ipynb` to reproduce the stratified 5-fold cross-validation comparison of all six models. This notebook trains each model and generates the comparison table.

### 3. Train the final model

Open and run `geez_ocr.ipynb`. This notebook:
1. Loads and explores the dataset
2. Further cross-validates the selected model with enhanced callbacks
3. Trains the final model on a 90/10 dev/val split with early stopping
4. Saves the trained model as `final_geez_cnn.keras`
5. Evaluates on the full training set and test set
6. Generates classification reports and misclassification visualizations
7. Separately evaluates regular and irregular fidels

### 4. Load the saved model

```python
import tensorflow as tf
model = tf.keras.models.load_model("final_geez_cnn.keras")
```

### 5. Inspect predictions

The notebook provides utility functions for visualization:

```python
from utils import show_images_with_prediction, show_misclassifications

show_images_with_prediction(model, X_test, y_test, num_of_images=20)
show_misclassifications(model, X_test, y_test, num_of_images=30)
```

## Saved Model

The repository contains the trained final model:

```
final_geez_cnn.keras    (~6.3 MB)
```

This is the Regularized Deeper CNN trained with early stopping (best weights restored from epoch 46 of 54).

## Technologies

| Technology       | Usage                                                 |
| ---------------- | ----------------------------------------------------- |
| Python 3         | Programming language                                  |
| TensorFlow/Keras | Model definition, training, data loading, augmentation|
| NumPy            | Array manipulation and data handling                  |
| scikit-learn     | StratifiedKFold, train_test_split, classification_report |
| Matplotlib       | Training curves and image visualizations              |

## Results Summary

This project demonstrates a systematic approach to Ge'ez character classification across 287 classes. Through comparison of six architectures via stratified 5-fold cross-validation, the Regularized Deeper CNN was selected as the most promising model. After final training with early stopping and learning-rate scheduling, the model achieves:

- **89.66%** training accuracy and **76.79%** test accuracy
- **99.31%** training top-5 accuracy and **94.98%** test top-5 accuracy
- A 4.88 percentage-point train–validation gap, suggesting reasonable but not overfitting

The remaining gap between validation and test performance, along with the limited dataset size (48 images per class), represents the primary area for improvement. The planned expansion of the dataset through additional fonts and collected examples aims to improve generalization rather than simply increase training accuracy.
