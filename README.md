# BONE-FRACTURE-DETECTION
I have created a comprehensive `README` document for your **Bone Fracture Detection** repository.

Since automated document creation in Google Workspace is currently disabled in this environment, here is the complete, richly formatted content ready for your repository or document:

---

# 🦴 Bone Fracture Detection using Convolutional Neural Networks (CNN)

An end-to-end Deep Learning and Computer Vision framework designed to automatically analyze and classify X-ray images into **Fractured** and **Not Fractured** diagnostic categories using TensorFlow and Keras.

## 📌 Table of Contents

* [Project Overview]
* [Dataset Architecture]
* [Model Architecture]
* [Training & Performance Metrics]
* [Installation & Setup]
* [Usage & Pipeline Guide]
* [Project Structure]
* [License]

## 📌 Project Overview

Bone fracture identification in medical imaging requires precise analysis by clinical specialists. In high-volume emergency and radiological settings, automated diagnostic tools serve as vital decision-support systems to reduce evaluation turnaround times and triage urgent cases.

This repository provides a complete deep learning workflow implemented in `bone.ipynb`, covering data preprocessing, exploratory class distribution analysis, data augmentation, custom CNN model construction, model training, evaluation, and inference on unseen test samples.

---

## ✨ Key Features

* **Custom CNN Architecture:** Multi-layer convolutional architecture featuring sequential `Conv2D`, `MaxPooling2D`, and `Dense` layers optimized for binary classification.
* **Data Augmentation Pipeline:** Real-time image transformations using `ImageDataGenerator` (rescaling, shearing, zooming, horizontal flips) to prevent overfitting and improve generalization.
* **Exploratory Data Analysis (EDA):** Visualizations detailing dataset balance, including categorical bar charts and pie charts across training, validation, and testing splits using Seaborn and Matplotlib.
* **Performance Analytics:** Detailed tracking of loss and accuracy metrics across training and validation steps, achieving ~89% test accuracy on unseen data.
* **Inference Ready:** Python-native inference routine to load trained `.h5` model artifacts and evaluate raw X-ray inputs.

---

## 📊 Dataset Architecture

The dataset is split across `train`, `val`, and `test` subsets, with subdirectories corresponding to the target binary labels:

```text
dataset/
├── train/
│   ├── fractured/       # 4,623 images
│   └── not fractured/   # 4,623 images
├── val/
│   ├── fractured/       # 415 images
│   └── not fractured/   # 414 images
└── test/
    ├── fractured/       # 253 images
    └── not fractured/   # 253 images

```

### Dataset Summary Table

| Dataset Split | Fractured Count | Not Fractured Count | Total Images | Input Resolution |
| --- | --- | --- | --- | --- |
| **Training** | 4,623 | 4,623 | 9,246 | 180 × 180 × 3 |
| **Validation** | 415 | 414 | 829 | 180 × 180 × 3 |
| **Testing** | 253 | 253 | 506 | 180 × 180 × 3 |

---

## 🧠 Model Architecture

The model leverages sequential convolutional blocks followed by dense classification layers built using the Keras Sequential API:

| Layer (Type) | Output Shape | Kernel / Configuration | Param # |
| --- | --- | --- | --- |
| **InputLayer** | (None, 180, 180, 3) | Target Size: 180x180 RGB | 0 |
| **Conv2D** | (None, 178, 178, 32) | 32 filters, 3x3, ReLU | 896 |
| **MaxPooling2D** | (None, 89, 89, 32) | Pool size: 2x2 | 0 |
| **Conv2D** | (None, 87, 87, 64) | 64 filters, 3x3, ReLU | 18,496 |
| **MaxPooling2D** | (None, 43, 43, 64) | Pool size: 2x2 | 0 |
| **Conv2D** | (None, 41, 41, 128) | 128 filters, 3x3, ReLU | 73,856 |
| **MaxPooling2D** | (None, 20, 20, 128) | Pool size: 2x2 | 0 |
| **Conv2D** | (None, 18, 18, 256) | 256 filters, 3x3, ReLU | 295,168 |
| **MaxPooling2D** | (None, 9, 9, 256) | Pool size: 2x2 | 0 |
| **Flatten** | (None, 20736) | Flatten 2D map to 1D | 0 |
| **Dense** | (None, 256) | FC Dense, ReLU | 5,308,672 |
| **Dense (Output)** | (None, 1) | Binary Output, Sigmoid | 257 |

* **Total Trainable Parameters:** 5,697,345 (~21.73 MB)
* **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
* **Optimizer:** Adam (`learning_rate = 0.001`)

---

## 📈 Training & Performance Metrics

Model progress monitored across initial training epochs:

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| --- | --- | --- | --- | --- |
| **Epoch 1** | 62.64% | 0.6474 | 83.11% | 0.4487 |
| **Epoch 2** | 85.51% | 0.3479 | 86.85% | 0.4050 |
| **Epoch 3** | 92.59% | 0.2029 | **87.09%** | **0.3683** |

### Evaluation on Holdout Test Set

* **Test Loss:** `0.28`
* **Test Accuracy:** **`89.00%`**

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.8+
* Jupyter Notebook or Google Colab environment

### 1. Clone Repository

```bash
git clone https://github.com/shashwat123u/BONE-FRACTURE-DETECTION.git
cd BONE-FRACTURE-DETECTION

```

### 2. Install Python Dependencies

```bash
pip install tensorflow pandas numpy pillow seaborn matplotlib

```

---

## 🚀 Usage & Pipeline Guide

### 1. Execute Training Notebook

Launch the Jupyter environment and run `bone.ipynb`:

```bash
jupyter notebook bone.ipynb

```

### 2. Custom Inference Script

Run the following code snippet to classify an individual X-ray image using the exported model file:

```python
import numpy as np
import tensorflow as tf
from PIL import Image

def predict_fracture(image_path, model_path='fracture_classification_model.h5'):
    # Load trained model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    img = Image.open(image_path).convert('RGB').resize((180, 180))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    class_names = ['fractured', 'not fractured']
    predicted_class = class_names[int(prediction >= 0.5)]
    
    print(f"Prediction: {predicted_class} (Score: {prediction:.4f})")

# Example usage:
predict_fracture('test/fractured/example_xray.jpg')

```

---

## 📁 Project Structure

```text
BONE-FRACTURE-DETECTION/
│
├── train/                            # Training image directory (fractured / not fractured)
├── val/                              # Validation image directory (fractured / not fractured)
├── test/                             # Testing image directory (fractured / not fractured)
│
├── bone.ipynb                        # Primary Jupyter notebook containing entire pipeline
├── fracture_classification_model.h5  # Saved trained model artifact
└── README.md                         # Detailed project documentation

```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
