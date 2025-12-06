# Fatigue Classification with PyTorch + Optuna

A portfolio project for **Data_science_demo_2025**.

This project builds an image‑based classifier that detects **fatigue** from facial images using **PyTorch** for model development and **Optuna** for hyperparameter optimization.

---

## 📌 Project Overview

This project aims to:

1. Download and explore the **Fatigue Dataset** from Kaggle.
2. Preprocess and augment image data.
3. Build a configurable PyTorch neural‑network classifier (CNN or transfer learning).
4. Use **Optuna** to search for the best hyperparameters.
5. Retrain the final model using the optimal configuration.
6. Evaluate performance on a test set.
7. Export the trained model for future inference.

---

## 📂 Dataset

**Dataset:** *Fatigue Dataset* from Kaggle
Kaggle ID: `rihabkaci99/fatigue-dataset`

Download using `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("rihabkaci99/fatigue-dataset")
print("Path to dataset files:", path)
```

### Dataset Notes

* Typically organized in folders like `fatigue/` and `non_fatigue/` or via paired CSV labels.
* Contains facial images for a binary classification task.
* Often requires preprocessing such as resizing, normalization, and augmentation.

---

## 🛠️ Installation & Requirements

Install required dependencies:

```bash
pip install torch torchvision optuna scikit-learn pandas pillow tqdm kagglehub
```

Example directory structure:

```
project/
│
├── README.md
├── data/
│   └── fatigue-dataset/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── optuna_search.py
│   └── eval.py
├── artifacts/
│   ├── best_model.pth
│   └── study_results/
└── notebooks/
    └── EDA.ipynb
```

---

## 🧩 Data Loading Example

```python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

dataset = ImageFolder(root="data/fatigue-dataset/train", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
```

---

## 🧠 Model & Optuna Hyperparameter Search

### Model Architecture Used

You used a **Custom CNN** built from scratch. The model includes configurable:

* Number of convolutional blocks
* Number of filters per block
* Dropout rate (optional)
* Fully connected layers

### Optuna Search Space Used

Your hyperparameter optimization tuned:

* **learning_rate** (lr)
* **number of CNN layers** (n_layers)
* **layer sizes / filters**

### Optimization Metric

You optimized **validation accuracy** (not F1).
Maximize **validation F1‑score**.

---

## 🧪 Training Process

1. Run Optuna to find best hyperparameters.
2. Log trial results to `artifacts/study_results/`.
3. Retrain best model on combined train + validation sets.
4. Evaluate on the test set.
5. Save final model:

```python
torch.save(model.state_dict(), "artifacts/best_model.pth")
```

---

## 📊 Evaluation Metrics

* Accuracy

* Confusion matrix


---


---

## 🔗 Summary

This project provides a complete pipeline for **fatigue detection from facial images**, combining deep learning best practices, reproducible training, and automated hyperparameter search. 
