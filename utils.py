"""
utils.py — Helper functions for Plant Disease Detection
--------------------------------------------------------
This file contains:
  - CLASS_NAMES : list of all 38 plant disease labels
  - plot_history()  : visualise training curves
  - plot_confusion_matrix() : evaluate model performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ── 38 PlantVillage class labels ─────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def plot_history(history):
    """
    Plot training vs validation accuracy and loss curves.

    WHY: These curves tell us if the model is learning correctly.
      - If train accuracy goes up but val accuracy stays low → OVERFITTING
      - If both go up together → GOOD training
      - If both stay low → model is too simple (underfitting)

    Args:
        history: the object returned by model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training Results", fontsize=16, fontweight="bold")

    # — Accuracy plot —
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val  Accuracy",  linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # — Loss plot —
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val  Loss",  linewidth=2, linestyle="--")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("📊 Saved: training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names=None, top_n=15):
    """
    Plot a confusion matrix for the top N classes.

    WHY: A confusion matrix shows which classes the model gets
    confused between. For example, it might confuse
    'Tomato Early Blight' with 'Tomato Late Blight'.

    Args:
        y_true      : true labels (list of integers)
        y_pred      : predicted labels (list of integers)
        class_names : list of string labels
        top_n       : only show top N most-frequent classes (keeps plot readable)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Limit to top_n classes for readability
    labels   = list(range(min(top_n, len(class_names))))
    cm       = confusion_matrix(y_true, y_pred, labels=labels)
    cm_names = [class_names[i].split("___")[-1][:20] for i in labels]  # shorten names

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cm_names,
        yticklabels=cm_names,
    )
    plt.title("Confusion Matrix (Top Classes)", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("📊 Saved: confusion_matrix.png")
