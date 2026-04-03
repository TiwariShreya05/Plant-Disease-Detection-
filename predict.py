"""
predict.py — Run predictions on new leaf images
-------------------------------------------------
WHAT THIS FILE DOES:
  Loads the saved model and predicts the disease class
  for any new leaf image you provide.

HOW TO RUN:
  cd src
  python predict.py --image path/to/leaf.jpg

  OR import in your own script:
    from predict import predict_disease
    label, confidence = predict_disease("leaf.jpg")
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from utils import CLASS_NAMES

IMG_SIZE   = 224
MODEL_PATH = "../models/plant_disease_model.h5"


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for the model.

    Steps:
      1. Open image with PIL
      2. Resize to 224x224 (what MobileNetV2 expects)
      3. Convert to numpy array
      4. Normalize pixels to [0, 1]
      5. Add batch dimension: (224,224,3) → (1,224,224,3)

    Args:
        image_path: path to the image file

    Returns:
        numpy array of shape (1, 224, 224, 3)
    """
    img        = Image.open(image_path).convert("RGB")   # ensure 3-channel RGB
    img        = img.resize((IMG_SIZE, IMG_SIZE))          # resize
    img_array  = np.array(img, dtype=np.float32) / 255.0  # normalize
    img_array  = np.expand_dims(img_array, axis=0)         # add batch dim
    return img, img_array


def predict_disease(image_path: str, model_path: str = MODEL_PATH, top_k: int = 3):
    """
    Predict the plant disease from a leaf image.

    Args:
        image_path : path to the leaf image
        model_path : path to the saved .h5 model
        top_k      : show top-k predictions (default 3)

    Returns:
        (predicted_class, confidence) tuple
    """
    # Load the saved model
    print(f"🔄 Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    # Preprocess
    img, img_array = preprocess_image(image_path)

    # Predict — model outputs a (1, 38) probability array
    predictions = model.predict(img_array, verbose=0)[0]   # shape: (38,)

    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]

    print("\n" + "="*50)
    print("🌿 PREDICTION RESULTS")
    print("="*50)
    for rank, idx in enumerate(top_indices):
        label  = CLASS_NAMES[idx]
        conf   = predictions[idx] * 100
        marker = "✅" if rank == 0 else "  "
        print(f"{marker} #{rank+1}  {label:<50}  {conf:.2f}%")
    print("="*50)

    best_label = CLASS_NAMES[top_indices[0]]
    best_conf  = predictions[top_indices[0]] * 100

    # Visualise the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Predicted: {best_label.split('___')[-1]}\nConfidence: {best_conf:.1f}%",
        fontsize=13,
        fontweight="bold",
        color="green" if "healthy" in best_label else "red",
    )
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    plt.show()

    return best_label, best_conf


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the leaf image (jpg/png)"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help="Path to the trained model (.h5)"
    )
    args = parser.parse_args()

    label, confidence = predict_disease(args.image, args.model)
    print(f"\n🎯 Final Answer: {label} ({confidence:.2f}% confidence)")
