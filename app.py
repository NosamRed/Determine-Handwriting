# =========================
# Suppress Warnings
# =========================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# =========================
# Imports
# =========================
import numpy as np
from tensorflow import keras
from PIL import Image

# =========================
# Load Model
# =========================

MODEL_PATH = "emnist_balanced_cnn.keras"
model = keras.models.load_model(MODEL_PATH)

# =========================
# EMNIST Labels
# =========================
emnist_labels = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# =========================
# Preprocess image function
# =========================
def preprocess_image(img_path, save_debug=True):
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img)

    # Make sure character is white and background is black
    if img_array.mean() > 127:
        img_array = 255 - img_array

    img_array = img_array.astype("float32") / 255.0
    x = np.expand_dims(img_array, axis=(0, -1))

    if save_debug:
        debug_img = (img_array * 255).astype(np.uint8)
        Image.fromarray(debug_img).save("debug_preprocessed.png")
        print("Saved debug image to debug_preprocessed.png")

    return x

# =========================
# Predict
# =========================
image_path = "test-image.png"
x = preprocess_image(image_path)

pred = model.predict(x, verbose=0)[0]
predicted_class = int(np.argmax(pred))

print(f"Predicted class index: {predicted_class}")

if predicted_class < len(emnist_labels):
    print(f"Predicted label: {emnist_labels[predicted_class]}")
else:
    print("Predicted label index is outside emnist_labels list!")

# Show top 5 predictions
top5 = np.argsort(pred)[-5:][::-1]
print("\nTop 5 predictions:")
for i in top5:
    label = emnist_labels[i] if i < len(emnist_labels) else f"UNKNOWN({i})"
    print(f"{i}: {label} -> {pred[i]:.4f}")