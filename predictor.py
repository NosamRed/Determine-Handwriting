import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np
from tensorflow import keras
from PIL import Image

MODEL_PATH = "emnist_balanced_cnn.keras"
model = keras.models.load_model(MODEL_PATH)

EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

def preprocess_image(img_path, save_debug=False):
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img)

    if img_array.mean() > 127:
        img_array = 255 - img_array

    img_array = img_array.astype("float32") / 255.0
    x = np.expand_dims(img_array, axis=(0, -1))

    if save_debug:
        debug_img = (img_array * 255).astype(np.uint8)
        Image.fromarray(debug_img).save("debug_preprocessed.png")

    return x

def predict_from_path(image_path, save_debug=False):
    x = preprocess_image(image_path, save_debug=save_debug)

    pred = model.predict(x, verbose=0)[0]
    predicted_class = int(np.argmax(pred))

    predicted_label = (
        EMNIST_LABELS[predicted_class]
        if predicted_class < len(EMNIST_LABELS)
        else "UNKNOWN"
    )

    confidence = float(pred[predicted_class])

    top5 = np.argsort(pred)[-5:][::-1]
    top5_results = [
        {
            "index": int(i),
            "label": EMNIST_LABELS[i] if i < len(EMNIST_LABELS) else f"UNKNOWN({i})",
            "confidence": float(pred[i])
        }
        for i in top5
    ]

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top5": top5_results
    }

if __name__ == "__main__":
    image_path = "test-image.png"
    result = predict_from_path(image_path, save_debug=True)

    print(f"Predicted class index: {result['predicted_class']}")
    print(f"Predicted label: {result['predicted_label']}")
    print("\nTop 5 predictions:")
    for item in result["top5"]:
        print(f"{item['index']}: {item['label']} -> {item['confidence']:.4f}")