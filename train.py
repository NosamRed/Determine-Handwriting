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
import gzip
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# File Paths
# =========================
DATA_PATH = "data/"  # Folder where your .gz files are

train_images_path = DATA_PATH + "emnist-balanced-train-images-idx3-ubyte.gz"
train_labels_path = DATA_PATH + "emnist-balanced-train-labels-idx1-ubyte.gz"
test_images_path  = DATA_PATH + "emnist-balanced-test-images-idx3-ubyte.gz"
test_labels_path  = DATA_PATH + "emnist-balanced-test-labels-idx1-ubyte.gz"

# Where to save the model
MODEL_PATH = "emnist_balanced_cnn.keras"

# =========================
# Load IDX Files
# =========================
def load_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        images = data.reshape(-1, 28, 28)
    return images

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

x_train = load_images(train_images_path)
y_train = load_labels(train_labels_path)
x_test  = load_images(test_images_path)
y_test  = load_labels(test_labels_path)

# =========================
# Preprocessing
# =========================
# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Match EMNIST orientation
x_train = np.transpose(x_train, (0, 2, 1))
x_test = np.transpose(x_test, (0, 2, 1))

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

num_classes = len(np.unique(y_train))

print("Training samples:", x_train.shape)
print("Test samples:", x_test.shape)
print("Classes:", num_classes)
print("Label range:", y_train.min(), "to", y_train.max())

# =========================
# Build CNN Model
# =========================
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# =========================
# Compile
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train
# =========================
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# =========================
# Evaluate
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# =========================
# Save
# =========================
input("Press Enter to save the model, otherwise press Ctrl+C...")  # Wait for user input before saving

model.save(MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")