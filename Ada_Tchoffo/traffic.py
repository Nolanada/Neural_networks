import os
import numpy as np
import cv2
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import collections

# Constants
EPOCHS = 10
IMG_HEIGHT = 32 
NUM_CATEGORIES = 43
TEST_SIZE = 0.2

def main():
    if len(sys.argv) != 2:
        print("Usage: python traffic.py data_directory")
        return

    images, labels = load_data(sys.argv[1])

    if len(images) == 0 or len(labels) == 0:
        print("No images or labels were loaded. Check your dataset path!")
        return

    print("Class Distribution:", collections.Counter(labels))

    labels = tf.keras.utils.to_categorical(labels, NUM_CATEGORIES)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test, verbose=2)

    model.save("best_model2.h5")
    print("Model saved as best_model2.h5")

def load_data(data_dir):
    """Loads images and labels from the dataset directory."""
    images, labels = [], []

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue

        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Skipping unreadable image {image_path}")
                continue

            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resizing to 32x32
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize

            images.append(image)
            labels.append(category)

    return images, labels

def get_model():
    """Defines and returns a CNN model based on MobileNetV2."""
    
    # Define base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False,
        weights=None  
    )
    base_model.trainable = True  # Allow training from scratch

    # Create the model
    model = keras.Sequential([
        base_model,  
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CATEGORIES, activation="softmax")  # Match the number of categories
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    main()
