import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # # Debugging: Check if images and labels are loaded
    # print(f"Loaded {len(images)} images and {len(labels)} labels.")

    # if len(images) == 0 or len(labels) == 0:
    #     sys.exit("Error: No images or labels were loaded. Check your dataset path.")

    # # Convert labels to categorical format
    # labels = tf.keras.utils.to_categorical(labels)

    # # Split data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(
    #     np.array(images), np.array(labels), test_size=TEST_SIZE
    # )

    # Get a compiled neural network model
    model = get_model()

    # Train model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate model performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    images = []
    labels = []
    
    print(f"Checking dataset path: {data_dir}")

    # Loop through each category directory
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        # Ensure the category path exists
        if not os.path.isdir(category_path):
            print(f"Skipping missing directory: {category_path}")
            continue

        # Iterate over image files in the category directory
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)

            # Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                continue

            # Resize the image to the required size
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Append the processed image and its label
            images.append(image)
            labels.append(category)

    print(f"Total images loaded: {len(images)}")
    print(f"Total labels loaded: {len(labels)}")

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = keras.Sequential([
        # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer

        # Second convolutional + pooling layer (64 filters)
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer (128 filters)
        layers.Conv2D(128, (3, 3), activation="relu"),

        # Flattening the 2D feature maps into a 1D vector
        layers.Flatten(),

        # Fully connected layer with 128 neurons
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # Dropout to reduce overfitting

        # Output layer with softmax activation for multi-class classification
        layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
