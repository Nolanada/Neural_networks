import cv2
import numpy as np
import sys
import tensorflow as tf

# Constants (same as used in training)
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

def main():
    # Check for correct usage
    if len(sys.argv) != 3:
        sys.exit("Usage: python recognition.py model.h5 image_path")

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    image = load_image(image_path)

    if image is None:
        sys.exit("Error: Could not load image. Check file path.")

    # Make a prediction
    prediction = model.predict(np.array([image]))  # Model expects a batch
    predicted_class = np.argmax(prediction)  # Get highest probability class
    confidence = np.max(prediction) * 100  # Convert to percentage

    print(f"Predicted Traffic Sign: {predicted_class} (Confidence: {confidence:.2f}%)")

def load_image(image_path):
    """
    Load and preprocess an image for model prediction.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Resize image to match model input size
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize pixel values (optional but recommended)
    image = image.astype("float32") / 255.0

    return image

if __name__ == "__main__":
    main()
