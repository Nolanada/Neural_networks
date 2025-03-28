import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

# Load trained model
MODEL_PATH = "model0.h5"  # Change this to your actual model file
model = tf.keras.models.load_model(MODEL_PATH)

def load_image(image_path):
    """ Load and preprocess an image for model prediction. """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize
    image = image.astype("float32") / 255.0  # Normalize

    return image

def predict(image_path):
    """ Run prediction on the selected image. """
    image = load_image(image_path)

    if image is None:
        result_label.config(text="Error: Could not load image!")
        return

    prediction = model.predict(np.array([image]))  # Model expects a batch
    predicted_class = np.argmax(prediction)  # Get highest probability class
    confidence = np.max(prediction) * 100  # Convert to percentage

    result_label.config(text=f"Predicted: {predicted_class} ({confidence:.2f}%)")

def browse_image():
    """ Open file dialog to select an image and display it. """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.ppm;*.png;*.jpg;*.jpeg")])
    
    if file_path:
        # Display image
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize for GUI display
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img  # Keep reference

        # Run prediction
        predict(file_path)

# GUI Setup
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("400x500")

# Widgets
title_label = Label(root, text="Traffic Sign Recognition", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

image_label = Label(root)  # For displaying the selected image
image_label.pack()

browse_button = Button(root, text="Browse Image", command=browse_image, font=("Arial", 12))
browse_button.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14), fg="blue")
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
