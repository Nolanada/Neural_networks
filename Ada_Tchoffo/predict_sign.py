import os
import numpy as np
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Constants
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 43
MODEL_PATH = "best_model2.h5"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (Update with actual traffic sign names if available)
CLASS_NAMES = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60",
    4: "Speed Limit 70", 5: "Speed Limit 80", 6: "End of Speed Limit 80", 7: "Speed Limit 100",
    8: "Speed Limit 120", 9: "No Overtaking", 10: "No Overtaking for Trucks",
    11: "Right of Way at Intersection", 12: "Priority Road", 13: "Yield", 14: "Stop",
    15: "No Vehicles", 16: "No Trucks", 17: "No Entry", 18: "General Caution", 
    19: "Dangerous Curve Left", 20: "Dangerous Curve Right", 21: "Double Curve",
    22: "Bumpy Road", 23: "Slippery Road", 24: "Road Narrows Right", 25: "Construction",
    26: "Traffic Signals", 27: "Pedestrians", 28: "Children Crossing", 29: "Bicycles Crossing",
    30: "Beware of Ice/Snow", 31: "Wild Animals", 32: "End of All Restrictions", 
    33: "Turn Right Ahead", 34: "Turn Left Ahead", 35: "Ahead Only", 36: "Go Right or Straight",
    37: "Go Left or Straight", 38: "Keep Right", 39: "Keep Left", 40: "Roundabout",
    41: "End of No Passing", 42: "End of No Passing for Trucks"
}

def predict_sign(image_path):
    """
    Predicts the traffic sign from an image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: (predicted_class (int), sign_name (str), confidence (float))
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return None

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Reshape for model input

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    probability = np.max(predictions)

    return predicted_class, CLASS_NAMES.get(predicted_class, "Unknown Sign"), probability

def upload_image():
    """Handles image upload and displays the prediction."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.ppm")])
    if not file_path:
        return

    # Predict the traffic sign
    result = predict_sign(file_path)
    if result is None:
        prediction_label.config(text="Error: Could not process image")
        return

    predicted_class, sign_name, probability = result
    prediction_label.config(text=f"Prediction: {sign_name} (Class {predicted_class})\nConfidence: {probability:.2f}")

    # Display the uploaded image
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Tkinter GUI
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("400x500")

Label(root, text="Upload a Traffic Sign Image", font=("Arial", 14)).pack(pady=10)
Button(root, text="Upload Image", command=upload_image, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

prediction_label = Label(root, text="", font=("Arial", 12))
prediction_label.pack(pady=10)

root.mainloop()
