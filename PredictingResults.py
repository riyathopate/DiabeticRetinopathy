import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# Load label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the trained model
model = load_model('best_model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    image_size = (224, 224)
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = image.astype("float") / 255.0
    return image


# Function to predict image class
def predict_image(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    confidence = np.max(prediction)
    return predicted_class, confidence


# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, confidence = predict_image(file_path)
        result_label.config(text=f"Prediction: {predicted_class}\nConfidence: {confidence * 100:.2f}%")
        display_image(file_path)
    else:
        messagebox.showerror("Error", "No file selected.")

# Function to display image in GUI
def display_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# GUI setup
root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x400")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
