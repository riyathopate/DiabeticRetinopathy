from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the trained model
model = load_model('best_model.h5')

# Function to preprocess image
def preprocess_image(image):
    image_size = (224, 224)
    image = image.resize(image_size)
    image = np.array(image)
    image = image.astype("float") / 255.0
    return image

# Function to predict image class
def predict_image(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    confidence = np.max(prediction)
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded image file
        file = request.files['file']
        if file:
            # Read image file
            image = Image.open(BytesIO(file.read()))
            # Predict image class
            predicted_class, confidence = predict_image(image)
            # Convert prediction results to JSON format
            prediction_result = {'prediction': predicted_class, 'confidence': float(confidence)}
            return jsonify(prediction_result)
        else:
            return jsonify({'error': 'No file uploaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
