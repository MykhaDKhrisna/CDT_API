import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify

# Load the saved model without custom objects (if any)
model = tf.keras.models.load_model("D:/Dewa/Magang/CDT/CDT-API-Network-main/CDT-API-Network-main/resnet125_model2.h5", compile=False)

# Define a function to preprocess the image
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = np.mean(img, axis=-1, keepdims=True)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Define a function to predict classes
def predict_classes(image_path, model):
    target_size = (224, 224)  # Update this to match your model's input size
    preprocessed_img = preprocess_image(image_path, target_size)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)  # Add batch dimension
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index

# Define class labels
class_labels = ['Berat', 'Sedang', 'Ringan', 'Tidak']  # Update these labels according to your classes

# Create a Flask app
app = Flask(__name__)

# Ensure the temporary directory exists
os.makedirs("temp", exist_ok=True)

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Save the file to a temporary location
        temp_image_path = os.path.join("temp", file.filename)
        file.save(temp_image_path)

        # Predict the class of the image
        predicted_class_index = predict_classes(temp_image_path, model)
        predicted_class_label = class_labels[predicted_class_index]

        # Remove the temporary file
        os.remove(temp_image_path)

        return jsonify({'predicted_class': predicted_class_label})

    return jsonify({'error': 'File processing error'}), 500

if __name__ == '__main__':
    app.run(debug=False)
