import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def load_trained_model(model_path='best_model.h5'):
    model = load_model(model_path)
    return model

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Failed to capture image")
    
    return frame

def predict_image(model, image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions

# Example usage
# model = load_trained_model()
# image = capture_image_from_webcam()
# predictions = predict_image(model, image)
# print(predictions)