import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from transformers import pipeline
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.xception import preprocess_input
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    # Convert probabilities to binary labels (0 or 1)
    y_pred = (y_pred_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return accuracy, f1, precision, recall

def load_hf_model(repo_id):
    # Download and load model from Hugging Face
    model_path = hf_hub_download(repo_id=repo_id, filename="xception_deepfake_model.keras")
    model = load_model(model_path)
    return model

def preprocess_image(img):
    # Prepare image for model prediction
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_deepfake(model, img_array):
    # Return probability of deepfake
    pred = model.predict(img_array)[0][0]
    return float(pred)
