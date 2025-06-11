from flask import Flask, render_template, redirect, url_for, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import requests

app = Flask(__name__)

# Cek dan download model jika belum ada
MODEL_PATH = 'psoriasis_model.h5'
MODEL_URL = 'https://drive.google.com/file/d/1ToVYnRTAjOgV9mZG1E_x6VHD7rnpDULL/view?usp=sharing'  # <--- Ganti ini!

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model: {response.status_code}")

# WAJIB: download dulu sebelum load
download_model()

# Load the CNN model once when the app starts
model = load_model(MODEL_PATH, compile=False)

class_labels = [
    'Psoriasis Gutteta',
    'Psoriasis Inversus ',
    'Psoriasis Pustular',
    'Psoriasis vulgaris'
]

@app.route('/')
def index():
    return redirect(url_for('klasifikasi'))

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('Klasifikasi.html')

@app.route('/information')
def information():
    return render_template('infromasi.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_index = int(np.argmax(predictions))
        predicted_label = class_labels[predicted_index]

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)