from flask import Flask, render_template, redirect, url_for, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the CNN model once when the app starts
model = load_model('psoriasis_model.h5', compile=False)

# Define the class labels according to the model's output
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
        # Read image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Preprocess the image to the model's expected input size
        img = img.resize((299, 299))  # Updated to 299x299 as per user input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]

        # Predict
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
