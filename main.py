from flask import Flask, render_template, redirect, url_for, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import gdown  # Tambahkan ini

app = Flask(__name__, static_url_path='/static')

model_path = "psoriasis_model.h5"

# Cek apakah model sudah ada, kalau belum, download dari Google Drive
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1nZGoLrNFUwdtTo8Q49zejYt1swI9MvGf"  # Ganti dengan ID kamu
    gdown.download(url, model_path, quiet=False)

# Setelah file tersedia, baru load model
model = load_model(model_path, compile=False)

# Kelas prediksi
class_labels = [    
    'Psoriasis Gutteta',
    'Psoriasis Inversus',
    'Psoriasis Pustular',
    'Psoriasis Vulgaris'
]

@app.route('/')
def index():
    return redirect(url_for('klasifikasi'))

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('Klasifikasi.html')

@app.route('/information')
def information():
    return render_template('informasi.html')

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
