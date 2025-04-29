from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model
model = tf.keras.models.load_model('fracture_classification_model.h5')

# Prediction function
def predict_fracture(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    prediction = prediction[0][0]

    if prediction > 0.5:
        return "not fractured"
    else:
        return "fractured"

# Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_fracture(filepath)
            image_url = os.path.join('uploads', filename)
            return render_template('upload.html', prediction=result, image_url=image_url)
    return render_template('upload.html', prediction=None, image_url=None)

if __name__ == '__main__':
    app.run(debug=True)
