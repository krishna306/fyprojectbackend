from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.metrics import Precision,Recall
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import keras.utils as image
import base64
import io
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/InceptionV3.hdf5'

# Load your trained model
# model = load_model(MODEL_PATH)

model =load_model(MODEL_PATH)
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def model_predict(img_path, model):
    img_height = 224
    img_width = 224
    class_names =['FMD', 'IBK', 'LSD']
    img = image.load_img(img_path, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    output = model.predict(img)
    output = output[0]
    result  = class_names[np.argmax(output)]
    return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Server Running on Port 5000"
    # return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST': 
        file = request.files['file']
        if file.filename == '':
            return "Empty File"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            extension = filename.split('.')[1]
            file.save("C:/Users/krish/OneDrive/Flask/" + "predict."+extension)
            path_to_file = "C:/Users/krish/OneDrive/Flask/" + "predict." + extension
            predictedDisease = model_predict(path_to_file, model)
            return predictedDisease
        return None
    return None
if __name__ == '__main__':
    app.run(debug=True)