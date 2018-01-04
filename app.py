import os
from flask import Flask, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import h5py
import csv

from keras.models import load_model

import numpy as np
from skimage import io, color, exposure, transform


NUM_CLASSES = 43
IMG_SIZE = 48

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)
model = load_model('./traffic_sign_keras_model.h5')

classNames = {}

with open('sign_name.csv') as f:
    reader = csv.reader(f)
    for entry in reader:
        classNames[int(entry[0])] = entry[1]


def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


@app.route('/<path:path>', methods=["POST"])
def detect_sign(path):
    """
    Take the input image and process it.
    Return the class id.
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_filepath = os.path.join('./images/', filename)
        file.save(input_filepath)

    processed_image = preprocess_img(io.imread(input_filepath))
    response_arr = model.predict_classes(np.array([processed_image]))

    response_class = response_arr[0]
    response_class_name = classNames[response_class]
    os.remove(input_filepath)
    return response_class_name


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
