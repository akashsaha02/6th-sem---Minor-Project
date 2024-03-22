import joblib
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Load the neural network model
model = load_model('BrainTumor10Epochs.h5')
print('Neural Network Model loaded.')

# Load the KNN model
knn_model = joblib.load('knnmodel.pkl')
print('KNN Model loaded.')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Get prediction using Neural Network
        neural_network_result = getResult(file_path)

        # Get prediction using KNN
        image = cv2.imread(file_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        image = np.array(image)
        input_image = image.flatten().reshape(1, -1)
        knn_result = knn_model.predict(input_image)

        # Get the class names
        neural_network_class = get_className(np.argmax(neural_network_result))
        knn_class = get_className(knn_result[0])

        # Return the results
        return f"Neural Network Prediction: {neural_network_class}<br>KNN Prediction: {knn_class}"

    return None


if __name__ == '__main__':
    app.run(debug=True)
