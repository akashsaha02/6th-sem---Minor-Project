import joblib
import cv2
import os
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.externals import joblib

image_directory = 'dataset/'

no_tumour_images = os.listdir(image_directory + 'no/')
yes_tumour_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64

for i, image_name in enumerate(no_tumour_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, "RGB") 
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)
        
for i, image_name in enumerate(yes_tumour_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, "RGB") 
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
        
dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Flatten the images
x_train_flatten = x_train.reshape(x_train.shape[0], -1)
x_test_flatten = x_test.reshape(x_test.shape[0], -1)

# Train the KNN model
knn_model.fit(x_train_flatten, y_train)

# Evaluate the KNN model
accuracy = knn_model.score(x_test_flatten, y_test)
print("KNN Accuracy:", accuracy)

# Save the KNN model
joblib.dump(knn_model, 'knnmodel.pkl')
