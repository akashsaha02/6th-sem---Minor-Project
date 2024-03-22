import joblib
import cv2
import numpy as np
from PIL import Image
from keras.utils import normalize

# Load the KNN model
knn_model = joblib.load('knnmodel.pkl')

INPUT_SIZE = 64

# Function to preprocess images
def preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    image = Image.fromarray(image, "RGB")
    image = image.resize((input_size, input_size))
    image_array = np.array(image)
    image_normalized = normalize(image_array, axis=1)
    image_flattened = image_normalized.reshape(1, -1)
    return image_flattened


# Path to the new image
new_image_path = 'pred/pred11.jpg'

# Preprocess the new image
new_image = preprocess_image(new_image_path, INPUT_SIZE)

# Make predictions
prediction = knn_model.predict(new_image)

# Get the predicted class
predicted_class = np.argmax(prediction)

# Output the prediction
if predicted_class == 0:
    print("No Tumor")
else:
    print("Tumor Detected")

