import json

import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)

# Load the saved machine learning model
model2 = tf.keras.models.load_model('save_at_50.keras')

class_names2 = ["Recyclable", "Non-Recyclable"]


@app.route('/', methods=['POST'])
def predict():
    # Read the image from the request as a binary file
    image_data = request.get_data()

    # convert the binary data to an image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Decode the JPEG image
    image = tf.image.resize(image, (180, 180))

    # Add the batch dimension
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Create batch axis

    prediction2 = model2.predict(image_array)
    print(prediction2[0])
    test2 = np.argmax(prediction2[0])
    print(class_names2[test2])

    return json.dumps(class_names2[test2])


