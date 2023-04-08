import json

import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)

# Load the saved machine learning model
model= tf.keras.models.load_model('save_at_50.keras')

class_names = ["Recyclable", "Non-Recyclable"]


@app.route('/', methods=['POST'])
def predict():
    try:
        # Read the image from the request as a binary file
        image_data = request.get_data()

        # convert the binary data to an image using PIL
        image = Image.open(io.BytesIO(image_data))

        # Resize the image to match the input size of the model
        image = image.resize((180, 180))

        # Convert the PIL image to a NumPy array
        image_array = np.asarray(image)

        # Normalize the image array
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Add the batch dimension
        image_tensor = np.expand_dims(normalized_image_array, axis=0)

        # Predict the class probabilities using the model
        predictions = model.predict(image_tensor)

        # Get the class label with the highest probability
        predicted_class = class_names[np.argmax(predictions)]

        # Return the predicted class as a JSON response
        response = {
            'predicted_class': predicted_class
        }
        return jsonify(response)
    except Exception as e:
        print("Error: ", e)
        return "Error: " + str(e), 500  # return an HTTP 500 error code



if __name__ == '__main__':
    app.run(host='0.0.0.0')
