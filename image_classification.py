# image_classification.py
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Custom function to load the model
def load_custom_model(model_path):
    from tensorflow.keras.utils import get_custom_objects
    from tensorflow.keras.layers import DepthwiseConv2D
    
    # Fix the issue with the groups parameter
    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, **kwargs):
            if 'groups' in kwargs:
                del kwargs['groups']
            super().__init__(**kwargs)
    
    get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})
    return load_model(model_path, compile=False)

# Load the model
model = load_custom_model("keras_model.h5")

# Load the labels
class_names = open("labels.txt", "r").readlines()

def classify_image(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score
