from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray

model = load_model('./binary_emotion.h5')

def load_model_func_b():
    model = load_model('./binary_emotion.h5')


def predict_b(image_path):
    """A function takes path of image and predicts emotion"""
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))
    numpydata = asarray(image).ravel()
    numpydata = numpydata.reshape(-1, 48, 48, 1)
    predictions = model.predict(numpydata)
    predictions = predictions[0]
    print(predictions)
    if predictions <=0.5:
        return 0
    else :
        return 1





    return  predicted_class
