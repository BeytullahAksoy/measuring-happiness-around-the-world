from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray

model = load_model('./Face_Emotion_detection.h5')

def load_model_func():
    model = load_model('./Face_Emotion_detection.h5')


def predict(image_path):
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))
    high_prob = False


    numpydata = asarray(image).ravel()
    numpydata = numpydata.reshape(-1, 48, 48, 1)

    predictions = model.predict(numpydata)
    predictions = predictions[0]


    max_value = max(predictions)
    predicted_class = int(np.where(predictions == max_value)[0][0])
    predictions.sort()

    high_prob = True


    return  predicted_class,high_prob
