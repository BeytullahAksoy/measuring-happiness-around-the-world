from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray

def emotion_prediction(image_param):

    image = Image.open(image_param)
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))

    numpydata = asarray(image).ravel()



    model = load_model('Face_Emotion_detection.h5')

    numpydata = numpydata.reshape(-1, 48, 48, 1)

    predictions = model.predict(numpydata)



    max_value = max(predictions[0])



    return np.where(predictions[0] == max_value)[0][0]


