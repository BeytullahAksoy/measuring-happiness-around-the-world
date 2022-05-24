import os
import cv2
import json
from keras.models import load_model
import numpy as np

#it returns array of binary class prediction results of user's images
def binary_emotion_predict():
    path = "user_data/"
    dir_list = os.listdir(path)
    results = []
    count = 0
    model = load_model('emotionclassification/binary_emotion.h5')

    for image_path in dir_list:
        image = cv2.imread(path+image_path)
        image = cv2.resize(image, (48, 48))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        yhat = model.predict(np.expand_dims(gray_image/255, 0))

        if yhat > 0.5:
            #print(f'Predicted class is not Happy')
            results.append(0)
        else:
            #print(f'Predicted class is Happy')
            results.append(1)

        with open('results/results.txt', 'w') as filehandle:
            json.dump(results, filehandle)

    return results





