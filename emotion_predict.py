import os
from emotionclassification import predict
import json


def predict_image(image):

    return predict.predict(image)



def emotion_prediction():
    # Get the list of all files and directories
    path = "facer_dir/london/"
    dir_list = os.listdir(path)
    results = []
    predict.load_model_func()
    count = 0
    for image in dir_list:
        output = predict.predict(path+image)
        if output[1]:
            results.append(output[0])
        count += 1
        print(count)

    with open('results/london.txt', 'w') as filehandle:
        json.dump(results, filehandle)


    return results
#emotion_prediction()


