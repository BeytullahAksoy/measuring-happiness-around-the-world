import os
from emotionclassification import predict
import json


#it returns array of multiclass prediction results of user's images
def emotion_prediction():
    # Get the list of all files and directories
    path = "user_data"
    dir_list = os.listdir(path)
    results = []
    predict.load_model_func()

    for image in dir_list:
        output = predict.predict(path+image)

        results.append(output)


    with open('results/multiclass_results.txt', 'w') as filehandle:
        json.dump(results, filehandle)


    return results

#emotion_prediction()
