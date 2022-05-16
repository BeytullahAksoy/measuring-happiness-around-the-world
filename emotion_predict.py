import os
from emotionclassification import predict

# Get the list of all files and directories
path = "/home/beytu/measuring-happiness-around-the-world/facer_dir/istanbul/"
dir_list = os.listdir(path)
results = []
predict.load_model_func()

for image in dir_list:
    output = predict.predict(path+image)
    if output[1]:
        results.append(output[0])

import json

with open('istanbul.txt', 'w') as filehandle:
    json.dump(results, filehandle)


print(results)



