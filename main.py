from emotion_predict import emotion_prediction
import os
import numpy as np


home_path = "./facer_dir/istanbul/"
files = os.listdir(home_path)
print(files[0])
emotions = []
print(len(files))

for face in files:
        image = f"{home_path}{face}"
        prediction = emotion_prediction(image_param=image)
        emotions.append(prediction)
        print("predictioning")
        if len(emotions)>100:
                break
print(emotions)
emotions = np.array(emotions)
np.savetxt('istanbul.txt', emotions)