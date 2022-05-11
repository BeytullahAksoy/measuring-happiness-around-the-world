import cv2
from facer_dir import facer
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
from numpy import asarray
import csv
from PIL import Image


#create face dataset from video
cap = cv2.VideoCapture("istanbul.mp4")
face_data = []
print("Showing frames...")
c = 1
count = 0
save_time = 0
file = 0
while True:

    grabbed, frame = cap.read()

    if c % 200 == 0:

        faces = facer.extract_faces(img_path=frame, align=True, threshold=0.95)
        if len(faces) > 0:
            face_data.append(faces)
            count += len(faces)
        cv2.waitKey(1)
    c += 1
    if count > 2 :
        print("in")

        for faces in face_data:
            for face in faces:
                im = Image.fromarray(face)

                im.save(f"your_file{file}.jpeg")
                file += 1
        face_data = []

        save_time+=1
    if save_time>10:
        break
    print(count)

cap.release()


