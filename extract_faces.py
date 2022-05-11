import cv2
from facer_dir import facer
from PIL import Image


#create face dataset from video
cap = cv2.VideoCapture("istanbul4.mp4")

print("Showing frames...")
c = 1
count = 0
save_time = 0
file = 3951
while True:

    grabbed, frame = cap.read()

    if c % 200 == 0:
        try:
            faces = facer.extract_faces(img_path=frame, align=True, threshold=0.95)
        except:
            print("An exception occurred")
        if len(faces) > 0:

            count += len(faces)

            for face in faces:
                print("saved")
                im = Image.fromarray(face)
                im.save(f"/home/beytu/measuring-happiness-around-the-world/facer_dir/istanbul/person-{file}.jpeg")
                file += 1

        cv2.waitKey(1)
    c += 1


    if count > 10000:
        break
    print(count)

cap.release()


