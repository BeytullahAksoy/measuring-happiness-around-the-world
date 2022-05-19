import cv2
from facer_dir import facer
from PIL import Image

def face_extraction_from_video(video_path):
    #create face dataset from video
    cap = cv2.VideoCapture(video_path)


    c = 1
    count = 0
    file = 0

    while True:

        grabbed, frame = cap.read()

        if c % 400 == 0:
            try:
                faces = facer.extract_faces(img_path=frame, align=True, threshold=0.95)
            except:
                break
            if len(faces) > 0:

                count += len(faces)

                for face in faces:

                    im = Image.fromarray(face)

                    im.save(f"/user_data/person-{file}.jpeg")
                    file += 1


            cv2.waitKey(1)
        c += 1

    cap.release()


