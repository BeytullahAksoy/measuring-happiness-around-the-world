from facer_dir import facer
faces = facer.detect_faces("people.jpg")

for i in faces:
    print(i)
    print(faces["face_1"]['facial_area'])
