import facer
a = facer.extract_faces(img_path="test.png", align=True, threshold=0.95)
print(a)