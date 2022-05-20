# Importing required libraries, obviously
import plost as plost
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from extract_faces import face_extraction_from_video
import multiclass_emotion_classification
from facer_dir import facer
from PIL import Image
import os, shutil

import os
import zipfile
import pandas as pd
import numpy as np
from emotionclassification import predict
import json
from visualization import multi_visualization,binary_visualization

def save_faces(video_file):

    import cv2 as cv
    import tempfile

    f = video_file

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    cap = cv.VideoCapture(tfile.name)

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

                    im.save(f"user_data/person-{file}.jpeg")
                    file += 1

            cv2.waitKey(1)
        c += 1

    cap.release()



def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

def delete_user_images():
    folder = "user_data/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def save_uploaded_file(uploadedfile):
    with open(os.path.join("",uploadedfile.name)) as f:
        f.write(uploadedfile.getbuffer())
    return st.success("saved")



def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 

		The algorithm has four stages:

			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Adaboost Training
			4. Cascading Classifiers



Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')


def main():
    delete_user_images()
    st.title("Face Analyze and Extraction Platform")


    activities = ["Home", "World Happiness Report" ,"Face Extraction from Image","Emotion Analyze in Video","Face Extraction from Video","About Models"]
    choice = st.sidebar.selectbox("Pick an option", activities)

    if choice == "Home":

        st.markdown("## World Happinnes Report Through AI")
        st.markdown("""
            Today, there are studies that reveal the happiness report in the world.
            Which participants were selected? How were the participants evaluated? 
            Did the participants make objective comments? Questions like these cast doubt on the accuracy of this report. 
            

            You can find the report [here](https://worldpopulationreview.com/country-rankings/happiest-countries-in-the-world)
            If there are people laughing and having fun on the street in a city, you will feel that the level of happiness is high in that city. 
            This study, on the other hand, aims to calculate the happiness index by looking at the facial expressions of people on the street, 
            not by the declaration.To review the report please select World Happiness Report from left menu.
            
        """)
        st.markdown("## Facial Emotion Analyze and Face Extraction from Images and Videos")
        st.markdown("""
                    Use the left menu to use the tools used in this study (face removal from video, emotion from pictures).

                """)





    elif choice == "World Happiness Report":
        st.markdown("## World Happinnes Report Through AI")
        st.markdown("""
        
        
        It is  used two different ML models from emotion predicition.One is binary classifier(happy or not happy), and the other one is multiclass classifiers
        (angry,disgust,happy,fear,sad,neutral,sad,surprise).
        The binary model has 0.83% accuracy.
        The multiclassifier model 0.69% accuracy.
        
        """)
        option = st.selectbox(
            'Select Model',
            ('Multi-Class', 'Binary'))

        if option == "Multi-Class":
            if st.button('Show'):
                total_df = multi_visualization("world_results/multi")
                total_df = total_df.pivot_table('Count', ['City'], 'Emotion')
                total_df["Happiness Rate"] = (total_df["happy"] * 100) / (total_df["angry"] + total_df["disgust"] + total_df["happy"] + total_df["fear"] + total_df["neutral"] + total_df["surprise"] + total_df["sad"])
                st.dataframe(total_df.style.highlight_max(axis=0))


        if option == "Binary":
            if st.button('Show'):
                total_df = binary_visualization("world_results/binary")
                #st.dataframe(total_df.style.highlight_max(axis=0))
                medals = total_df.pivot_table('Count',['City'],'Emotion')
                medals =  medals.rename(index={"happy": "not happy", "not happy": "happy", "Happiness Rate": "Happiness Rate"})
                medals["Happiness Rate"] = (medals["happy"]*100) /(medals["happy"] + medals["not happy"])
                medals = medals.sort_values(by=['Happiness Rate'], ascending=False)
                st.dataframe(medals)
    elif choice == "Face Extraction from Image":


        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:



            image = Image.open(image_file)
            image_path = "user_data"


            image.save(f"{image_file.name}")

            st.image(Image.open(image_file), caption='Enter any caption here')

            face_num=0
            if st.button("Process"):
                faces = facer.extract_faces(img_path=f"{image_file.name}",align= True)
                if len(faces)>0:
                    st.success("Extracted Faces")
                else:
                    st.success("No face found")
                for face in faces:
                    st.image(face)
                    img = Image.fromarray(face)
                    img.save(f"{image_path}/{face_num}.jpeg")
                    face_num+=1
            def zip_directory(folder_path, zip_path):
                with zipfile.ZipFile(zip_path, mode='w') as zipf:
                    len_dir_path = len(folder_path)
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path[len_dir_path:])

            zip_directory('user_data', 'faces.zip')


            with open("faces.zip", "rb") as fp:

                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name="faces.zip",
                    mime="application/zip"
                )



    elif choice == "Emotion Analyze in Video":
        delete_user_images()

        video_file = st.file_uploader("Upload video", type=['mp4'])

        if video_file is not None:
            # file_details = {"FileName":video_file.name,"FileType":video_file.type}
            # save_uploaded_file(video_file)

            option = st.selectbox("Select a classification model",
                ("Binary Classification",
                "Multiclass Classification"))

            if option == "Multiclass Classification":
                save_faces(video_file)
                path = "user_data/"
                dir_list = os.listdir(path)
                results = []
                predict.load_model_func()
                count = 0
                for image in dir_list:
                    output = predict.predict(path + image)

                    results.append(output)
                    count += 1
                    print(count)

                import numpy as np
                import pandas as pd

                array = np.array(results)

                unique, counts = np.unique(array, return_counts=True)

                result = np.column_stack((unique, counts))
                clean_data = [[item for item in row] for row in result]

                # clean_data: [[294, 294], [981, 981]]

                df = pd.DataFrame(clean_data, columns=("emotion","number"))
                Subjects = {0: "angry",
                            1: "disgust",
                            2: "fear",
                            3: "happy",
                            4: "neutral",
                            5: "sad",
                            6: "surprise"
                            }

                df["emotion"] = df["emotion"].map(Subjects)

                plost.bar_chart(data=df,
                                bar='emotion', height=400, width=100,
                                value=['number'],
                                group=True)

                with open('results/user_results.txt', 'w') as filehandle:
                    json.dump(results, filehandle)

            if option == "Binary Classification":
                save_faces(video_file)
                path = "user_data/"
                dir_list = os.listdir(path)
                results = []
                predict.load_model_func()
                count = 0
                for image in dir_list:
                    output = predict.predict(path + image)

                    results.append(output)
                    count += 1
                    print(count)

                import numpy as np
                import pandas as pd

                array = np.array(results)

                unique, counts = np.unique(array, return_counts=True)

                result = np.column_stack((unique, counts))
                clean_data = [[item for item in row] for row in result]

                # clean_data: [[294, 294], [981, 981]]

                df = pd.DataFrame(clean_data, columns=("emotion","number"))
                Subjects = {0: "angry",
                            1: "disgust",
                            2: "fear",
                            3: "happy",
                            4: "neutral",
                            5: "sad",
                            6: "surprise"
                            }

                df["emotion"] = df["emotion"].map(Subjects)

                plost.bar_chart(data=df,
                                bar='emotion', height=400, width=100,
                                value=['number'],
                                group=True)

                with open('results/user_results.txt', 'w') as filehandle:
                    json.dump(results, filehandle)



            def zip_directory(folder_path, zip_path):
                with zipfile.ZipFile(zip_path, mode='w') as zipf:
                    len_dir_path = len(folder_path)
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path[len_dir_path:])



            zip_directory('user_data', 'faces.zip')

            with open("faces.zip", "rb") as fp:

                btn = st.download_button(
                    label="Download Zipped Faces",
                    data=fp,
                    file_name="faces.zip",
                    mime="application/zip"
                )





    elif choice == "About Models":
        option = st.selectbox(
            'Select Model',
            ('CNN for Binary Classification', 'CNN for MultiClassification'))

        if option == 'CNN for Binary Classification':
            pass
        elif option == 'CNN for MultiClassification':
            pass


    elif choice == "Face Extraction from Video":
        delete_user_images()




        video_file = st.file_uploader("Upload video", type=['mp4'])

        if video_file is not None:
            #file_details = {"FileName":video_file.name,"FileType":video_file.type}
            #save_uploaded_file(video_file)




            if st.button("Extract Faces"):
                print(f"type:{video_file.type}")
                save_faces(video_file)
               # face_extraction_from_video(video_file.filename)

            def zip_directory(folder_path, zip_path):
                with zipfile.ZipFile(zip_path, mode='w') as zipf:
                    len_dir_path = len(folder_path)
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path[len_dir_path:])

            zip_directory('user_data', 'faces.zip')

            with open("faces.zip", "rb") as fp:

                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name="faces.zip",
                    mime="application/zip"
                )


if __name__ == "__main__":
    main()
