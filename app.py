import plost as plost
import streamlit as st
import cv2
from facer_dir import facer
from PIL import Image
import os
import zipfile

import json
from visualization import multi_visualization, binary_visualization
import tempfile
import numpy as np
import pandas as pd
import shutil

from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray



def predict_b(image_path):
    """A function takes path of image and predicts emotion"""
    model_b = load_model('./binary_emotion.h5')

    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))
    numpydata = asarray(image).ravel()
    numpydata = numpydata.reshape(-1, 48, 48, 1)
    predictions = model_b.predict(numpydata)
    predictions = predictions[0]
    print(predictions)
    if predictions <=0.5:
        return 0
    else :
        return 1







def predict(image_path):
    """A function takes path of image and predicts emotion"""
    model = load_model('./Face_Emotion_detection.h5')
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))
    numpydata = asarray(image).ravel()
    numpydata = numpydata.reshape(-1, 48, 48, 1)
    predictions = model.predict(numpydata)


    predictions = predictions[0]

    max_value = max(predictions)
    predicted_class = int(np.where(predictions == max_value)[0][0])




    return  predicted_class

def delete_user_images():
    """A function to clean the user_data/ directory that stores files created by user when using streamlit."""
    folder = "user_data/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def save_faces(video_file):
    """A function detects faces in one of 400 frames and saves to the directory user_data/ """
    f = video_file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    cap = cv2.VideoCapture(tfile.name)
    c = 1
    file = 0

    while True:
        grabbed, frame = cap.read()
        if c % 400 == 0:
            try:
                faces = facer.extract_faces(img_path=frame, align=True, threshold=0.95)
            except:
                break
            if len(faces) > 0:
                for face in faces:
                    im = Image.fromarray(face)
                    im.save(f"user_data/person-{file}.jpeg")
                    file += 1
           # cv2.waitKey()
        c += 1
    cap.release()


def zipdir(path, ziph):
    """A function that creates zip with given folder"""
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )




def save_uploaded_file(uploadedfile):
    """A function to save uploaded file in streamlit"""
    with open(os.path.join("", uploadedfile.name)) as f:
        f.write(uploadedfile.getbuffer())
    return st.success("saved")


def main():

    st.title("Face Analyze and Extraction Platform")
    activities = [
        "Home",
        "World Happiness Report",
        "Face Extraction from Image",
        "Emotion Analyze in Video",
        "Face Extraction from Video",
        "About Models",
    ]
    choice = st.sidebar.selectbox("Pick an option", activities)

    if choice == "Home":
        st.markdown("## World Happinnes Report Through AI")
        st.markdown(
            """
            Today, there are studies that reveal the happiness report in the world.
            Which participants were selected? How were the participants evaluated? 
            Did the participants make objective comments? Questions like these cast doubt on the accuracy of this report. 


            You can find the report [here](https://worldpopulationreview.com/country-rankings/happiest-countries-in-the-world)
            If there are people laughing and having fun on the street in a city, you will feel that the level of happiness is high in that city. 
            This study, on the other hand, aims to calculate the happiness index by looking at the facial expressions of people on the street, 
            not by the declaration.To review the report please select World Happiness Report from left menu.

        """
        )
        st.markdown(
            "## Facial Emotion Analyze and Face Extraction from Images and Videos"
        )
        st.markdown(
            """
                    Use the left menu to use the tools used in this study (face removal from video, emotion from pictures).

                """
        )

    elif choice == "World Happiness Report":
        st.markdown("## World Happinnes Report Through AI")
        st.markdown(
            """


        It is  used two different ML models from emotion predicition.One is binary classifier(happy or not happy), and the other one is multiclass classifiers
        (angry,disgust,happy,fear,sad,neutral,sad,surprise).
        The binary model has 0.83% accuracy.
        The multiclassifier model 0.69% accuracy.


        """
        )
        #Select model to show results
        option = st.selectbox("Select Model", ("Multi-Class", "Binary"))

        if option == "Multi-Class":
            if st.button("Show"):
                total_df = multi_visualization("world_results/multi")
                total_df = total_df.pivot_table("Count", ["City"], "Emotion")
                total_df["Happiness Rate"] = (total_df["happy"] * 100) / (
                    total_df["angry"]
                    + total_df["disgust"]
                    + total_df["happy"]
                    + total_df["fear"]
                    + total_df["neutral"]
                    + total_df["surprise"]
                    + total_df["sad"]
                )
                st.dataframe(total_df.style.highlight_max(axis=0))

        if option == "Binary":
            if st.button("Show"):
                total_df = binary_visualization("world_results/binary")
                # st.dataframe(total_df.style.highlight_max(axis=0))
                medals = total_df.pivot_table("Count", ["City"], "Emotion")
                medals = medals.rename(
                    index={
                        "happy": "not happy",
                        "not happy": "happy",
                        "Happiness Rate": "Happiness Rate",
                    }
                )
                medals["Happiness Rate"] = (medals["happy"] * 100) / (
                    medals["happy"] + medals["not happy"]
                )
                medals = medals.sort_values(by=["Happiness Rate"], ascending=False)
                st.dataframe(medals.style.highlight_max(axis=0))

    elif choice == "Face Extraction from Image":

        image_file = st.file_uploader(
            "Upload image", type=["jpeg", "png", "jpg", "webp","HEIC"]
        )
        if image_file is not None:
            image = Image.open(image_file)
            image_path = "user_data"
            image.save(f"{image_file.name}")
            st.image(Image.open(image_file), caption="Uploaded Image")
            face_num = 0
            if st.button("Process"):
                faces = facer.extract_faces(img_path=f"{image_file.name}", align=True)
                if len(faces) > 0:
                    st.success("Extracted Faces")
                else:
                    st.success("No face found")
                for face in faces:
                    st.image(face)
                    img = Image.fromarray(face)
                    img.save(f"{image_path}/{face_num}.jpeg")
                    face_num += 1

            def zip_directory(folder_path, zip_path):
                with zipfile.ZipFile(zip_path, mode="w") as zipf:
                    len_dir_path = len(folder_path)
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path[len_dir_path:])

            zip_directory("user_data", "faces.zip")

            with open("faces.zip", "rb") as fp:

                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name="faces.zip",
                    mime="application/zip",
                )

            option = st.selectbox(
                "Select a classification model",
                ("Binary Classification", "Multiclass Classification"),
            )
            if st.button('Analyze'):
                if option == "Multiclass Classification":
                    path = "user_data/"
                    dir_list = os.listdir(path)
                    results = []
                    count = 0
                    for image in dir_list:
                        output = predict(path + image)
                        results.append(output)
                        count += 1

                    array = np.array(results)
                    unique, counts = np.unique(array, return_counts=True)
                    result = np.column_stack((unique, counts))
                    clean_data = [[item for item in row] for row in result]

                    df = pd.DataFrame(clean_data, columns=("emotion", "number"))
                    Subjects = {
                        0: "angry",
                        1: "disgust",
                        2: "fear",
                        3: "happy",
                        4: "neutral",
                        5: "sad",
                        6: "surprise",
                    }

                    df["emotion"] = df["emotion"].map(Subjects)

                    plost.bar_chart(
                        data=df,
                        bar="emotion",
                        height=400,
                        width=100,
                        value=["number"],
                        group=True,
                    )

                    with open("results/user_results.txt", "w") as filehandle:
                        json.dump(results, filehandle)

                if option == "Binary Classification":
                    path = "user_data/"
                    dir_list = os.listdir(path)
                    results = []

                    count = 0

                    for image in dir_list:
                        output = predict_b(path + image)

                        results.append(output)
                        count += 1

                    array = np.array(results)
                    unique, counts = np.unique(array, return_counts=True)
                    result = np.column_stack((unique, counts))
                    clean_data = [[item for item in row] for row in result]

                    df = pd.DataFrame(clean_data, columns=("emotion", "number"))
                    Subjects = {
                        0: "happy",
                        1: "not happy"
                    }
                    df["emotion"] = df["emotion"].map(Subjects)
                    print(df["emotion"])
                    plost.bar_chart(
                        data=df,
                        bar="emotion",
                        height=400,
                        width=100,
                        value=["number"],
                        group=True,
                    )

                    with open("results/user_results.txt", "w") as filehandle:
                        json.dump(results, filehandle)

                def zip_directory(folder_path, zip_path):
                    with zipfile.ZipFile(zip_path, mode="w") as zipf:
                        len_dir_path = len(folder_path)
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, file_path[len_dir_path:])

                zip_directory("user_data", "faces.zip")

                with open("faces.zip", "rb") as fp:
                    btn = st.download_button(
                        label="Download Zipped Faces",
                        data=fp,
                        file_name="faces.zip",
                        mime="application/zip",
                    )
            if st.button('Clear Dataset'):
                delete_user_images()

    elif choice == "Emotion Analyze in Video":
        video_file = st.file_uploader("Upload video", type=["mp4"])
        if video_file is not None:
            option = st.selectbox(
                "Select a classification model",
                ("Binary Classification", "Multiclass Classification"),
            )
            if st.button('Analyze'):
                if option == "Multiclass Classification":
                    save_faces(video_file)
                    path = "user_data/"
                    dir_list = os.listdir(path)
                    results = []
                    count = 0
                    for image in dir_list:
                        output = predict.predict(path + image)
                        results.append(output)
                        count += 1

                    array = np.array(results)
                    unique, counts = np.unique(array, return_counts=True)
                    result = np.column_stack((unique, counts))
                    clean_data = [[item for item in row] for row in result]

                    df = pd.DataFrame(clean_data, columns=("emotion", "number"))
                    Subjects = {
                        0: "angry",
                        1: "disgust",
                        2: "fear",
                        3: "happy",
                        4: "neutral",
                        5: "sad",
                        6: "surprise",
                    }

                    df["emotion"] = df["emotion"].map(Subjects)

                    plost.bar_chart(
                        data=df,
                        bar="emotion",
                        height=400,
                        width=100,
                        value=["number"],
                        group=True,
                    )

                    with open("results/user_results.txt", "w") as filehandle:
                        json.dump(results, filehandle)

                    if st.button('Delete Images'):
                        delete_user_images()

                if option == "Binary Classification":
                    save_faces(video_file)
                    path = "user_data/"
                    dir_list = os.listdir(path)
                    results = []

                    count = 0

                    for image in dir_list:
                        output = predict_b(path + image)

                        results.append(output)
                        count += 1

                    array = np.array(results)
                    unique, counts = np.unique(array, return_counts=True)
                    result = np.column_stack((unique, counts))
                    clean_data = [[item for item in row] for row in result]

                    df = pd.DataFrame(clean_data, columns=("emotion", "number"))
                    Subjects = {
                        0: "happy",
                        1: "not happy",
                    }
                    df["emotion"] = df["emotion"].map(Subjects)

                    plost.bar_chart(
                        data=df,
                        bar="emotion",
                        height=400,
                        width=100,
                        value=["number"],
                        group=True,
                    )

                    with open("results/user_results.txt", "w") as filehandle:
                        json.dump(results, filehandle)

                def zip_directory(folder_path, zip_path):
                    with zipfile.ZipFile(zip_path, mode="w") as zipf:
                        len_dir_path = len(folder_path)
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, file_path[len_dir_path:])

                zip_directory("user_data", "faces.zip")

                with open("faces.zip", "rb") as fp:
                    btn = st.download_button(
                        label="Download Zipped Faces",
                        data=fp,
                        file_name="faces.zip",
                        mime="application/zip",
                    )
                if st.button('Clear Dataset'):
                    delete_user_images()

    elif choice == "About Models":
        option = st.selectbox(
            "Select Model",
            ("CNN for Binary Classification", "CNN for MultiClassification"),
        )

        if option == "CNN for Binary Classification":
            image = Image.open("images/binary_model.png")
            st.image(image)
            image2 = Image.open("images/accuracy_binary.png")
            st.image(image2)
            image3 = Image.open("images/loss_binary.png")
            st.image(image3)

        elif option == "CNN for MultiClassification":
            image = Image.open("images/multi_model.png")
            st.image(image)
            image2 = Image.open("images/accuracy_multi.png")
            st.image(image2)
            image3 = Image.open("images/loss_multi.png")
            st.image(image3)

    elif choice == "Face Extraction from Video":
        video_file = st.file_uploader("Upload video", type=["mp4"])

        if video_file is not None:
            if st.button("Extract Faces"):
                save_faces(video_file)

            #zip the directory that includes extracted face images
            def zip_directory(folder_path, zip_path):
                with zipfile.ZipFile(zip_path, mode="w") as zipf:
                    len_dir_path = len(folder_path)
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path[len_dir_path:])

            zip_directory("user_data", "faces.zip")
            #Download button to download the zip file
            with open("faces.zip", "rb") as fp:
                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name="faces.zip",
                    mime="application/zip",
                )

if __name__ == "__main__":
    main()
