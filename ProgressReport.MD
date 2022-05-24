<div id="top"></div>  
<!-- PROJECT LOGO -->  
<br />  
<div align="center">  
    
  
  <h3 align="center">Measuring Happiness Around The World</h3>  
  
  <p align="center">  
    Progress Report  
    
   Beytullah Aksoy  
   20170808016  
    ·  
   
  </p>  
</div>  
  
  
  
  
  
## About The Project  
  
The aim of this project is to try to measure the emotional state and happiness levels of people in various cities of the world by detecting their facial expressions.  
  
  
## Goals In The Project
Street walking videos in various cities show people's natural daily life situations. I aimed to save human faces from these videos and find their emotional states from these faces.

 
### Challenge: Extract Faces from Videos  
Most current models can't find the small faces in the videos I want to extract. Therefore, I couldn't get the faces of most of the people who appeared in the video with these models. The models I created myself and the models I used ready (Haar Cascades, Dlib) were unsuitable for my project.  
Therefore, by detecting a larger object, I decided to follow people's entire bodies instead of faces and save them as images. I was going to extract the faces of the people I recorded. It was easier to create an effective model to follow people rather than face tracking. I tried assigning IDs to the people in the videos and recording each person once. Then I would extract the faces from these recorded people. Since I assigned ID to each person and saved once, there would be no data duplication. Because people got ahead of each other, sometimes the same person could be perceived as a new object and most of the time the human tracking was wrong. For this reason, tracking an object in videos was not an effective solution.  
That's why I decided to take only certain frames from the videos and extract the faces here. The videos I used to extract the dataset included street walks. Since only the faces of people coming from the opposite side were visible in the video, the faces were only briefly visible in the video. In order not to save a person more than once, I tried to face extract in one of every 400 frames. For face removal, I took the project called Retina Face and modified it to be suitable for my own project. This project was very effective at finding small size faces. In this way, I extracted people from videos from various parts of the world and created my dataset.  
  
### Challenge: Facial Emotion Recognition  
Emotion recognition from face has not yet achieved very high results. The most popular data set in this field is fer-2013 and the highest rate reached with this data set is 75.2.Source:[Stanford](http://cs230.stanford.edu/projects_winter_2020/reports/32610274.pdf)  
This dataset contains 7 kinds of emotions. But what matters to me is whether the people in the videos are happy or not. Therefore, I decided to create two types of models. While the first model is a model that classifies 7 emotional states, my second model is a model that makes binary classification.  
The binary model achieved a validation accuracy score of approximately 83%. In contrast, the multiclassifier model has accuracy rate of 55%.  
I decided to use these two models to measure world happiness.  So I can compare results of my models.
  
  
### Outcome: World Happiness Report  
  
I collected 15000 pictures of people from various cities. The number of people varies, as I eliminated the corrupted ones and the smaller ones from these pictures.  Number of face images used :  
Barcelona: 5325  
Istanbul: 10766  
Cairo: 3320  
Copenhagen :5083  
London: 9233  
New York: 5870  
Paris: 4981  
Tokyo: 2490  
And I tried these pictures on both my models. In both models, Tokyo was the second unhappiest city, and Cairo the unhappiest city. Suicide rates are very high in Japan. Egypt ranks among the most unhappy countries in most happiness surveys. In any case, the most common mood in the results is neutral. People generally do not feel in a mood when they are alone while walking on the street. Therefore, neutral mood was expected to be the most in the results. These show that the result of the study makes sense
The happiness rate was found by dividing the number of happy people by the total number of people in that city.
The results:  
-with multiclass classifier model
![](https://i.ibb.co/NTQhg1T/results1.png)  
  
with binary classifier model
![](https://i.ibb.co/rQZwQ9h/results2.png)  
  
  
### Outcome: Emotion Recognition Binary Classifier  
The notebook I used to create the model is in the directory (emotionclassification/binary_emotion.ipynb) in github.
In my dataset, I used the happy pictures in fer2013 for the happy class and the other pictures for the not happy class.  
I used TensorFlow Keras Sequential API and  TensorFlow Keras layers to create a CNNs model.

The Model Building:
![](https://i.ibb.co/QdVDzYt/Screenshot-from-2022-05-22-16-58-23.png)
Fer-2013 dataset contains grayscale images. So, the dimension of the input layer is 1. Size of the images are 48x48. So, shape in the input layer is (48, 48, 1).
I chose architectural decisions(HyperParameters) through experimentation.
I used Conv2D after each convolution operation to reduce image size in half.
The first layer has 16 filters and the filters have size 3x3 ant it has stride of 1.
The second layer has 32 filters and the filters have size 3x3 ant it has stride of 1.
The third layer has 16 filters and the filters have size 3x3 ant it has stride of 1.
I added the Flatten and Dense Layers, as I wanted to obtain a single value as it should be in the CNN structure. Second to last Dense Layer has 48 parameters, as in Input Layer .
The last Dense layer has value 1 because this model is a binary classifier. By sigmoid function, we get 0 or 1 as result. 
If result is 0 , it is gonna be happy person and if result is 1, it is gonna be a not happy person.

Compiling Model:
![](https://i.ibb.co/JKpHh8g/Screenshot-from-2022-05-22-17-18-23.png)
I both tried adam and nadam optimizer and decided to use adam.
I decided to user Binary Cross Entrpy loss function because it is a binary classifier model.
The model is valid for the project with an accuracy rate of 0.83.  
  
![](https://i.ibb.co/989YTmX/download.png)  
  
  
![](https://i.ibb.co/0XKJ7tT/accuracy.png)  
  
  
  
![](https://i.ibb.co/vZ0mbRB/loss.png)  
  
  
### Outcome: Emotion Recognition Multiclass Classifier  
  
I developed the model using the fer2013 dataset.  
Architecture of the model:  

The Model Building:
![](https://i.ibb.co/XyS2VqT/Screenshot-from-2022-05-22-18-46-53.png)

The input shape of input layer is the same as the binary model as it uses the same dataset.

The ouput layer has 7 value because the dataset has 7 different classes.


![](https://i.ibb.co/R02H08R/son.png)
![](https://i.ibb.co/wdtjDPC/accuracy.png)
![](https://i.ibb.co/BPBnL9N/loss.png)  
  
### Outcome: Streamlit App  
I turned the project into an app.  
I made the tools I used in the project available in this application.

Home Page:
![](https://i.ibb.co/ZcPjNZF/Screenshot-from-2022-05-20-22-04-50.png)


Visualization of The Happiness Report with Artificial Intelligence, the result of my project:
![](https://i.ibb.co/bF4GQ1M/worldreport.png)

Extracting faces from uploaded image or video:
![](https://i.ibb.co/tXPvXQT/faceextract2-1.png)
![](https://i.ibb.co/YBdr538/faceextract2-2.png)



Find and visualize emotion of the people in the uploaded video.
![](https://i.ibb.co/1Z0jRvK/emotionvid.png )
