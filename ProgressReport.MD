
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



### Challenge: Extract Faces from Videos
Most current models can't find the small faces in the videos I want to extract. Therefore, I couldn't get the faces of most of the people who appeared in the video with these models, and I couldn't create a data set. The models I created myself and the models I used ready (Haar Cascades, Dlib) were unsuitable for my project.
It was easier to create an effective model to follow people rather than face tracking. I tried assigning IDs to the people in the videos and recording each person once. Then I would remove the faces from these recorded people. But because there were too many people in the videos, a person often got more than one ID, and most of the time the human tracking was wrong. For this reason, tracking an object in videos was not an effective solution.
That's why I decided to take only certain frames from the videos and remove the faces here. The videos I used to extract the dataset included street walks. In order not to save a person more than once, I made face removal in one of every 400 frames. For face removal, I took the project called Retina Face and modified it to be suitable for my own project. This project was very effective at finding small size faces. In this way, I extracted people from videos from various parts of the world and created my dataset.

### Challenge: Facial Emotion Recognition
Emotion recognition from face has not yet achieved very high results. The most popular data set in this field is fer-2013 and the highest rate reached with this data set is 75.2.Source:[Stanford](http://cs230.stanford.edu/projects_winter_2020/reports/32610274.pdf)
This dataset contains 7 kinds of emotions. But what matters to me is whether the people in the videos are happy or not. Therefore, I decided to create two types of models. While the first model is a model that classifies 7 emotional states, my second model is a model that makes binary classification.
The binary model achieved a val_accuracy score of approximately 83%. In contrast, the multiclassifier model has an accuracy rate of 55%.
I decided to use these two models to measure world happiness.


### Outcome: World Happiness Report

I collected 15000 pictures of people from various cities. The number of people varies, as I eliminated the corrupted ones and the smaller ones from these pictures. But I realized that 2000 people are enough to look at a city. For example, 2000 people and 10000 people in Istanbul gave similar results. Number of images used:
Barcelona: 5325
Istanbul: 10766
Cairo: 3320
Copenhagen :5083
London: 9233
New York: 5870
Paris: 4981
Tokyo: 2490
And I tried these pictures on both my models. As a result of both models, Tokyo was the second from the last and Cairo the last happy city. Suicide rates are very high in Japan. Egypt is generally accepted as an underdeveloped country. h+Moods were mostly neutral in each city. p+People usually just walk down the street and show no emotion. These show that the result of the study makes sense.

The results:
with multiclass classifier model:
![](https://i.ibb.co/NTQhg1T/results1.png)

with binary classifier model:
![](https://i.ibb.co/rQZwQ9h/results2.png)


### Outcome: Emotion Recognition Binary Classifier
In my dataset, I used the happy pictures in fer2013 for the happy class and the other pictures for the not happy class.
The model is valid for the project with an accuracy rate of 0.83.

![](https://i.ibb.co/989YTmX/download.png)


![](https://i.ibb.co/0XKJ7tT/accuracy.png)



![](https://i.ibb.co/vZ0mbRB/loss.png)


### Outcome: Emotion Recognition Multiclass Classifier

I developed the model using the fer2013 dataset.
Architecture of the model:

![](https://i.ibb.co/R02H08R/son.png)

![](https://i.ibb.co/wdtjDPC/accuracy.png)

![](https://i.ibb.co/BPBnL9N/loss.png)

### Outcome: Streamlit App
I turned the project into an app.
I made the tools I used in the project available in this application.
Specifications:
Visualization of the Happiness Report with Artificial Intelligence, the result of my project.
Extracting faces from uploaded image.
Extracting faces from uploaded video.
Find and graph the emotion of the people in the uploaded video.

![](https://i.ibb.co/ZcPjNZF/Screenshot-from-2022-05-20-22-04-50.png)





