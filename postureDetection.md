# Posture detection via webcam

## Intro
A lot of "office-jobs" require us to sit for prolonged periods of time, which can have adverse effects on several parts of the body due to poor sitting posture. 
I'm (among a heap of others) guilty of leaning in all possible directions while sitting putting strain on my back and neck. 
The goal of this project is to combat my bad habits and create an automated system that can alert me in case my posture becomes suboptimal.

## Strategy

1. **Find posture detection method**
   
     Find the most robust computer vision based method for detecting posture real time while sitting at a desk

3. **Process real time camera feed**

     Use posture detection modell on real time camera feed to detect leaning, twisted body posture

4. **Feedback**

     Create some way for the system to alert me when necessary


### Detection method

I've found a computer vision solution from Google called [Mediapipe](https://developers.google.com/mediapipe/solutions/guide). 
This is a whole suite of different solutions for various machine learning problems. In this project I used their pose landmark detection models. 
These models take an image or video stream as input and try to find various key points (coordinates within the image) with respect to given parameters. Key points are also called landmarks,
and refer to charactersitic body parst such as elbows, knees, shoulders etc. The following image shows the available landmarks.
![image](images/pose_landmarks_index.png)
*Available landmarks in mediapipe models along with their reference indices*

Since in my case I'll be sitting the camera will not be able to see most of my body. Fortunately these models are robust enought to produce an output (a number of landmarks) even with such input limitations. It would be nice to have additional information about the relative position of my lower body but it would also require me to sit really far from the camera which will not work with my integrated webcam. 

Given the constraints of my specific application I've decided to use the landmarks corresponding to the shoulders (11,12) and some on the face (0,7,8).
The position of the soulders should tell me if I'm leaning left/right (in-plane) or twisting (one shoulder is closer to camera). The landmarks on the head should 
allow for head posture characterisation such as left/right leaning, twisting and forward leaning. I should also mention that the models used not only return x,y coordinates for each landmark, but they also provide a z coordinate which represents the relative position of landmarks depth-wise. This will be especially useful for detecting leaning towards and away from the camera. 

The Mediapipe solution package offers multiple models (similar to MobileNetV2) for the same pose detection task. These are optimized for slightly different applications, which means a tradeoff between latency (due to processing time) and landmark accuracy and consistency. There are 3 models available: posture detector -lite, -heavy and -full. 
The lite model has the lowest latency but should also have a somewhat lower accuracy compared to the others. I tested all 3 to see which one would me the best. 



