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

I've found a computer vision solution from Google called Mediapipe ()
