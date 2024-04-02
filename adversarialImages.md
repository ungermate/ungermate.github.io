# Adversarial Images

## Intro
<div align="justify">
It feels like we are living in the future as AI (in some form) becomes part of more and more everyday events. People can't help but wonder where this might lead 
and how could things go wrong. This got me thinking: how can I make things go wrong? How wrong? 
<br>

Whith these questions in mind the goal of this project is to explore some of the vulnerabilities of neural networks used for visual image classification.
</div>

## Strategy

1. **Select models to attack**
   
     Find a few models with similar usecases but different complexities. 

3. **Set the models up, train them**

     Get a sense of how these would normally operate for later reference.

4. **Attack models**

     Create ways that mess with the models.

5. **Compare pre- and post-attack performance**

     Gauge the extent of the effectiveness of different mehtods.


### Model selection

<div align="justify">
   I opted to use models that deal with visual classification problems (what object is on the image) because the input (image) alteration and output are easy to observe. Classification problems vary in difficulty depending on the similarity between the target classes (eg: banana or fish, fish A or fish B). Naturally there are more and less complex models for different kinds of problems. 
 <br>
  I wanted to see the effects of my planned mischief at multiple levels of complexity, so I chose a few models that are drastically different but were desinged to handle the same type of problems.
</div>

<br>
<div align="justify">
  <b>Simple model:</b>
   <br>
  This one has no  distict official name. It is used for illustrating certain functionalities in coding tutorials. It has relatively few layers and a simple dataflow. It consists of 2 convolutional layers and 2 fully connected ones. 
</div>

<br>
<div align="justify">
   <b>Complex model:</b>
   <br>
   I've used Resnet18 which has a much more convoluted architecture. It is 18 layers deep (as opposed to the 4 layers of the simple model) and has a few tricks up its sleeve such as non-linear data flow (data is fed forward at certain layers skipping a couple). Since these deep networks (lot of layers) come with a large number of parameters I opted not to train it from scratch but adapt a pretrained model for my purposes. 
</div>

### Training

<div align="justify">
   I wanted to try misleading the models multi-class and binary classification problems. The multiclass variety means there are more than 2 categories of things the model should put the given input into. Binary classification deals with exaclty 2 output classes. 
</div>
<br>


 **In short I had these models:**

1. **Simple model binary**

3. **Simple model for 10 classes**

4. **Resnet18 binary**

5. **Resnet18 for 10 classes**



### Attack methods

<div align="justify">
   Several methods exist to attack neural networks and they can be grouped multiple ways. The ones I'm interested in are non-targeted and targeted methods. In a classification setting non-targeted means the ouput should be anything but the correct output. In the targeted case we want the output to be a specific class defined by us. 
</div>

   <br>
   I decided to use reliable and relatively simple attack methods such as:

   1. **FGSM (Fast Gradient Sign Method)**

        In classification problem during trainig the model we optimize the model to fit the groups/classes in our data. In this attack we try to interfere with this principle by modifying the input image such that the true class will seem less probable than everything else.
      <br>
      We can achieve this by adding noise to our input (image) in a specific way. First we make a prediciton with the original image, then adjust the image by addign noise and then making a second prediction to see if the model has misclassified the noisy image. The noise is calculated by the following formula:
      <br>
      
      X<sub>adv</sub> = X<sub>original</sub> +  ϵ * sign (∇<sub>X</sub> J(X,Y<sub>true</sub>))
      <br>
      
      Where:
      <br>
      
      X: input
      X<sub>original</sub>: adversarial input
      
      Y<sub>true</sub>: correct/true class
      
      ϵ: magnitude/strength of perturbation (added noise)
      
      (∇<sub>X</sub> J(X,Y<sub>true</sub>)): gradien of loss function used (for the input X)
      
   3. **One-step target class**

      We can wiev this attack as a modified FGSM where we do not minimize the likelyhood of the true class but maximize the likelyhood of an adversarial one. We go through the same steps as with FGSM but the formula for perturbation is slightly different:
      <br>
      
      X<sub>adv</sub> = X<sub>original</sub> -  ϵ * sign (∇<sub>X</sub> J(X,Y<sub>true</sub>))
      <br>
      (Instead of addig noise, we substract it from the original input image)

In most cases attack such as FGSM are intended to be undetectable by the human eye, so if in order for an image to be misclassified there is clearle visible perturbation the attack may not be considered succesfull.
      

### Comparison

<div align="justify">
</div>


