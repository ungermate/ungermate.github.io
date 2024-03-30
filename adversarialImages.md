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
  There are many types of models we regularly employ for various purposes such as navigation (autonomous vehicles), marketing (video-ad pairing),
  validation (money transfer) and so on. A common feature of these usecases is they have to conform to our human needs. We handle an overbearing majority of information in visual form so the models also have to tackle the problem of making sense of what they "see". This is why I chose models that handle visual classification problems. 
  Also this way the output will be much easier to observ for us (humans) too. 
  <br>
  As there are a great number of problems models can solve, there are a number of models to do so with different levels of complexity. This generally means that more 
  complex models have a greater capacity for learning, they could "make sense" of more abstract concepts. 
  I wanted to see the effects of my planned mischief at multiple levels of complexity, so I chose a few models that are drastically different but were desinged to handle 
  the same type of problems.
  <br>
  <b>Simple model:</b>
   <br>
  This one has no  distict official name. It is used for illustrating certain functionalities in coding tutorials. It has relatively few layers and a simple dataflow. It consists of 2 convolutional layers and 2 fully connected ones. 

   <b>Complex model:</b>
   <br>
   I've used Resnet18 which has a much more convoluted architecture. It is 18 layers deep (as opposed to the 4 layers of the simple model) and has a few tricks up its sleeve such as non-linear data flow (data is fed forward at certain layers skipping a couple). Since these deep networks (lot of layers) come with a large number of parameters I opted not to train it from scratch but adapt a pretrained model for my purposes. 
</div>

### Training

<div align="justify">
</div>

### Attack methods

<div align="justify">
</div>

### Comparison

<div align="justify">
</div>


