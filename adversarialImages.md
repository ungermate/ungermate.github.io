# Adversarial Images

## Intro
<div align="justify">
It feels like we are living in the future as AI (in some form) becomes a part of more and more everyday events. People can't help but wonder where this might lead and how things could go wrong. This got me thinking: how can I make things go wrong? How wrong?
<br>

Whith these questions in mind the goal of this project is to explore some of the vulnerabilities of neural networks used for visual image classification.
</div>

## Strategy

1. **Select model to attack**
   
     Find model designed for a image classification task.

3. **Set the model up, train it**

     Get a sense of how it would normally operate for later reference.

4. **Attack model**

     Create ways that mess with the model.

5. **Compare pre- and post-attack performance**

     Gauge the extent of the effectiveness of different mehtods.


## Model selection

<div align="justify">
   I opted to use a model that deals with visual classification problems (what object is on the image) because the input (image) alteration and output are easy to observe.
 <br>
  I wanted to make sure that any change in performance of the model was due to my mischief and not just its limited capacity to learn, so I picked Resnet18, a relatively complex one. 
</div>

<br>
<div align="justify">
  It is 18 layers deep and has a few tricks up its sleeve such as non-linear data flow (data is fed forward at certain layers skipping a couple). Since these deep networks (lots of layers) come with a large number of parameters I opted not to train it from scratch but adapt a pretrained model for my purposes. 
</div>

## Training

<div align="justify">
   I wanted to try misleading the model in a binary classification problem (2 classes) and multiclass problem (>2 classes). I used transfer learning to create custom heads for the pretrained model. 
</div>

I worked with these models:

1. Resnet18 - binary classification
   
     Differentiates between Chihuahuas and muffins (in some cases it's harder than you'd think) based on <a href="https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification">this dataset</a> with around 97% accuracy.

2. Resnet18 - multiclass
   
      Can label images belonging to 10 classes of animals (horse, elephant, spider...) based on <a href="https://www.kaggle.com/datasets/alessiocorrado99/animals10">this dataset</a> with an accuracy of 90%.

<br>


## Attack methods

<div align="justify">
   I was interested in non-targeted and targeted methods. In a classification setting non-targeted means the output should be anything but the correct output. In the targeted case we want the output to be a specific class defined by us. 
</div>

   <br>

   1. **FGSM (Fast Gradient Sign Method)**
         <div align="justify">
        In classification problem during training the model we optimise the model to fit the groups/classes in our data. In this attack we try to interfere with this principle by modifying the input image such that the true class will seem less probable than everything else.
      We can achieve this by adding noise to our input (image) in a specific way. First we make a prediction with the original image, then adjust the image by adding noise and then making a second prediction to see if the model has misclassified the noisy image. 
         </div>
      The noise is calculated by the following formula:
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

      <center>
      <img width="1489" height="300" src="images/adversarial_images/non_targeted_resnet/flow.png">
      </center>
      <br>


   
   3. **One-step target class**
      <div align="justify">
      We can view this attack as a modified FGSM where we do not minimise the likelihood of the true class but maximise the likelihood of an adversarial one. We go through the same steps as with FGSM but the formula for perturbation is slightly different:
      </div>
         <br>
      
      X<sub>adv</sub> = X<sub>original</sub> -  ϵ * sign (∇<sub>X</sub> J(X,Y<sub>true</sub>))
      <br>
      
      (Instead of addig noise, we substract it from the original input image)

In most cases attacks such as FGSM are intended to be hard to detect or undetectable by the human eye. If there is clearly visible perturbation the attack may not be considered successful.

      

## Results

### Non-targeted

#### Resnet18 binary

The following series of images show how the model output changes due to the amount of perturbation (epsilon) added. The title of each image has the following structure: [original label] -> [label of modified image] 

Epsilon = 0 (control)
<center>
<img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/binary/eps_0.png">
</center>

<div align="center">
<em>Unmodified control images. Notice how the model is able to tell the last image also contains dogs.  </em>
</div>
<br>


Epsilon = 0.05
<center>
<img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/binary/eps_005.png">
</center>


<div align="center">
<em> With only a small amount of added perturbation the images look mostly normal, however in these cases the change was enough for the model to assign incorrect labels.  </em>
</div>
<br>


Epsilon = 0.1
<center>
<img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/binary/eps_01.png">
</center>

<div align="center">
<em>With a bit more perturbation the images look rather grainy, low quality. In some cases the modification is apparent. </em>
</div>
<br>

Epsilon = 0.3
<center>
<img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/binary/eps_03.png">
</center>

<div align="center">
<em>These are not fooling anyone (except the model), they are clearly modified</em>
</div>
<br>

<div align="justify">
Notice how in all of the above examples the attack only worked with a muffin image. Maybe these cookies are more dog-like than the dogs were muffin-like. Or perhaps a dog's facial features are more characteristic to its species than a muffin's general shape (model is less confident if something is a muffin originally).
<br>
<br>
The next image summarises the models accuracy in relation to the amount of added perturbation.
</div>


<br>
<center>
   <img width="460" height="475" src="images/adversarial_images/non_targeted_resnet/binary/resnet_no_target_dogs_acc_eps.png">
</center>

<div align="center">
<em>Model accuracy for each amount of added perturbation</em>
</div>
<br>

<div align="justify">
The graph clearly shows the attack worked. Accuracy scores almost halved with larger epsilon values. It's interesting to see the accuracy scores plateau after epsilon reaches 0.15. This suggests that the amount of added perturbation is more than enough after this point and we can get away with far less for similar results. 
</div>



#### Resnet8 multiclass

Again, some images to illustrate the effects of perturbation.

Epsilon = 0
<center>
   <img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/multiclass/eps_0.png">
</center>

<div align="center">
<em>Control group, model makes correct predictions</em>
</div>


Epsilon = 0.05
<center>
   <img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/multiclass/eps_005.png">
</center>


<div align="center">
<em>In some cases it's hard to tell whether the image has been modified or not.</em>
</div>

Epsilon = 0.3
<center>
   <img width="1489" height="230" src="images/adversarial_images/non_targeted_resnet/multiclass/eps_03.png">
</center>


<div align="center">
<em>Obvious signs of modification.</em>
</div>
<br>

<center>
   <img width="460" height="475" src="images/adversarial_images/non_targeted_resnet/multiclass/resnet_no_target_10_acc_eps.png">
</center>



<div align="center">
<em>Model accuracy as a function of added perturbation</em>
</div>
<br>

<div align="justify">
   Again we can see a sharp drop in accuracy after 5% added perturbation. This time we have not managed to hit a plateau, however it would most certainly come with a further 5-10% added perturbation as we are nearing the 10% accuracy mark (which would be expected with random guesses). 
</div>





### Targeted

#### Resnet18 binary

<div align="justify">
   Since in the non-targeted case I only managed to get muffin -> dog misclassifications, here I set the target class to be muffins to see if I can force it the other way. Unfortunately I was not able to do so, again all mistakes were of the muffing -> dog type. 
</div>

Epsilon = 0
<center>
   <img width="1489" height="230" src="images/adversarial_images/targeted_resnet/muffin_target/t_eps_0.png">
</center>

<div align="center">
<em>Control group, model makes correct predictions</em>
</div>

Epsilon = 0.3
<center>
   <img width="1489" height="230" src="images/adversarial_images/targeted_resnet/muffin_target/t_eps_03.png">
</center>


<div align="center">
<em>Images with most amount of perturbation. Still all labeled dogs.</em>
</div>
<br>

<center>
   <img width="460" height="475" src="images/adversarial_images/targeted_resnet/muffin_target/acc_vs_eps.png">
</center>

<div align="center">
<em>Similar curve as before. However since all observed misclassifications are muffin -> dog, it suggests after epsilon = 0.15 the model labels almost everything as a dog. Hence the drop in accuracy.</em>
</div>

#### Resnet18 multiclass

Epsilon = 0.05

<center>
   <img width="1489" height="230" src="images/adversarial_images/targeted_resnet/multiclass/chicken_target/eps_005.png">
</center>


<div align="center">
<em>5% added perturbation targeting the "chicken" class</em>
</div>

Epsilon = 0.3

<center>
   <img width="1489" height="230" src="images/adversarial_images/targeted_resnet/multiclass/chicken_target/eps_03.png">
</center>

<div align="center">
<em>30% added perturbation, also "chicken" target. Note that almost all misclassification by this point are falsely labelled as "butterfly". Even Though the target was missed, at least it seems somewhat consistent</em>
</div>
<br>
The accuracy-epsilon relationship is identical to the non-targeted case. 


I've also tried another targeted approach. In this one I used a dynamic target class selected by picking the least likely one from the model's initial prediction. While this seemed promising it did not deliver any better results than the previous method.  


### Conclusion

<div align="justify">
FGSM is a relatively simple and computationally inexpensive method. While it is not the most reliable one, it's definitely able to significantly drop model accuracies even with a small amount of perturbation. 
<br>
The targeted attack doesn't work exactly as I hoped however it seems effective as well but is not worth the extra hassle. Maybe a less covert method like adversary patches might work much better. 

</div>


### Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Utilities.Util import imshow, data_transforms, image_reverse_transform
from Models.CustomModels import SimpleNet_ft, ResNet18_stock
```

```python
def ResNet18_lp(num_classes: int):
    """
    Args:
        num_classes: number of output classes for the model
    Returns:
        ResNet18 model configured for linear probing (fixed feature extractor) with
        a custom head.
    """
    # pretrained model (model structure and weights)
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # replace the base model's last layer with our custom one
    # number of features being passed to the last (fully connected) layer of the base model has to be the
    # same as in base model (fc.in_features)
    # number of features we want the custom head (replaces the base model's fc) to output has to be
    # equal to num classes in our problem

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # freeze everything (weights and biases for layers cannot be updated)
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last layer (will be updated during training)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
```

```python
data_dir = "./data/animals"
t = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

image_datasets = datasets.ImageFolder(data_dir, transform=t)
smaller_dataset, _ = random_split(image_datasets, [0.1, 0.9])
dataloaders = torch.utils.data.DataLoader(smaller_dataset, batch_size=1, shuffle=True, num_workers=4)

dataset_sizes = len(dataloaders)*dataloaders.batch_size
class_names = image_datasets.classes
```

```python
# Define what device we are using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("../Pastry_or_dogs/trained_models/best_model_params_10_classes.pth")

# Initialize the network
model = model.to(device)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
```

```python
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, targeted:bool):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()


    # Create the perturbed image by adjusting each pixel of the input image
    if targeted:
        perturbed_image = image - epsilon*sign_data_grad
    else:
        perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# restores the tensors to their original scale
def denorm(batch, mean, std):
    """
    Convert a batch of tensors to their original format

    Args:
        batch: batch of normalized tensors
        mean: mean used for normalization
        std: standard deviation used for normalization

    Returns:
        batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
```

```python
def test( model, device, loader, epsilon , mean, std, target_label = -1):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # if there is a target
    if target_label != -1 and target_label != 'least_likely':
        adversary_label = torch.tensor([target_label]).to(device)
        # adversary_label = adversary_label.to(device)

    
    # Loop over all examples in test set
    for image, original_label in tqdm(loader):

        # Send the data and label to the device
        image, original_label = image.to(device), original_label.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        image.requires_grad = True

        # Forward pass the data through the model
        output = model(image)
        initial_prediction = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        prob = F.softmax(outputs, dim=1)
        least_likely_class = prob.squeeze().cpu().numpy().argmin()
        

        # If the initial prediction is wrong, don't bother attacking, just move on
        if initial_prediction.item() != original_label.item():
            continue
        
        # targeted
        if target_label != -1 and target_label != 'least_likely':
            # Calculate the loss for targeted attack
            loss = F.nll_loss(output,adversary_label)

        # least likely class
        elif target_label == 'least_likely':
            least_likely_class = torch.tensor([least_likely_class]).to(device)
            loss = F.nll_loss(output,least_likely_class)

        # non-targeted
        else:
            # Calculate the loss for non-targeted attack
            loss = F.nll_loss(output, original_label)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = image.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(image, mean, std)
        # data_denorm = torch.tensor(data_denorm)

        # Call FGSM Attack
        perturbed_image = fgsm_attack(data_denorm, epsilon, data_grad,target_label==-1)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_image)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_prediction = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_prediction.item() == original_label.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                og_img = data_denorm.squeeze().detach().cpu().numpy()
                adv_examples.append( (initial_prediction.item(), final_prediction.item(), adv_ex,og_img) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                og_img = data_denorm.squeeze().detach().cpu().numpy()
                adv_examples.append( (initial_prediction.item(), final_prediction.item(), adv_ex,og_img) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
```

```python
# amount of adjustments to the input data (amoount of pixel wise perturbation)
epsilons = [0, .05, .1, .15, .2, .25, .3]

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, dataloaders, eps, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],'least_likely')
    accuracies.append(acc)
    examples.append(ex)
```

```python

```

```python

```

```python

```

```python

```

```python

```
