# DCGAN-on-Anime-Faces

The respository contains the code for the Deep Convolutional Generative Adversarial Network (DCGAN) trained on the anime faces dataset.  
  
Blog post: https://idiotdeveloper.com/dcgan-implementing-deep-convolutional-generative-adversarial-network-in-tensorflow/
<br/>
Video explaination: [https://youtu.be/rJBp2L-JM_g](https://youtu.be/rJBp2L-JM_g)
  
There are two files:
- gan.py: It is used to train the DCGAN on the anime faces dataset.
- test.py: This file loads the saved generator model and generates some fake samples and save them with the name fake.png

## Dataset
Download the dataset and put all the images in the data folder.<br/>
Link: [https://www.kaggle.com/soumikrakshit/anime-faces](https://www.kaggle.com/soumikrakshit/anime-faces)

## Training samples
| ![Image at epoch 1](samples/generated_plot_epoch-1.png) |
|:--:| 
| *Image at epoch 1* |

| ![Image at epoch 1](samples/generated_plot_epoch-60.png) |
|:--:| 
| *Image at epoch 60* |


## Fake samples generated after training
| ![Fake sample](fake.png) |
|:--:| 
