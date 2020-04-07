# Behavioral Cloning documentation


### General discussion

I began by setting up infrastructure. That means, setting a maximally MLP with
three linear layers, a generator which makes use of center, left and right
camera images. Afterwards I dealt with color conversion, i.e. BGR/RGB.
Next I built a model using cropping, convolutional layers, dropout layers which
help dealing with image information and linear layers which deal with decision
making. The respective inductive bias, i.e. local vs global bias,
constitutes the theoretic idea.

## Model description

The following picture shows the general architecture.

![Model summary](model_description.png)

First we normalize and crop, according to the recommendations given in the
task description.
As we're working on images, we chose convolutions as a fundamental building block.
We're using a stride of (2, 2) and kernel sizes of 19, 32 and 64 to maximize
information.
In the end we use three linear layers to aggregate the information and perform
the regression using MSE loss.
We use 4 Dropout layers with a dropout factor of 0.3 to fight overfitting.

### Training Data

We used the supplied training data mixed with additional data augmentation.
No extra training data was created.
In general, each example consists of three images taken from the center, the left
and the right side of the car. We correct the angle by 0.2 for each side in order
to achieve some randomization. which will support dealing with overfitting.
Here are examples of such images:
