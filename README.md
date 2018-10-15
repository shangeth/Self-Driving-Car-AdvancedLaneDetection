# Advanced Lane Detection


In this Repo, Deep Learning approach is going to be used instead of just Image Processing / video processing like [Self-Driving-Car-LaneDetection](https://github.com/shangeth/Self-Driving-Car-LaneDetection)

![](https://github.com/shangeth/Self-Driving-Car-LaneDetection/raw/master/img.png)

# Datasets
A opensourced car driving lane image datasets is used for training the model.

# Model
Ofcourse ConvNets, But implementing this paper on [Fully Convolutional Networks
for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf]).

The basic idea behind Fully ConvNets is all the layers are Convolution layers instead of Fully connected layer in the end.

1. X is the image of the lane taken from the car's camera like this
![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Strada_Provinciale_BS_510_Sebina_Orientale.jpg/1200px-Strada_Provinciale_BS_510_Sebina_Orientale.jpg)

2. y is the coefficients of the 2nd degree polynomial used to fit the lane on both sides. y = [a1,b1,c1,a2,b2,c2]
ax^2+bx+c=0 fits the lanes.


