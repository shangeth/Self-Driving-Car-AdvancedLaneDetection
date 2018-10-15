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
<pre>
**Model Summary :**
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
batch_normalization_1 (Batch (None, 80, 160, 3)        12        
_________________________________________________________________
Conv1 (Conv2D)               (None, 78, 158, 8)        224       
_________________________________________________________________
Conv2 (Conv2D)               (None, 76, 156, 16)       1168      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 38, 78, 16)        0         
_________________________________________________________________
Conv3 (Conv2D)               (None, 36, 76, 16)        2320      
_________________________________________________________________
dropout_1 (Dropout)          (None, 36, 76, 16)        0         
_________________________________________________________________
Conv4 (Conv2D)               (None, 34, 74, 32)        4640      
_________________________________________________________________
dropout_2 (Dropout)          (None, 34, 74, 32)        0         
_________________________________________________________________
Conv5 (Conv2D)               (None, 32, 72, 32)        9248      
_________________________________________________________________
dropout_3 (Dropout)          (None, 32, 72, 32)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 36, 32)        0         
_________________________________________________________________
Conv6 (Conv2D)               (None, 14, 34, 64)        18496     
_________________________________________________________________
dropout_4 (Dropout)          (None, 14, 34, 64)        0         
_________________________________________________________________
Conv7 (Conv2D)               (None, 12, 32, 64)        36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 12, 32, 64)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 16, 64)         0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 12, 32, 64)        0         
_________________________________________________________________
Deconv1 (Conv2DTranspose)    (None, 14, 34, 64)        36928     
_________________________________________________________________
dropout_6 (Dropout)          (None, 14, 34, 64)        0         
_________________________________________________________________
Deconv2 (Conv2DTranspose)    (None, 16, 36, 64)        36928     
_________________________________________________________________
dropout_7 (Dropout)          (None, 16, 36, 64)        0         
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 32, 72, 64)        0         
_________________________________________________________________
Deconv3 (Conv2DTranspose)    (None, 34, 74, 32)        18464     
_________________________________________________________________
dropout_8 (Dropout)          (None, 34, 74, 32)        0         
_________________________________________________________________
Deconv4 (Conv2DTranspose)    (None, 36, 76, 32)        9248      
_________________________________________________________________
dropout_9 (Dropout)          (None, 36, 76, 32)        0         
_________________________________________________________________
Deconv5 (Conv2DTranspose)    (None, 38, 78, 16)        4624      
_________________________________________________________________
dropout_10 (Dropout)         (None, 38, 78, 16)        0         
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 76, 156, 16)       0         
_________________________________________________________________
Deconv6 (Conv2DTranspose)    (None, 78, 158, 16)       2320      
_________________________________________________________________
Final (Conv2DTranspose)      (None, 80, 160, 1)        145       
</pre>
