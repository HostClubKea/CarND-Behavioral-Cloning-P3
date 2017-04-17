#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/out.png "Out of track"
[image2]: ./examples/distribution.png "Original distribution"
[image3]: ./examples/distribution_adv.png "Improved distribution"
[image4]: ./examples/samples.png "Samples"
[image5]: ./examples/brightness.png "Brightness"
[image6]: ./examples/shadows.png "Shadows"
[image7]: ./examples/nvidia.png "Nvidia"
[image8]: ./examples/crop.png "Crop"
[image9]: ./examples/resize.png "Resize"


###Model Architecture and Training Strategy

####1. Solution Design Approach

I've started with nvidia architecture, as it proof to fit well for the task. Later I used more simple version based on my previous project (3 convolutional layers and 3 flatout layers) and while it worked well on track one, I switch back to Nvidia architecture in attempts to generalize network to work on second unseen track. 


As I havily used dataset augmentation and dropouts from the beginning so my training validation loss was always bigger then validation loss, and this difference was even bigger on simpler model which was also the reason to go back to Nvidia model. 

As I noticed in previous project playing with models many different architecture can give you almost same result, and more difference gives you amount of training data. So in this project I concentrated more on data augmentation and not on searching best network architecture. Also I notice that bad data can harm your model, so I was recording my data by small chuncks and test their combinations. For example I recorded extrim recovery actions, when car is perpendicular the road and try to return to the center, whyle this data helped a little bit with track2 it gave some unpredictable results on track1 so I ended up not using this one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Also vehicle was able to drive through old track 2 never saw data from it.

####2. Final Model Architecture

I used Nvidia's architecture with small changes: 

1. The first layer is 3 1X1 filters, this has the effect of transforming the color space of the images.
2. Used dropouts layer for better generalization

Base Nvidia network:

![alt text][image7]


My architecture and pramaters are(model.py lines 191-227):

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     

====================================================================================================

lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 3)     12          lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 64, 64, 3)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 32, 24)    1824        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 32, 24)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 31, 24)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 16, 16, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 16, 16, 36)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 15, 15, 36)    0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 8, 8, 48)      43248       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 8, 8, 48)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 7, 7, 48)      0           activation_4[0][0]               
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 7, 7, 64)      27712       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 7, 7, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 6, 6, 64)      0           activation_5[0][0]               
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 6, 6, 64)      36928       maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 6, 6, 64)      0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 5, 5, 64)      0           activation_6[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1600)          0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1863564     flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           activation_7[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           activation_8[0][0]               
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           activation_9[0][0]               
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          activation_10[0][0]    
          
====================================================================================================

Total params: 2,116,995
Trainable params: 2,116,995
Non-trainable params: 0
____________________________________________________________________________________________________


####3. Creation of the Training Set & Training Process

Previous project have showed me that big and balanced dataset is very import to successfully solve the task. In addition to Udacity dataset I collect 6 forward and 6 backward laps. With this data I was able almost finish first track, the only problem was near the finish where car suddenly turn to the left and go into the lake (there was similar looking place at old track 2 where car also wanted to go off the road)

![alt text][image1]

Then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle which helped my model to recover if it miss center at some point. That helped model 
 I ended up with 24582 samples (each contains 3 images). But distribution wasn't very nice, 0 steering was dominating

![alt text][image2]

I used simple algorithm to remove some of 0 steering examples if they were in sequence(My intuition was that they would provide the same image). After that my dataset contained only 12670 samples

![alt text][image3]

Here is an example of collected images:

![alt text][image4]

To get additional samples I've used:

1. randomly getting left or right camera sample with changing steering value (+-0.25)

1. increase or decrease brightness of the image

1. add shadows on the image

1. flip some samples and angles 

Example of changed brightness:

![alt text][image5]

Example of shadows:

![alt text][image6]

Augmented samples was generated on the fly for each batch.

After augmentation I crop images to size 32*128 as we don't need most of the top and some bottom part:

![alt text][image8]

Then I resize images to 64*64, this works better then passing 32*128 images to network:

![alt text][image9]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Most of the time I use 10 epochs to train model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Optimization

Training speed is very important for successfull finish of the project. So at the beginning I was looking for increasing speed of training. I didn't want to pregenerate augmented data as I changed algorithms of augmentation and data I used for training network, but augmentation took quite a lot of time - 1 sec for a batch and about 90 sec for epoch. To increase speed of generating augmented samples I've used **multiprocessing**, it decreased time needed to augment samples for epoch from to 35 sec, batch to 0.4 sec.

```
with Pool(3) as p:
    results = p.starmap(augment_sample, zip(batch_indices, repeat(data), repeat(augment)))
```python


####5. Results

After training on data from first track model can drive on first track and old second track. (see video.mp4 and video2old.mp4) 

I wasn't able to make car drive on new second track based only on first track data, the best result I had is passing about 6 first turns and then car stuck. 