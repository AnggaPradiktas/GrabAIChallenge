# GrabAIChallenge

Computer Vision on Predicting Car Feature
How might we automate the process of recognizing the details of the vehicles from images, including make and model?

## Problem Statement
Given a dataset of distinct car images, can you automatically recognize the car model and make?

## Dataset
We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

Training images can be downloaded <a href=http://imagenet.stanford.edu/internal/car196/cars_train.tgz>here</a>. 

Testing images can be downloaded <a href=http://imagenet.stanford.edu/internal/car196/cars_test.tgz>here</a>. 

A devkit, including class labels for training images and bounding boxes for all images, can be downloaded <a href=https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz>here</a>.

### ImageNet Pretrained Model for fine-tuning
Download fine-tuning imagenet pre-trained model for theano and tensorflow backend <a href = https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6>here.</a>

## Code

### Preprocessing

First download all the data needed from the link above and then split and assign them into train, test, and valid. You can see the preprocessing code in <a href = https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/img_annot_preprocessing.ipynb>img_annot_preprocessing.ipynb</a>

### Batch Normalization
Then create a custom layer for BatchNormalization. I am using Resnet-152 as a fine-tuning method to do car recognition. Learns a set of weights and biases used for scaling the input data. You can access the layer code here <a href = https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/scale_layer.py>scale_layer.py</a>

### Resnet 152 Model
Then I created Resnet-152 pre-trained model. Model Schema and layer naming follow that of the original Caffe implementation https://github.com/KaimingHe/deep-residual-networks. I put the code in <a href=https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/training_keras.py>training_keras.py</a> (.py) instead of Jupyter so that I can import it easily for my next code.

### Train the Data
And the I trained the data using the model to get the best accuracy. After some loopings I decided to use [model.46-0.99.hdf5](https://drive.google.com/file/d/10JIhvdwyitwrE1tURktU39b54_KxpUPq/view?usp=sharing) model. You can find the code here <a href=https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/train.ipynb>train.ipynb</a>.

## Environment
I run almost my entire code using Google Colab, here's the link to my drive and Colab https://drive.google.com/file/d/1NodsfZ_wNRdSE0UkpSMYFUiyQs0NNpye/view?usp=sharing


## demo_test.ipynb results

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/0.jpg)  | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/1.jpg) | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/2.jpg)|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/3.jpg) |
|Ford Mustang Convertible 2007', 'prob': '0.9883'|'Audi TT RS Coupe 2012', 'prob': '0.8998'|'Audi S5 Coupe 2012', 'prob': '0.9446'|'Aston Martin V8 Vantage Convertible 2012', 'prob': '0.9771'|
|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/4.jpg)  | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/5.jpg) | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/6.jpg)|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/7.jpg) |
|'Audi RS 4 Convertible 2008', 'prob': '0.9997'|'BMW Z4 Convertible 2012', 'prob': '0.9999'|'Fisker Karma Sedan 2012', 'prob': '1.0'|'Dodge Challenger SRT8 2011', 'prob': '1.0'|
|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/8.jpg)  | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/9.jpg) | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/10.jpg)|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/11.jpg) |
|'Lamborghini Diablo Coupe 2001', 'prob': '0.9612'|'Nissan Leaf Hatchback 2012', 'prob': '1.0'|'Land Rover Range Rover SUV 2012', 'prob': '0.9999'|'Jeep Liberty SUV 2012', 'prob': '0.9961'|
|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/12.jpg)  | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/13.jpg) | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/14.jpg)|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/15.jpg) |
|'Buick Regal GS 2012', 'prob': '0.9996'|'GMC Savana Van 2012', 'prob': '0.4698'|'Bugatti Veyron 16.4 Coupe 2009', 'prob': '0.9959'|'BMW 3 Series Sedan 2012', 'prob': '0.9999'|
|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/16.jpg)  | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/17.jpg) | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/18.jpg)|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/19.jpg) |
|'Daewoo Nubira Wagon 2002', 'prob': '0.9821'|'Ford F-150 Regular Cab 2007', 'prob': '0.9995'|'smart fortwo Convertible 2012', 'prob': '0.9995'|'Chevrolet Impala Sedan 2007', 'prob': '0.9991'|



