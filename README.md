# GrabAIChallenge

Computer Vision on Predicting Car Feature
How might we automate the process of recognizing the details of the vehicles from images, including make and model?

## Problem Statement
Given a dataset of distinct car images, can you automatically recognize the car model and make?

## Dataset
We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

## Environment
I simply run all my code using Google Colab, here's the link to my drive and Colab https://drive.google.com/file/d/1NodsfZ_wNRdSE0UkpSMYFUiyQs0NNpye/view?usp=sharing


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



