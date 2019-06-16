# GrabAIChallenge

Computer Vision on Predicting Car Feature
How might we automate the process of recognizing the details of the vehicles from images, including make and model?

## Problem Statement
Given a dataset of distinct car images, can you automatically recognize the car model and make?

## Dataset
We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.


1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/0.jpg)  | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/1.jpg) | ![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/2.jpg)|![image](https://github.com/AnggaPradiktas/GrabAIChallenge/blob/master/results_img/3.jpg) |
|Ford Mustang Convertible 2007', 'prob': '0.9883'|'Audi TT RS Coupe 2012', 'prob': '0.8998'|'Audi S5 Coupe 2012', 'prob': '0.9446'|'Aston Martin V8 Vantage Convertible 2012', 'prob': '0.9771'|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/6_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/7_out.png) |
|BMW 1 Series Coupe 2012, prob: 0.9948|Suzuki Aerio Sedan 2007, prob: 0.9982|Ford Mustang Convertible 2007, prob: 1.0|BMW 1 Series Convertible 2012, prob: 1.0|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/9_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/10_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/11_out.png)|
|Mitsubishi Lancer Sedan 2012, prob: 0.4401|Cadillac CTS-V Sedan 2012, prob: 0.9801|Chevrolet Traverse SUV 2012, prob: 0.9999|Bentley Continental GT Coupe 2012, prob: 0.9953|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/12_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/13_out.png)| ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/14_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/15_out.png)|
|Nissan Juke Hatchback 2012, prob: 0.9935|Chevrolet TrailBlazer SS 2009, prob: 0.987|Hyundai Accent Sedan 2012, prob: 0.9826|Ford Fiesta Sedan 2012, prob: 0.6502|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/17_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/19_out.png)|
|Acura TL Sedan 2012, prob: 0.9999|Aston Martin V8 Vantage Coupe 2012, prob: 0.5487|Infiniti G Coupe IPL 2012, prob: 0.2621|Ford F-150 Regular Cab 2012, prob: 0.9995|
