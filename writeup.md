 
[//]: # (Image References)
[image1]:./color_gray.png
[image2]:./sign_2.jpg
[image3]:./sign_12.jpg
[image4]:./sign_20.jpg
[image5]:./sign_26.jpg
[image6]:./sign_31.jpg

**Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


**Rubric Points**

This document describes the Traffic Sign Recognition project, and here is a link to my [project code](https://github.com/Walpro/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

Data Set Summary & Exploration


I used the proprieties of the input data to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32X32X3
* The number of unique classes/labels in the data set is 43

Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because images color is not an important
factor in determining the nature of the road sign also this conversion helps to reduce
the complexity of the model.

Here is an example of a traffic sign image before and after grayscaling.

![color to gray_scale][image1]

I have also normalized the image data to reduce the fluctuation of the input
and to get more plausible values in the caculation of the gradiant.

I used the Lenet5 convulutional neural network 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs  28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,valid padding  outputs 14x14x6.				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs  10x10x16     									|
| Max pooling	      	| 2x2 stride,valid padding,  outputs 5x5x16.				|
| FLATTEN				|												|
| Fully connected		|      									|
|  RELU					|												|
|Fully connected		|      									|


To train the model, I used an Adam optimizer with a learning rate of  0.001 and 70 epochs.

I tried different learning rates and epochs values to reach the desired values.
I have started also with colored images and a depth 3 model and changed to gray scale and 
depth 1 model after not being able to reach 93% validation accuracy.

My final model results were:
* validation set accuracy of 93.5% 
* test set accuracy of 92.7%
 
**Test Model on New Images**

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The last image might be difficult to classify because it is turned around 
and not the whole sign shape is present.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| speed limt (50km/h)      		| speed limt (50km/h)   									| 
| Priority road     			| Priority road 									|
| Dangerous curve to the left					| No passing for vehicles over 3.5 metric tons											|
| Trafic signals      		| General caution					 				|
| Wild animals crossing			| speed limt (20km/h)      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a speed limt (50km/h) (probability of 1), and the image does contain a a speed limt (50km/h) . 

For the second image, the model is sure that this is a Priority road (probability of 0.99), and the image does contain a Priority road. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99%       			| Priority road								| 
|  7.46120668e-06    				| Ahead only									|
| 3.68073074e-06 					| Yield											|
| 1.23104476e-15      			| No passing		 				|
| 1.23104476e-15			    |  Trafic signals      							|

For the rest of the images the model is not anymore sure which raod sign the image contains, and this translates to the softmac probabilities for exemple for the dangerous curve to the left	image the five softmax probabilites were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.32850313e-01       			| No passing for vehicles over 3.5 metric tons									| 
| 2.67149687e-01   				| dangerous curve to the left							|
| 4.97967864e-16 					| Road work											|
| 2.19870795e-20      			| slippery road	 				|
| 1.24141159e-27	    |  right-of-weight at the next interstection      							|

