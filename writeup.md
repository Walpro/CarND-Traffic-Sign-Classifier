 
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

the selected architecture I used the Lenet5 convulutional neural network 
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




The first architecture I used was a LENET convulutional neural network with an input depth of 3, it was at first selected because of its ability to learn from images by capturing the caractristics of them through different analysis and hints deduced in the learning process from each layer.
The initial architecture was complex due to the input depth which confused the model with images color and made it not able to reach the desired accuracy, the model was over to the training dataset and an important accuracy difference was observed between the validation and testing dataset.

To avoid letting the model try to learn from the image colors which is not an important propriety of road signs images, the model depth was reduced to 1 which reduced as well the complexity of the model and the model layers sizes are also adjusted.

I tried different learning rates to tune the importance of each learning step.
I also started with reduced number of epochs to have a first idea of the model accuracy and when the first values were promising I increased the values and evaluated the outcomes.

an Adam optimizer was used and the final learning rate is  0.001 and the number of epochs is 70 .

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
The accuracy on the images from the web is 40% while it was 92.7 on the testing set thus  the model seems to be overfitting
to the training, testing and validation images, it shows also that the used dataset is not enough and does not capture all the possible variations of sign road images.

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

