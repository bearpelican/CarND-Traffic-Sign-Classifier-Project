# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./examples/class_hist.png "Class Histogram"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/external_images/orig/12.jpg "Traffic Sign 1"
[image5]: ./examples/external_images/orig/15.jpg "Traffic Sign 2"
[image6]: ./examples/external_images/orig/1.jpg "Traffic Sign 3"
[image7]: ./examples/external_images/orig/22.jpg "Traffic Sign 4"
[image8]: ./examples/external_images/orig/25.jpg "Traffic Sign 5"
[image9]: ./examples/external_images/orig/30.jpg "Traffic Sign 6"
[image10]: ./examples/external_images/orig/38.jpg "Traffic Sign 7"
[image11]: ./examples/external_images/orig/40.jpg "Traffic Sign 8"


[image12]: ./examples/external_images/cropped/12.jpg "Cropped Traffic Sign 1"
[image13]: ./examples/external_images/cropped/15.jpg "Cropped Traffic Sign 2"
[image14]: ./examples/external_images/cropped/1.jpg "Cropped Traffic Sign 3"
[image15]: ./examples/external_images/cropped/22.jpg "Cropped Traffic Sign 4"
[image16]: ./examples/external_images/cropped/25.jpg "Cropped Traffic Sign 5"
[image17]: ./examples/external_images/cropped/30.jpg "Cropped Traffic Sign 6"
[image18]: ./examples/external_images/cropped/38.jpg "Cropped Traffic Sign 7"
[image19]: ./examples/external_images/cropped/40.jpg "Cropped Traffic Sign 8"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bearpelican/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_VGG16.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* Resize: images were resized to 32x32
* Mean normalization - (x - mean) / std

I did not grayscale the images as I believe the color would help determine what type of traffic sign it was
I also did not use data augmentation as I did not have enough time. Though I would like to add it in the future


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        		        			| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   					        		| 
| Convolutional Block 1	|:-----------------------------------------------------:|
| Convolution 2x2     	| 32 depth, 1x1 stride, same padding, outputs 32x32x32	|
| RELU					|												        |
| Batch Normalization	|												        |
| Convolution 2x2     	| 64 depth, 1x1 stride, same padding, outputs 32x32x64	|
| RELU					|												        |
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				        |
| Batch Normalization	|												        |
| Dropout           	| Keep prob = 0.35								        |
| Convolutional Block 2	|:-----------------------------------------------------:|
| Convolution 2x2     	| 32 depth, 1x1 stride, same padding, outputs 16x16x32	|
| RELU					|												        |
| Batch Normalization	|												        |
| Convolution 2x2     	| 64 depth, 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												        |
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				            |
| Dense layers          |:-----------------------------------------------------:|
| Batch Normalization	|												        |
| Flatten           	|												        |
| Fully Connected   	| size = 256									        |
| Batch Normalization	|												        |
| Dropout           	| Keep prob = 0.35								        |
| Fully Connected   	| size = 512									        |
| Batch Normalization	|												        |
| Dropout           	| Keep prob = 0.35								        |
| Softmax            	| size = 43		    							|

 
I tried to model it off a simpler version of VGG network with dropout and batch normalization

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I ran 20 epochs with a batch size of 256. I used a starting learning rate .001 with exponential decay. The learning rate was cut in half every ~2 batches run

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.977865 
* test set accuracy of 0.978441

If an iterative approach was chosen:

Lenet
* The first architecture chosen was taken from the lab based on LeNet [project code](https://github.com/bearpelican/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* This got me to about 95%.

VGG
* For fun, just wanted to see if VGG could perform better. It has smaller convolution filters and multiple conv layers before each max pool.
* Initially, it was way overfitting for the training data with very low accuracy. As a result, I reduced the dense layer sizes and the depth of the convolutional layers.
* I also added batch normalization and dropout as a regularizer to prevent overfitting.
* Set the initial learning rate to .001 through trial and error. I ran epochs and decayed the learning rate until the training accuracy reached 100%.


* Results: I'm still overfitting by quite a bit. Training accuracy is 100% while validation and test hover around 97.7%.
* Augmenting my input data could help with overfitting. Perhaps adding more regularization techniques will help. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

These images are a bit difficult to classify because they are not cropped to show the sign only. They also are not square, so become distorted when resized to 32x32


#### 1b. Manual cropping

The accuracy for the above images is super low - 2 out of 8 correct. 

Predictions seemed to work much better when I manually cropped the images
![alt text][image12] ![alt text][image13] ![alt text][image14] 
![alt text][image15] ![alt text][image16] ![alt text][image17]
![alt text][image18] ![alt text][image19]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right      		| Keep Right   									| 
| Roundabout mandatory	| Roundabout mandatory   						| 
| Beware Ice/Snow  		| Right-of-Way   						    	| 
| No Vehicles      		| No Vehicles   								| 
| Priority road     	| Priority road 								|
| Bumpy road			| Bumpy road									|
| Road work             | Road work 					 				|
| Speed limit			| Speed limit         							|


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5. The one that it messed up on - Beware Ice/Snow has the same shape as the Right-of-way sign.

The accuracy for the uncropped versions was 2 out of 8. Pretty poor accuracy

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 34th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0             		| Keep Right   									| 
| 1.0               	| Roundabout mandatory   						| 
| 1.0             		| Right-of-Way   						    	| 
| 1.0             		| No Vehicles   								| 
| 1.0                	| Priority road 								|
| 1.0       			| Bumpy road									|
| 1.0                   | Road work 					 				|
| 1.0       			| Speed limit         							|


I'm not sure why the probabilities are so skewed towards 1 or 0. It could be that I'm overfitting too much.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the convolutional layers, it looks like there are certain filters that activate when it sees different shapes or colors.

