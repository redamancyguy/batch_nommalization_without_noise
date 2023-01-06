## This is a method to solve the noise problem of simple batch normalization


## problem
* The standardScalar of x and y can make the study process better than without scalar,
but a bad batch normalization will make the study good at one place but 
very bad others place if the samples of the batch is not good (distributed unevenly).
if the sample is bad,then the what the process studied will be not fitted in the next batch
and is will be changed largely in the next batch.This is not what we want.
We want a function that is fitted for the whole section with is distribution.

## method
* To solve this problem,when we get in a batch to train the model,we first get the
standard parameters of x and y called x_mean,y_mean,x_sigma,y_sigma
* Before training we eliminate the influence of the standardScaler to model,we scrape the influence from the model about the new parameter for standardization
* After training,we integrate parameters into the model.

## full process

![IMG_0170.PNG](..%2FIMG_0170.PNG)