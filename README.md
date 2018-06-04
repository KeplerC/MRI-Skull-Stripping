# MRI Skull Stripping

Kaiyuan Chen(chenkaiyuan@ucla.edu)

Jingyue Shen(brianshen@ucla.edu)

More documents about this project is on http://kychen.xyz/2018/06/02/MRI-2018/. 

### Introduction 

Please refer to
http://kychen.xyz/2018/05/16/skullreview-2018/
for a comprehensive review on
* what is MRI
* why do we need skull stripping
* how does this work
* a review of recent works
* bridging skull stripping to machine learning
* a plan of this work

### Outline

1) The baseline model should be sklearn-based supervised learning. These models are very easy to implement, so I will try out different supervised learning approach like random forest, decision trees and other linear models to see which one works better. 

2) Autoencoder. We plan to stack CNNs and calculate RMSE on recovered image in Tensorflow. 

3) Other Compressed Sensing approaches. Other compressed sensing generative models like VAE. 

### I/O
#### Feature Selection

For sklearn models, we choose 
P(this pixel should be removed | (position x, position y), color, surrounding pixels as a local patch) 
and currently, because of the limitation of my computer memory, the patch size is 4 * 4. One can update it by changing m in my source code. 

For CNN Autoencoder model, we feed the entire image to CNN, since we think dealing with matrices of size 256x256 should work fine on a laptop computer. Due to the limited amount of data (about 660 images of brains), we currently set batch size = 15
and epoch time = 100 for training. One can modify the configuration by changing the hyperparameters in code/config.py.

#### Preprocessing
For sklearn models,  we perform a comparison on local patch or other features.

For CNN model, we now simply divide each image's pixel value by 255. We are going to explore batch normalization technique later.

### Code Snippets 

For all baseline codes and experiments, you can go to a blog post(http://kychen.xyz/2018/05/16/jpskull-2018/) or the **jupyter notebook** in the ./code. 

For CNN Autoencoder code, you can find it in the ./code folder in this repository.

### Dates 

2018-5-28 Finished Report

2018-5-26 Finished CNN Autoencoder models

2018-5-16 Finished up other sklearn models in jupyternotebook

2018-5-13 Finished Baseline model(random forest)

2018-5-12 Creating Starter codes 

2018-5-4 Writing Introduction/Literature Reviews 

2018-5-1 Selecting Topics 
