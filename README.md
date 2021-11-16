# CV_logo_detection

## Fine-Tuning with Keras

#### 1. Data
* Resize
* Standardizing/Normalization
* Data augumentation: Logos might not be symmetric, consider rotation/resize than overturn

#### 2. Build the Model
* Base model/pre-trained model selection: 
![image](https://user-images.githubusercontent.com/83235873/138552330-dc31be07-ec87-4824-bd16-7f79a926258e.png)

* Complie new layers, rescale-basemodel-pooling-dropout-dense 

#### 3. Train and Fine-Tuning
* try different top-layer inclusion


# ROBOFLOW
By Domenique: I registered using my institutional mail and it gives me the possibility to upload up to 10k images.

#1. In a new LogoDet project, Upload the annot_train.csv file 

#2. Upload only the train folder with images (coordinate with the others: upload only the images that were not already processed by someone else. I put the first 10k of train). Choose the option 100% training:
 
 ![image](https://user-images.githubusercontent.com/51834820/142015508-5b486c67-d10c-4b01-9b61-575a90167cfa.png)

#3. Wait for the infinite-time-consuming upload: for me, 9851 images out of 10k were uploaded as the remaining ones did not have a correspondence in the csv file.

#4. Generate a new version of the dataset, including the following pre-processing and augmentation steps:

![image](https://user-images.githubusercontent.com/51834820/142015543-e3021ab9-6960-4f63-84c0-eb47f6727326.png)
 
#5. Click on Generate and wait. Give a name as: first 10k, second 10k augmented, etc…

#6. Download the dataset choosing the most convenient format:
(For me, Tensorflow Object Detection would lead to restart my pc over and over so I still did not discover what will be the format)

 ![image](https://user-images.githubusercontent.com/51834820/142015589-147d7dcd-c1d0-4558-aa90-593fadd57ba7.png)


