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
