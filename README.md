# Deep Learning for Computer Vision - Logo Detection

Files:
- train images (80%) -> link
- test images (20%) -> link
- annot_train.csv -> yes
- annot_test.csv -> yes
- detecto_weights_xlogos.pth -> link
- detecto_x_logos.py -> yes
- predict_detecto_xlogos.py -> yes
- detecto_results_xlogos.csv -> yes (do we really want this?)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Welcome to the repository for Logo Detection using the Detecto model![^1] 

<a href="https://user-images.githubusercontent.com/81080301/144510250-f960b63c-ac2a-41db-b77c-f3d4d5a0f181.jpg"><img src="https://user-images.githubusercontent.com/81080301/144510250-f960b63c-ac2a-41db-b77c-f3d4d5a0f181.jpg" width="300" height="300"/></a>
<a href="https://user-images.githubusercontent.com/81080301/144510291-e2c3e970-7166-48d1-8bc9-49fba00c465c.jpg"><img src="https://user-images.githubusercontent.com/81080301/144510291-e2c3e970-7166-48d1-8bc9-49fba00c465c.jpg" width="400" height="300"/></a>
<a href="https://user-images.githubusercontent.com/81080301/144511248-b86bbd14-4e99-40a4-b8c3-e243792c5630.jpg"><img src="https://user-images.githubusercontent.com/81080301/144511248-b86bbd14-4e99-40a4-b8c3-e243792c5630.jpg" width="250" height="300"/></a>




### The Model - Detecto  :mag: 👀
[Detecto](https://detecto.readthedocs.io/en/latest/) is a Python package built on top of Pytorch that allows you to perform object detection and make inference on still images and videos. It creates and runs a pre-trained RCNN ResNet-50 FPN. <br />	

To install Detecto, run the following command: <br />
	
```
$pip install detecto
```
Note that installing with pip should automatically download all the required module versions, however if there are still issues, manually download the dependencies from [requirements.txt](https://github.com/xx-liu-2244/CV_logo_detection/blob/main/requirements.txt).<br />
Moreover, in order to run Detecto, there are also more technical requirements, such as: <br />
* The annotations file in .csv has to be structured with the following order and heading names:  <br />
	>filename, height, width, class, xmin, ymin, xmax, ymax, image_id<br />
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where `image_id` are unique integers in ascending order starting from 0.
* The model must run in GPU <br />
	
### Train Test Split  
Train and test images have been split with a 80/20 ratio, moving the test images to a new folder `test`: 
```	
np.random.seed(123)
for f in files_name:
    if np.random.rand(1) < 0.2:
        	shutil.move('train_images/'+f, 'test_images/'+f) 
```
Based on the two images sets created, we split the annotations accordingly by filtering the file names.
```
train_files = os.listdir('train') 
test_files = os.listdir('test')

logos = ['Adidas','Apple Inc.','Chanel','Coca-Cola','Emirates','Hard Rock Cafe','Mercedes-Benz','NFL','Nike','Pepsi','Puma','Starbucks','The North Face','Toyota','Under Armour']

#annot_train.csv is the original .csv file with all annotations
annot_data = pd.read_csv(‘annot_train.csv’).rename({‘photo_filename’:’filename’}, axis=1)

annot_train = annot_data[annot_data.filename.isin(train_files)]
annot_train[‘image_id’] = [i for i in range(len(annot_train))]
annot_train.loc[~annot_train[‘class'].isin(logos),'class'] = 'Other'
annot_train.to_csv('annot_train.csv')

annot_test = annot_data[annot_data.filename.isin(test_files)]
annot_test.loc[~annot_test[‘class'].isin(logos),'class'] = 'Other'
annot_test.to_csv('annot_test.csv')
```


### Train the Model  
Before feeding the data to Detecto, we have performed some augmentations that can be found within [detecto_15_logos.py](https://github.com/xx-liu-2244/CV_logo_detection/blob/main/detecto_15_logos.py)
```	
$python detecto_15_logos.py
```
NB: Since neural networks tend to take long training periods (on average it took us **5:30h** per epoch!), we suggest to implement the following command within your terminal (or Powershell) before running the above python code:
```
$nohup python detecto_15_logos.py &  
```
nohup --> “not hanging up” and running the model in background <br />
	
### Prediction and Evaluation  
Logo predictions are performed through [predict_detecto_15logos.py](https://bocconi-my.sharepoint.com/:f:/g/personal/alessia_lin_studbocconi_it/Ehn6_H1j4hVGgJHL8DJq8dQBwDDedYqAR7qZ9yZVGDVliA?e=hccapm) by calculating the respective Intersection over Union (IoU). IoU is an evaluation metric used to measure the accuracy of an object detector on a particular dataset, especially with convolutional neural networks. In order to apply IoU we need:<br />
* the ground-truth bounding boxes (the true hand-labeled bounding boxes, i.e. given by [annot_test.csv](https://github.com/xx-liu-2244/CV_logo_detection/blob/main/annot_test.csv) )
* the predicted bounding boxes from our model (by applying the weights file [detecto_weights_15logos.pth](https://bocconi-my.sharepoint.com/:f:/g/personal/alessia_lin_studbocconi_it/Ehn6_H1j4hVGgJHL8DJq8dQBwDDedYqAR7qZ9yZVGDVliA?e=hccapm) ).

	
	> IoU = <sup>Area of Overlap</sup>&frasl;<sub>Area of Union</sub> 
	

true_logo | IoU |
--- | --- | 
Adidas | 0.843515  |
Apple Inc.| 0.809512|
Chanel | 0.600565 |
Coca-Cola | 0.715212 |
Emirates | 0.646927 |
Hard Rock Cafe | 0.768398 |
Mercedes-Benz | 0.858898 |
NFL | 0.772257|
Nike  | 0.782048 |
Other | 0.547186 |
Pepsi | 0.708084 |
Starbucks | 0.846809 |
The North Face  | 0.798458 |
Toyota | 0.713046 |
Under Armour  | 0.778998 |




## Resources

* The files used for analysis are in OneDrive: https://bocconi-my.sharepoint.com/:f:/g/personal/alessia_lin_studbocconi_it/Ehn6_H1j4hVGgJHL8DJq8dQBwDDedYqAR7qZ9yZVGDVliA?e=hccapm
* The original dataset (pre-training): https://www.dropbox.com/s/nkoxs4boe8m48xf/DLCV_logo_project.tar.gz?dl=0


[^1]: The purpose of this repository is solely for the final submission of the course of Deep Learning for Computer Vision at Bocconi University. 


# ROBOFLOW
By Domenique: if you register as a public account it gives you the possibility to upload up to 10k images.

#1. In a new LogoDet project, upload the annot_train.csv file 

#2. Upload only the train folder with images (coordinate with the others: upload only the images that were not already processed by someone else. I put the first 10k of train). Choose the option 100% training:
 
 ![image](https://user-images.githubusercontent.com/51834820/142015508-5b486c67-d10c-4b01-9b61-575a90167cfa.png)

#3. Wait for the infinite-time-consuming upload: for me, 9851 images out of 10k were uploaded as the remaining ones did not have a correspondence in the csv file.

#4. Generate a new version of the dataset, including the following pre-processing and augmentation steps:

![image](https://user-images.githubusercontent.com/51834820/142015543-e3021ab9-6960-4f63-84c0-eb47f6727326.png)
 
#5. Click on Generate and wait. Give a name as: first 10k, second 10k augmented, etc…

#6. Download the dataset choosing the most convenient format:
Tensorflow Object Detection exports also a csv file that is like ours!

 ![image](https://user-images.githubusercontent.com/51834820/142196699-576792ab-be13-4638-b513-0e40a292d04d.png)
 
Choose to download the zip file via an URL:

![image](https://user-images.githubusercontent.com/51834820/142196826-20a05715-3c74-4632-9e1d-4df90a212c96.png)

#7. Repeat this process 4 times to augment approximately all the train images


