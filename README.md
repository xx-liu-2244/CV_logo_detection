# Deep Learning for Computer Vision - Logo Detection

Welcome to the repository for Logo Detection using the Detecto model![^1] :raised_hands: 

<a href="https://user-images.githubusercontent.com/81080301/144510250-f960b63c-ac2a-41db-b77c-f3d4d5a0f181.jpg"><img src="https://user-images.githubusercontent.com/81080301/144510250-f960b63c-ac2a-41db-b77c-f3d4d5a0f181.jpg" width="300" height="300"/></a>
<a href="https://user-images.githubusercontent.com/81080301/144510291-e2c3e970-7166-48d1-8bc9-49fba00c465c.jpg"><img src="https://user-images.githubusercontent.com/81080301/144510291-e2c3e970-7166-48d1-8bc9-49fba00c465c.jpg" width="400" height="300"/></a>
<a href="https://user-images.githubusercontent.com/81080301/144511248-b86bbd14-4e99-40a4-b8c3-e243792c5630.jpg"><img src="https://user-images.githubusercontent.com/81080301/144511248-b86bbd14-4e99-40a4-b8c3-e243792c5630.jpg" width="350" height="300"/></a>
<a href="https://user-images.githubusercontent.com/81080301/144519452-b867e752-e070-404c-abbe-f6f53c93e97b.jpg"><img src="https://user-images.githubusercontent.com/81080301/144519452-b867e752-e070-404c-abbe-f6f53c93e97b.jpg" width="350" height="300"/></a>



### The Model - Detecto  :mag: 👀
[Detecto](https://detecto.readthedocs.io/en/latest/) is a Python package built on top of Pytorch that allows you to perform object detection and make inference on still images and videos. It creates and runs a pre-trained RCNN ResNet-50 FPN. <br />
Instead of using pre-existing weights, we trained the model on a custom dataset with the aim to predict the following logos and construct their corresponding bounding boxes: _Adidas, Apple Inc., Chanel, Coca-Cola, Emirates, Hard Rock Cafe, Mercedes-Benz, NFL, Nike, Pepsi, Puma, Starbucks, The North Face, Toyota, Under Armour_.

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
	
### Import the Dataset :floppy_disk:
The required raw data utilised to run the model and to make predictions can be downloaded from the following [dropbox link](https://www.dropbox.com/s/nkoxs4boe8m48xf/DLCV_logo_project.tar.gz?dl=0). It contains the `train` folder with all the images and the respective annotations in `annot_train.csv`. To unzip it, run the following command:
```
$tar -xvf DLCV_logo_project.tar.gz
```

### Train Test Split  :scissors:
After downloading the file within the desired working directory, we split images into train and test with a 80/20 ratio, moving the test images to a new folder `test`: 
```	
np.random.seed(123)
for f in files_name:
    if np.random.rand(1) < 0.2:
        	shutil.move('train_images/'+f, 'test_images/'+f) 
```
Based on the two images sets created, we split the annotations accordingly by filtering the file names.
Complete code reference: [data_processing.ipynb](https://github.com/xx-liu-2244/CV_logo_detection/blob/main/data_preprocessing.ipynb)
```
train_files = os.listdir('train') #train images folder
test_files = os.listdir('test') #test images folder

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


### Train the Model  :weight_lifting:
Before feeding the data to Detecto, we have performed some augmentations that can be found within [detecto_15_logos.py](https://github.com/xx-liu-2244/CV_logo_detection/blob/main/detecto_15_logos.py)
```	
$python detecto_15_logos.py
```
NB: Since neural networks tend to take long training periods (on average it took us **5:30h** per epoch!), we suggest to implement the following command within your terminal (or Powershell) before running the above python code:
```
$nohup python detecto_15_logos.py &  
```
`nohup` --> “not hanging up”, i.e. it will keep running regardless of the connection status
`&` --> run the command in background <br />
	
### Prediction and Evaluation :crystal_ball:
Logo predictions are performed through [predict_detecto_15logos.py](https://bocconi-my.sharepoint.com/:f:/g/personal/alessia_lin_studbocconi_it/Ehn6_H1j4hVGgJHL8DJq8dQBwDDedYqAR7qZ9yZVGDVliA?e=hccapm) by calculating the respective Intersection over Union (IoU). IoU is an evaluation metric used to measure the accuracy of an object detector on a particular dataset, especially with convolutional neural networks. In order to apply IoU we need:<br />
* the ground-truth bounding boxes, i.e. given by [annot_test.csv](https://github.com/xx-liu-2244/CV_logo_detection/blob/main/annot_test.csv) 
* the predicted bounding boxes from our model by applying the weights file [detecto_weights_15logos.pth](https://bocconi-my.sharepoint.com/:u:/r/personal/alessia_lin_studbocconi_it/Documents/CV/DLCV_15logos/detecto_weights_15logos.pth?csf=1&web=1&e=UcWDKW)

	
	> IoU = <sup>Area of Overlap</sup>&frasl;<sub>Area of Union</sub> 

Just as before, run the below command in the terminal to get the prediction results for the trained model on the test set:
```
$nohup python predict_detecto_15logos.py &
```
The outcome of our prediction can be found in [detecto_result_15logos.csv](https://bocconi-my.sharepoint.com/:u:/r/personal/alessia_lin_studbocconi_it/Documents/CV/DLCV_15logos/detecto_weights_15logos.pth?csf=1&web=1&e=UcWDKW), which includes: the test images, IoU, the true logo, and the predicted logo.


From `predict_detecto_15logos.py` we obtained the following results for the prediction metric with respect to the true bounding boxes:<br>



true_logo |  IoU
--- | ---
Adidas | 0.843515 
Apple Inc. | 0.809512
Chanel | 0.600565 
Coca-Cola  | 0.715212
Emirates | 0.646927
Hard Rock Cafe | 0.768398
Mercedes-Benz | 0.858898
NFL | 0.772257 
Nike| 0.782048
Other | 0.547186
Pepsi | 0.546280
Puma | 0.708084
Starbucks | 0.846809
The North Face| 0.798458 
Toyota | 0.713046 
Under Armour |  0.778998

_Model Accuracy for the 5 compulsory logos:__ 91.985%



[^1]: The purpose of this repository is solely for the final submission of the course of Deep Learning for 20600 Computer Vision at Bocconi University. 

----------------------------------------------------------------------------------------------------------------------------------------------------------------
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


