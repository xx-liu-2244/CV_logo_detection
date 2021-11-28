# CV_logo_detection

GitHub Logo Detection

Files:
- train images (80%)
- test images (20%)
- annot_train.csv
- annot_test.csv
- detecto_weights_xlogos.pth
- detecto_x_logos.py
- detecto_evaluation.ipynb

ReadME
Intro: this is a repo for Logo Detection using Detecto model.

1. The Model - Detecto
		General overview, explanation
		LInk to documentation
		Requirements:
		- csv structure: filename, height, width, class, xmin, ymin, xmax, ymax, image_id
		- in GPU
	
2. how train and test images were split, moving the test images to a new folder ‘test’

	np.random.seed(123)
	for f in files_name:
    		if np.random.rand(1) < 0.2:
        	shutil.move('train_images/'+f, 'test_images/'+f) 

3. train and test annotations generation, saved to csv. Explain how many classes and which logos we used to train our model. And why we included the “Other” class as well.


annot_train.csv = original csv with all annots

annot_data= pd.read_csv(‘annot_train.csv’).rename({‘photo_filename’:’filename’}, axis=1)

logos = […….]

annot_train = annot_data[annot_data.filename.isin(train_files)]
annot_train[‘image_id’] = [i for i in range(len(annot_train))]
annot_train.loc[~annot_train[‘class'].isin(logos),'class'] = 'Other'
annot_train.to_csv(……)

annot_test = annot_data[annot_data.filename.isin(test_files)]
annot_test.loc[~annot_test[‘class'].isin(logos),'class'] = 'Other'
annot_test.to_csv(….)


4. Training the model
	Showing the augmentation used in the model
	
	$python detecto_x_logos.py
	FYI: nohup python detecto_x_logos.py &  —> for “not hanging up” and running the model in bg
	this saves the model weights in the directory….
	 
	NB: this takes a lot of time. on average 5:30h per epoch
	
5. Prediction and Evaluation
	Putting the nb for predicting test images and calculating the respective IoU
		ref: detecto_evaluation.ipynb
	Brief explanation of IoU
	Showing the results of the model predictions 
	(maybe also accuracy metric—> logos correctly predicted) 

	put screenshot of images with bouding boxes drawn on it…
	
	
## Limitations (TBD)

* Unbalanced dataset: too many Nike & Adidas
* The files used for analysis are in OneDrive: https://bocconi-my.sharepoint.com/:f:/g/personal/alessia_lin_studbocconi_it/EslFGa-ZMuNItHzhlJS-7owBlOYlMowfnZ7RTLaiiHdbeA?e=6yHCJq



## Fine-Tuning with Keras

#### 1. Data
* Resize
* Standardizing/Normalization
* Data augumentation: Logos might not be symmetric, consider rotation/resize than overturn

#### 2. Build the Model
* Base model/pre-trained model selection: 
![image](https://user-images.githubusercontent.com/83235873/138552330-dc31be07-ec87-4824-bd16-7f79a926258e.png)

* Compile new layers, rescale-basemodel-pooling-dropout-dense 

#### 3. Train and Fine-Tuning
* try different top-layer inclusion


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


