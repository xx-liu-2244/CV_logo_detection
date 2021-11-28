import pandas as pd
import numpy as np
import os
from detecto import core, utils
from detecto.visualize import show_labeled_image
import cv2
from collections import namedtuple

ANNOT_TEST_PATH = "./annot_test.csv"
TEST_IMAGES_PATH = "./test"

annot_test = pd.read_csv(ANNOT_TEST_PATH) #annotations of test set
test_files = os.listdir(TEST_IMAGES_PATH) #list of test images file names
logos = annot_test['class'].unique().tolist() #list of logos

# Loading the detecto saved model
model_load = core.Model.load("model_weights_17logos.pth", logos) #change file name based on how it was saved

def IoU(y_true,y_pred):
    xA = max(y_true[0], y_pred[0])       
    yA = max(y_true[1], y_pred[1])
    xB = min(y_true[2], y_pred[2])
    yB = min(y_true[3], y_pred[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxtrueArea = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])    
    boxpredArea = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxtrueArea + boxpredArea - interArea)

    # return the intersection over union value
    return iou
  

# store prediction results of test images for labels, boxes and scores in respective dictionaries
labels_prediction = {}
boxes_prediction = {}
scores_prediction = {}

for f in test_files:
  image = utils.read_image(TEST_IMAGES_PATH + '/' + f) 
  predictions = model_load.predict(image)
  labels, boxes, scores = predictions

  filtered_index = np.where(max(scores))
  filtered_score = scores[filtered_index]
  filtered_box = boxes[filtered_index]
    
  for i in filtered_index:
    ind = i[0]
  filtered_label = labels[ind]

  labels_predictions[f] = filtered_label
  boxes_predictions[f] = filtered_box
  scores_predictions[f] = filtered_score.tolist()
  
# create one dataframe storing all the results (iou, logo predicted, score)
lst = []

for f in test_files:
  true_box = annot_test[annot_test['filename']==f].iloc[0,-4:].tolist()
  pred_box = boxes_predictions[f]
  iou = float(IoU(true_box, pred_box[0]))
  true_logo = annot_test[annot_test['filename']==f]['class'].iloc[0]
  pred_logo = labels_predictions[f]
  pred_score = scores_predictions[f]
  lst.append([f,iou,true_logo,pred_logo, pred_score])

test_results = pd.DataFrame(lst, columns=['filename','IoU','true logo','pred logo','score'])
