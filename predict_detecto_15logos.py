import pandas as pd
import numpy as np
import os
import cv2

from detecto import core, utils
from tqdm import tqdm

ANNOT_TEST_PATH = "annot_test.csv"
TEST_IMAGES_PATH = "test"

annot_test = pd.read_csv(ANNOT_TEST_PATH) #annotations of test set
test_files = os.listdir(TEST_IMAGES_PATH) #list of test images file names
logos = annot_test['class'].unique().tolist() #list of logos

model_load = core.Model.load("detecto_weights_15logos.pth", logos) 

def IoU(y_true,y_pred):
    xA = max(y_true[0], y_pred[0])       
    yA = max(y_true[1], y_pred[1])
    xB = min(y_true[2], y_pred[2])
    yB = min(y_true[3], y_pred[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxtrueArea = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])    
    boxpredArea = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])

    # compute the intersection over union by taking the intersection area and dividing it 
    # by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxtrueArea + boxpredArea - interArea)

    # return the intersection over union value
    return iou 



labels_prediction = {}
boxes_prediction = {}
scores_prediction = {}

for f in tqdm(test_files):
    image = utils.read_image(TEST_IMAGES_PATH + '/' + f) 
    predictions = model_load.predict(image)
    labels, boxes, scores = predictions

    if list(scores) == []:
        continue
    else:
        filtered_index = np.where(max(scores))
        filtered_score = scores[filtered_index]
        filtered_box = boxes[filtered_index]

        for i in filtered_index:
            ind = i[0]
        filtered_label = labels[ind]

        labels_prediction[f] = filtered_label
        boxes_prediction[f] = filtered_box
        scores_prediction[f] = filtered_score.tolist()
    
    
lst = []

for f in tqdm(test_files):
    if f not in labels_prediction.keys():
        continue
    else:
        true_box = annot_test[annot_test['filename']==f].iloc[0,-4:].tolist() #coordinates of the ground-truth bounding boxes
        pred_box = boxes_prediction[f] #coordinates of the predicted bounding boxes
        iou = float(IoU(true_box, pred_box[0]))
        true_logo = annot_test[annot_test['filename']==f]['class'].iloc[0]
        pred_logo = labels_prediction[f]
        pred_score = scores_prediction[f]
        lst.append([f,iou,true_logo,pred_logo, pred_score])

test_results = pd.DataFrame(lst, columns=['filename','IoU','true logo','pred logo','score'])

test_results.to_csv(r'detecto_results_15logos.csv', header = True, index = False)


#To download images with true (green) and predicted (red) bounding boxes 
f = 'phoenix_890226270370620432_20150104.jpg' #insert name of interested file 

true_bbox = annot_test[annot_test['filename']==f].iloc[0,-4:].tolist()
pred_bbox = boxes_prediction[f]
pred_bbox = [int(num) for num in pred[0].tolist()] #convert tensorflow into list

image = cv2.imread('test/'+f)
cv2.rectangle(image, tuple(true_bbox[:2]),tuple(true_bbox[2:]), (0, 255, 0), 2)
cv2.rectangle(image, tuple(pred_bbox[:2]),tuple(pred_bbox[2:]), (0, 0, 255), 2)
iou = IoU(true_bbox, pred_bbox)
cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
cv2.putText(image, labels_prediction[f], tuple([c-10 for c in pred_bbox[2:]]), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
cv2.imwrite('filename.jpg', image) 