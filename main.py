# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import json


def getPoly(pts):
    new_pts = []
    for pt in pts:
        new_pt = list(pt.values())
        new_pts.append(new_pt)
    return new_pts


abs_path = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(abs_path, "meta/Sample Dataset/sample-image.jpg.json")

meta_json = pd.read_json(meta_path, orient = 'index')
label_id = meta_json.loc['label_id',0]
label_path = os.path.join(abs_path, meta_json.loc['label_path',0])

project_json = pd.read_json(os.path.join(abs_path, "project.json"), orient = 'index')
class2id = {}
for i in range(5):
    class2id[project_json.loc['objects',i]['class_name']] \
        = project_json.loc['objects',i]['class_id']
h = meta_json.loc['image_info', 0]['height']
w = meta_json.loc['image_info', 0]['width']
mask_img = np.zeros((h,w), np.int32)

with open(label_path) as json_file:
    image_json = json.load(json_file)
    
objects = image_json['result']['objects']

valid_ids = meta_json.loc['masks', 0]['default']['instance_id']['instance_ids']
mask_id = []
for i in valid_ids.values():
    mask_id += i

for obj in objects:
    if obj['id'] in mask_id:
        for key in obj['shape']:
            if key == 'polygon':
                pts = np.array(getPoly(obj['shape']['polygon']),np.int32)
                mask_img = cv2.fillPoly(mask_img, [pts], class2id[obj['class']])

mask_img *= 50
cv2.imshow('mask_img', mask_img.astype(np.uint8))
cv2.waitKey(0)


                
                