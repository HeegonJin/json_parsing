# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import json
from itertools import chain


def drawPoly(df_row):
    global mask_img
    mask_label = df_row['id']
    pts = df_row['shape']
    pts = np.array(getPoly(pts),np.int32)
    mask_img = cv2.fillPoly(mask_img, [pts], int(mask_label))
    
    
def getPoly(pts):
    new_pts = []
    for pt in pts:
        new_pt = list(pt.values())
        new_pts.append(new_pt)
    return new_pts


abs_path = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(abs_path, "meta", "Sample Dataset", "sample-image.jpg.json")#"meta/Sample Dataset/sample-image.jpg.json")
label_path = os.path.join(abs_path, "labels","61651091-be32-4d05-b0bd-e27b4fab09cb.json")
project_path = os.path.join(abs_path, "project.json")

with open(meta_path, "r") as meta_file:
    meta_json = json.load(meta_file)
mask_img = np.zeros((meta_json['image_info']['height'], meta_json['image_info']['width']), np.int32)
valid_ids = list(chain.from_iterable(list(meta_json['masks']['default']['instance_id']['instance_ids'].values())))

with open(project_path, "r") as project_file:
    project_json = json.load(project_file)
classes = project_json['objects']

with open(label_path, "r") as label_file:
    label_json = json.load(label_file)
objects = label_json['result']['objects']
for obj in objects:
    obj['shape'] = list(obj['shape'].values())[0]
    
obj_df = pd.DataFrame(objects)
cls_df = pd.DataFrame(classes).rename(columns = {"class_name" : "class"})

df = pd.merge(obj_df, cls_df)
df = df.loc[df['id'].isin(valid_ids)].loc[:,['id','shape']]
for idx in range(df.shape[0]):
    drawPoly(df.iloc[idx])

mask_img *= 50
cv2.imshow('mask_img', mask_img.astype(np.uint8))
cv2.waitKey(0)
'''
meta_json = pd.read_json(meta_path, orient = 'index')
label_path = os.path.join(abs_path, meta_json.loc['label_path',0])

project_json = pd.read_json(os.path.join(abs_path, "project.json"), orient = 'index') #json to pandas, merge
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
'''           