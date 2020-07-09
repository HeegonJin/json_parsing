import os
import cv2
import numpy as np
import pandas as pd
import json
from itertools import chain


def makeClsDict(project_json):
    classes = project_json['objects']
    class_dict = {}
    
    for _class in classes:
        class_dict[_class['class_name']] = _class['class_id']
        
    return class_dict

if __name__ == "__main__":

    abs_path = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(abs_path, "meta", "Sample Dataset", "sample-image.jpg.json")#"meta/Sample Dataset/sample-image.jpg.json")
    label_path = os.path.join(abs_path, "labels","61651091-be32-4d05-b0bd-e27b4fab09cb.json")
    project_path = os.path.join(abs_path, "project.json")
    
    with open(meta_path, "r") as meta_file:
        meta_json = json.load(meta_file)

    valid_ids = list(chain.from_iterable(list(meta_json['masks']['default']['instance_id']['instance_ids'].keys())))

    # make class dictionary
    with open(project_path, "r") as project_file:
        project_json = json.load(project_file)        
    
    class_dict = makeClsDict(project_json)
    
    # make polygon points
    with open(label_path, "r") as label_file:
        label_json = json.load(label_file)
        
    objects = label_json['result']['objects']
    object_list = []
    
    for i, obj in enumerate(objects):

        if 'polygon' not in obj['shape']:
            continue

        _shape_dict = obj['shape']['polygon']
        points_list = []

        for _points in _shape_dict:
            points_list.append([_points['x'], _points['y']])

        object_list.append([obj['id'], class_dict[obj['class']], points_list])

        

    # make mask image

    _img_height = meta_json['image_info']['height']
    _img_width = meta_json['image_info']['width']
    mask_img = np.zeros((_img_height, _img_width), np.int32)

    for _obj_id, _obj_class, _obj_points in object_list:
        pts = np.array(_obj_points, dtype=np.int32)
        mask_img = cv2.fillPoly(mask_img, [pts], int(_obj_class))

    mask_img *= 50

    cv2.imshow('mask_img', mask_img.astype(np.uint8))
    cv2.waitKey(0)    