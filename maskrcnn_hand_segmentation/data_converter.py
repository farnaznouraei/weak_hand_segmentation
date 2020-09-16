'''

Hand segmentation using Detectron2
Author: Farnaz Nouraei
Email: farnaz_nouraei@brown.edu

<<Dataset Conversion>>
Dataset: EgoHands 
Model: Mask R-CNN
'''

''' Some basic setup:'''
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

root_path = '../..'

'''Convert dataset into detectron2 format'''
#NOTE: The EgoHands annotation data is in .mat format

import os
import numpy as np
import json
from scipy.io import loadmat
from tqdm import tqdm


def get_hand_dicts(path , mode = 'train'):

  annot_file = loadmat(os.path.join(path, 'egohands_data', 'metadata.mat'))
  num_videos = annot_file['video'].shape[1]
  num_frames = annot_file['video']['labelled_frames'][0,0].shape[1]

  mode_train = int(mode=='train')
  mode_val = int(mode=='val')
  mode_test = int(mode=='test')

  dataset_dicts = []
  images = []
  count_fr = 0
  count_vid = 0
  for i in tqdm(range((mode_val+mode_test)*int(num_videos*6/10),mode_train*int(num_videos*6/10)+(mode_val+mode_test)*int(num_videos)),desc='videos registered for '+mode+': '):

    for j in range(mode_test*int(num_frames*7/10),(mode_train+mode_test)*int(num_frames)+mode_val*int(num_frames*7/10)):

      idx = (count_vid)*count_fr+j # general index for images

      # Extract the polygons from the metadata (.mat file)

      my_right = annot_file['video']['labelled_frames'][0,i]['myright'][0,j]
      my_left = annot_file['video']['labelled_frames'][0,i]['myleft'][0,j]
      your_right = annot_file['video']['labelled_frames'][0,i]['yourright'][0,j]
      your_left = annot_file['video']['labelled_frames'][0,i]['yourleft'][0,j]
      annots = dict(my_right = my_right , my_left = my_left , your_right=your_right , your_left = your_left)
      # The above variables will be numpy arrays each containing a
      # (num_poly_points x 2) matrix showing poly coordinates

      poly_coordinates = dict(my_left = [], my_right = [], your_right = [], your_left=[]);
      for key in annots.keys():
        for _ in range(annots[key].shape[0]):
            poly_coordinates[key].append ((annots[key][_,0],annots[key][_,1]))

      record = {}
      frame_num = str(int(annot_file['video']['labelled_frames'][0,i]['frame_num'][0,j].flatten()))

      if len(frame_num)==3:
          frame_num = '0'+ frame_num
      if len(frame_num)==2:
          frame_num = '00'+ frame_num
      if len(frame_num)==1:
          frame_num = '000'+ frame_num
      else:
          pass



      # Load the corresponding image
      video_id = annot_file['video']['video_id'][0,i]
      filename = os.path.join(path+'/egohands_data/_LABELLED_SAMPLES/'+video_id[0]+'/frame_'+ frame_num +'.jpg')
      img = cv2.imread(filename, cv2.IMREAD_COLOR)

      height, width = img.shape[:2]

      record["file_name"] = filename
      record["image_id"] = idx
      record["height"] = height
      record["width"] = width

      objs = []


      for key in annots.keys():
        poly = poly_coordinates[key]
        poly = [p for x in poly for p in x]

        if len(poly) == 0 :
          pass
        else:
          obj = {
                  "bbox": [np.min([p[0] for p in poly_coordinates[key]]), np.min([p[1] for p in poly_coordinates[key]]), np.max([p[0] for p in poly_coordinates[key]]), np.max([p[1] for p in poly_coordinates[key]])],
                  "bbox_mode": BoxMode.XYXY_ABS,
                  "segmentation": [poly],
                  "category_id": 0, 
                  "iscrowd": 0
                }
          objs.append(obj)

      record["annotations"] = objs

      dataset_dicts.append(record)
      images.append(img)

      count_fr += 1
    count_vid += 1

  return dataset_dicts,images



