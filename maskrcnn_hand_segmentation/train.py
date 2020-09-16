'''

Hand segmentation using Detectron2
Author: Farnaz Nouraei
Email: farnaz_nouraei@brown.edu

<<Training>>
Dataset: EgoHands 
Model: Mask R-CNN
'''

''' Some basic setup:'''
# Setup detectron2 logger
import detectron2

# import some common libraries
import numpy as np
import cv2
import random
import sys,getopt
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from data_converter import get_hand_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from MyTrainer import MyTrainer


root_path = '../..'


os.system("rm output/*.json") # remove the most recent evaluation results to make room for the current ones
os.system("rm output/validation/*.json") # remove the most recent evaluation results to make room for the current ones


''' set default hyperparams '''

nms_threshold = 0.7 #default is .7 
images_per_batch = 2
learning_rate = 0.001
num_epochs = 6
roi_heads_batch_size_per_im = 128

num_itr_per_ep = int(4800*(0.6)/images_per_batch) #4800 is total number of dataset images

''' get user args for hyperparams '''

try:
    opts, args = getopt.getopt(sys.argv[1:],"h",
    ["lr=","bpim=","impb=","nmsthr=","ep="])
except getopt.GetoptError as err:
    print('option not recognized!')  
    sys.exit(2)
output = None

for o, a in opts:
    if o == "-h":
        print(
        'Please follow the format: python filename --lr x --bpim x --impb x --nmsthr x --ep x'
        )
        sys.exit(2)
    elif o in ("--lr"):
        learning_rate = float(a)
    elif o in ("--bpim"):
        roi_heads_batch_size_per_im = int(a)
    elif o in ("--impb"):
        images_per_batch = int(a)
    elif o in ("--nmsthr"):
        nms_threshold = float(a)
    elif o in ("--ep"):
        num_epochs = int(a)
    else:
        assert False, "unhandled option"

''' keep the time '''

import torch, torchvision
print('Torch - Cuda Version: {}, is_cuda_available: {}'.format(torch.__version__, torch.cuda.is_available()))
print('Current Cuda Device:',torch.cuda.current_device())
print('Device Name:',torch.cuda.get_device_name())



''' Register the converted dataset'''

classes = ["hand"]

dicts = dict()
images = dict()
for d in ['train','val']:
    dicts[d],images[d] = get_hand_dicts(os.path.join(root_path,"data"),d)
    DatasetCatalog.register("hand_" + d, lambda d=d: dicts[d])
    MetadataCatalog.get("hand_" + d).set(thing_classes=classes)
hand_metadata = MetadataCatalog.get("hand_train")



'''TRAIN'''
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


#TO TRAIN (fine-tune):
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[ 16, 32, 64, 128, 256]]
cfg.MODEL.DEVICE='cuda:0'
cfg.DATASETS.TRAIN = ("hand_train",)
cfg.DATASETS.TEST = ("hand_val",)
cfg.TEST.EVAL_PERIOD = 5000
cfg.SOLVER.CHECKPOINT_PERIOD = 20000
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

''' HYPERPARAMS '''
cfg.MODEL.RPN.NMS_THRESH = nms_threshold #0.7 NMS threshold used on RPN proposals (default: .7)
cfg.SOLVER.IMS_PER_BATCH = images_per_batch #2
cfg.SOLVER.BASE_LR = learning_rate #0.00025
cfg.SOLVER.MAX_ITER = num_epochs*num_itr_per_ep
cfg.SOLVER.STEPS = (1200,) 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_heads_batch_size_per_im #128 (default: 512)


cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(root_path,
                        'logs',
                        'train_hand_segmentation'), exist_ok=True)

from detectron2.utils.logger import setup_logger
setup_logger(output = os.path.join(root_path,
                        'logs',
                        'train_hand_segmentation',
                        'model_lr{}_ep{}_bspim{}_impb{}_thr{}.pth'.format(cfg.SOLVER.BASE_LR,
                           num_epochs, 
                           cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, 
                           cfg.SOLVER.IMS_PER_BATCH, 
                           cfg.MODEL.RPN.NMS_THRESH)))
trainer = MyTrainer(cfg) #this trainer involves RandomFlip as a data augmentation method
trainer.resume_or_load(resume=False)
trainer.train()

os.system("mv output/model_final.pth ../../models_detectron2/models_hand_segmentation/model_lr{}_ep{}_bspim{}_impb{}_thr{}.pth".
            format(cfg.SOLVER.BASE_LR,
                   num_epochs, 
                   cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, 
                   cfg.SOLVER.IMS_PER_BATCH, 
                   cfg.MODEL.RPN.NMS_THRESH))



