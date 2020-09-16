'''

Hand segmentation using Detectron2
Author: Farnaz Nouraei
Email: farnaz_nouraei@brown.edu

<<Inference>>
Dataset: EgoHands 
Model: Mask R-CNN

'''


import os
import random
import cv2
import sys,getopt

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_converter import get_hand_dicts

root_path = '../..'


''' set default hyperparams '''

nms_threshold = 0.7 
images_per_batch = 2
learning_rate = 0.001
num_epochs = 10
roi_heads_batch_size_per_im = 128
vis_score_threshold = 0.8

''' get user args for hyperparams '''

try:
    opts, args = getopt.getopt(sys.argv[1:],"h",
    ["visthr=","lr=","bpim=","impb=","nmsthr=","ep="])
except getopt.GetoptError as err:
    print('option not recognized!')  
    sys.exit(2)
output = None

for o, a in opts:
    if o == "-h":
        print(
        'Please follow the format: python filename --visthr x --lr x --bpim x --impb x --nmsthr x --ep x'
        )
        sys.exit(2)

    elif o in ("--lr"):
        learning_rate = a
    elif o in ("--bpim"):
        roi_heads_batch_size_per_im = a
    elif o in ("--impb"):
        images_per_batch = a
    elif o in ("--nmsthr"):
        nms_threshold = a
    elif o in ("--ep"):
        num_epochs = a
    elif o in ("--visthr"):
        vis_score_threshold = float(a)

    else:
        assert False, "unhandled option"


''' check devices '''

import torch, torchvision
print('Torch - Cuda Version: {}, is_cuda_available: {}'.format(torch.__version__, torch.cuda.is_available()))
print('Current Cuda Device:',torch.cuda.current_device())
print('Device Name:',torch.cuda.get_device_name())


classes = ["hand"]


dicts = dict()
images = dict()
for d in ['train','test']:
    dicts[d],images[d] = get_hand_dicts(os.path.join(root_path,"data"),d)
    DatasetCatalog.register("hand_" + d, lambda d=d: dicts[d])
    MetadataCatalog.get("hand_" + d).set(thing_classes=classes,thing_colors=[(177, 205, 223), (223, 205, 177)])
hand_metadata = MetadataCatalog.get("hand_train")

''' INFERENCE per image '''


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cuda'
cfg.DATASETS.TEST = ("hand_test",)
cfg.MODEL.WEIGHTS = os.path.join(root_path,"models_detectron2","models_hand_segmentation",
                                 "model_lr{}_ep{}_bspim{}_impb{}_thr{}.pth".
                                    format(
                                        learning_rate,
                                        num_epochs,
                                        roi_heads_batch_size_per_im,
                                        images_per_batch,
                                        nms_threshold
                                    ))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = vis_score_threshold 

predictor = DefaultPredictor(cfg)

"""
''' Visulize random images from TEST set '''

from detectron2.utils.visualizer import ColorMode # need this if BW image is needed

for ind,d in enumerate(random.sample(dicts['test'], 5)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=hand_metadata,
                   scale=0.8,
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(('test_visthr{}_lr{}_ep{}_bspim{}_impb{}_thr{}_'+str(ind)+'.jpg').format(
                                                                                vis_score_threshold,
                                                                                learning_rate,
                                                                                num_epochs,
                                                                                roi_heads_batch_size_per_im,
                                                                                images_per_batch,
                                                                                nms_threshold
                                                                                ),
                v.get_image()[:, :, ::-1]
               )


    # TODO: for multi-class case add [0] to and uncomment the following:
    '''
    pred_class = (outputs['instances'].pred_classes).detach()
    pred_score = (outputs['instances'].scores).detach()
    '''
"""

''' keep time '''

import time
t0= time.clock()

''' Visulize images from a video '''

import pathlib, natsort

try:
    os.mkdir('./test_results_maskrcnn')
except FileExistsError:
    pass

files_ = []
#for path in pathlib.Path(root_path+"/data/egohands_all_frames/puzzle_office_S_T").iterdir():
for path in pathlib.Path("../../Downloads/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/test/P09").iterdir():
    if path.is_file():  
        files_ . append (str(path))         
sorted_ = natsort.natsorted(files_,reverse=False)

ind = 0
for p in sorted_[0:640]:
    ind += 1
    im = cv2.imread(p)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=hand_metadata,
                    scale=0.8,
                  )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(('./test_results_maskrcnn/test_'+str(ind)+'.jpg').format(
                                                    vis_score_threshold,
                                                    learning_rate,
                                                    num_epochs,
                                                    roi_heads_batch_size_per_im,
                                                    images_per_batch,
                                                    nms_threshold
                                                    ),
                   v.get_image()[:, :, ::-1]
                   )

os.system("rm output/*.json") # remove the most recent evaluation results to make room for the current ones
os.system("rm output/testing/*.json") # remove the most recent evaluation results to make room for the current ones

os.makedirs(os.path.join(root_path,
                        'logs',
                        'test_hand_segmentation'), exist_ok=True)
from detectron2.utils.logger import setup_logger
setup_logger(output = os.path.join(root_path,
                        'logs',
                        'test_hand_segmentation',
                        'model_lr{}_ep{}_bspim{}_impb{}_thr{}.pth'.format(learning_rate,
                           num_epochs, 
                           roi_heads_batch_size_per_im, 
                           images_per_batch, 
                           nms_threshold)))


''' evaluate model on TEST set '''

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader #this loader uses default datamapper (involves RandomFlip
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

model = build_model(cfg)
DetectionCheckpointer(model).load(os.path.join(root_path,"models_detectron2","models_hand_segmentation",
                                 "model_lr{}_ep{}_bspim{}_impb{}_thr{}.pth".
                                    format(
                                        learning_rate,
                                        num_epochs,
                                        roi_heads_batch_size_per_im,
                                        images_per_batch,
                                        nms_threshold
                                    )))

evaluator = COCOEvaluator("hand_test", cfg, False, output_dir="./output/testing")
val_loader = build_detection_test_loader(cfg, "hand_test")
inference_on_dataset(model, val_loader, evaluator)


t1 = time.clock() - t0
print("total inference time: ", t1," seconds") # CPU seconds elapsed (floating point)
