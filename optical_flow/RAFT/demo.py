import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

print(torch.cuda.device_count())

#os.system("ffmpeg -i P04_26.MP4 -vcodec libx264 -crf 20 epic.mp4")

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]
        

def viz(img, flo,i):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    cv2.imwrite("flow_ego_"+str(i)+".png", flo[:, :, [2,1,0]])

# Opens the Video file
cap= cv2.VideoCapture('./puzzle_courtyard_B_S.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('./ego-frames/ego_'+str(i)+'.png',frame)
    #if i == 401:
        #break
    i+=1

cap.release()
cv2.destroyAllWindows()

''' Load images from sorted file list - two at a time - to save memory '''

import pathlib, natsort
import fnmatch
import os



def run(args):

        
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():

            files_ = []
            for path in pathlib.Path(args.path).iterdir():
                if path.is_file():
                    files_ . append (str(path))         
            sorted_ = natsort.natsorted(files_,reverse=False)

            ind = 0
            for p in sorted_[0:len(files_)]:
                if ind < len(files_): images = glob.glob(sorted_[ind]) + glob.glob(sorted_[ind+1])
                if ind<10 : print(sorted_[ind])
                images = load_image_list(images)
                flow_low, flow_up = model(images[0,None], images[1,None], iters=20, test_mode=True)
                viz(images[0,None], flow_up, ind)
                ind += 1


        # Convert output flows into videos and remove the images

        os.system("ffmpeg -f image2  -i ./flow_ego_%d.png -vcodec libx264 -s 640x480 -pix_fmt yuv420p ego_flow.mp4")
        os.system("rm ./*.png")

DEVICE = 'cuda'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)

