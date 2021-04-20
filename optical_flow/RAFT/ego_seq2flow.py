import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

print("number of cuda devices available: ", torch.cuda.device_count())

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
    #print("[DEBUG] shape of stacked images: ", images.shape)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]
        

def viz(flo,flo_path,filename):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    #print("[DEBUG] optical flow array: ",np.unique(flo))
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #print("[DEBUG] flo image array: ", np.unique(flo))
    #img_flo = np.concatenate([img, flo], axis=0)
    cv2.imwrite(os.path.join(flo_path,filename+".png"), flo[:, :, [2,1,0]])


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

            for path in pathlib.Path(args.path).iterdir():
                # loop over subdirectories 
                if path.is_dir():
                    files_ = []
                    names_ = []
                    os.makedirs(os.path.join(path,"flo"),exist_ok = True)
                    print("path: ",str(path.name))
                    for filepath in path.glob("*.jpg"):
                        files_ . append (str(filepath))
                        names_.append(str(filepath.stem))
                    sorted_ = natsort.natsorted(files_,reverse=False)
                    #print("[DEBUG] sorted:", sorted_)
                    sorted_names = natsort.natsorted(names_,reverse=False)
                    ind = 0
                    #print("[DEBUG] len(files_):\n ",len(files_))
                    #print("[DEBUG] file list:\n ",files_)
                    for p in sorted_[0:len(files_)]:
                        if ind < (len(files_)-1): 
                            images = glob.glob(sorted_[ind]) + glob.glob(sorted_[ind+1])
                            images = load_image_list(images)
                            flow_low, flow_up = model(images[0,None], images[1,None], iters=20, test_mode=True)
                            
                            #save flo files in the same dir as images
                            print("creating flow image for {} ...".format(sorted_names[ind]))
                            viz(flow_up
                                ,os.path.join(path,"flo")
                                ,sorted_names[ind])
                        ind += 1
                        print("[DEBUG] index: ", ind)



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

