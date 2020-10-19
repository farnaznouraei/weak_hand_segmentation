import os
import torch
import argparse
import numpy as np
import scipy.misc as misc


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict


def encode_decode(args):

    img = misc.imread(args.img_path)

    data_loader = get_loader(args.dataset)
    loader = data_loader(root=None, is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes
    resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")

    orig_size = img.shape[:-1]
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

# NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    print('input image shape: ',img.shape)
    print('numpy input image=',img)
    img = torch.from_numpy(img).float()

    img = misc.imread('../data/VOC/VOCdevkit/VOC2012/SegmentationClass/pre_encoded_reduced/2007_000032.png')
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")

    img = img.astype(np.float64)

    print('gt image shape: ',img.shape)
    #print('numpy gt image=',img[50:120,50:130])
    print(np.unique(img))
    decoded = loader.decode_segmap(img)
    misc.imsave('./resized_decoded.png', decoded)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )
    args = parser.parse_args()
    encode_decode(args)
