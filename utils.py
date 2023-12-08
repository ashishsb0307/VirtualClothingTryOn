import torch
from torch.utils import data
from os import path as osp
from torchvision import transforms
import json
import numpy as np
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import init
import torchgeometry as tgm
from torch.nn import functional as F
import cv2
import os


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        return im