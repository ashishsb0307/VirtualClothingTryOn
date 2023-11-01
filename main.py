import json
from os import path as osp
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F
import cv2


from DataProcessing import VITONDataset, VITONDataLoader

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise
  
class Options:
    def __init__(self):
        self.load_height = 1024
        self.load_width = 768
        self.semantic_nc = 13
        self.dataset_dir = r'datasets/'
        self.dataset_mode = r'test'
        self.dataset_list = r'test_pairs.txt'
        self.batch_size = 1
        self.workers = 1
        self.semantic_nc = 13
        self.init_type = 'xavier'
        self.init_variance = 0.
        self.checkpoint_dir = r'./checkpoints/'
        self.seg_checkpoint = r'seg_final.pth'
opt = Options()

test_dataset = VITONDataset(opt)
test_loader = VITONDataLoader(opt, test_dataset)

with torch.no_grad():
    for i, inputs in enumerate(test_loader.data_loader):
        img_names = inputs['img_name']
        c_names = inputs['c_name']['unpaired']

        img_agnostic = inputs['img_agnostic']
        parse_agnostic = inputs['parse_agnostic']
        pose = inputs['pose']
        c = inputs['cloth']['unpaired']
        cm = inputs['cloth_mask']['unpaired']
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
        c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
        cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
        seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size())), dim=1)
      
# Clothing-agnostic person image
Image.fromarray(np.uint8(img_agnostic[0][0] * 255))

# pose map
Image.fromarray(np.uint8(pose[0][0] * 255))

# clothing
Image.fromarray(np.uint8(c[0][0] * 255))

# agnostic segmentation without the body and hands
Image.fromarray(np.uint8(parse_agnostic_down[0][0] * 255))

parse_pred_down = seg(seg_input)
parse_pred = gauss(up(parse_pred_down))
parse_pred = parse_pred.argmax(dim=1)[:, None]

parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).cuda()
parse_old.scatter_(1, parse_pred, 1.0)

labels = {
    0:  ['background',  [0]],
    1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
    2:  ['upper',       [3]],
    3:  ['hair',        [1]],
    4:  ['left_arm',    [5]],
    5:  ['right_arm',   [6]],
    6:  ['noise',       [12]]
}
parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cuda()
for j in range(len(labels)):
    for label in labels[j][1]:
        parse[:, j] += parse_old[:, label]
# For debugging
seg_op = np.uint8(parse[0][2])
for i in range(3, 6):
    seg_op = seg_op + np.uint8(parse[0][i])
Image.fromarray(seg_op * 255)
