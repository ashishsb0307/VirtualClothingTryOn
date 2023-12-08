# -*- coding: utf-8 -*-
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

from dataset_loader import VITONDataset, VITONDataLoader
from options import Options
from segmentation_generator import SegGenerator
from geometric_matching_module import GMM
from alias_generator import ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def run():
        
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
    # Image.fromarray(np.uint8(img_agnostic[0][0] * 255))

    # pose map
    # Image.fromarray(np.uint8(pose[0][0] * 255))

    # clothing
    # Image.fromarray(np.uint8(c[0][0] * 255))

    # agnostic segmentation without the body and hands
    # Image.fromarray(np.uint8(parse_agnostic_down[0][0] * 255))

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)



    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    # os.path.join(opt.checkpoint_dir, opt.seg_checkpoint)

    opt.semantic_nc = 7
    parse_pred_down = seg(seg_input)
    # print(parse_pred_down)

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    parse_pred = gauss(up(parse_pred_down))

    parse_pred = parse_pred.argmax(dim=1)[:, None]

    parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
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
    parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
    for j in range(len(labels)):
        for label in labels[j][1]:
            parse[:, j] += parse_old[:, label]

    # parse_pred_down.shape

    # parse_pred_down[0][5]

    # parse_pred_down = parse_pred_down.detach().numpy()

    # parse_pred_down[0][5]

    # Image.fromarray(np.uint8(parse_pred_down[0][12] * 255))

    seg_op = np.uint8(parse[0][2])
    for i in range(3, 6):
        seg_op = seg_op + np.uint8(parse[0][i])
    Image.fromarray(seg_op * 255)

    # tmp_image = Image.fromarray(np.uint8(parse[0][2]) * 255)
    # rgb_image = tmp_image.convert("RGB")
    # red_color = (0, 150, 255)
    # for x in range(tmp_image.width):
    #     for y in range(tmp_image.height):
    #         pixel_value = tmp_image.getpixel((x, y))
    #         if pixel_value == 255:  # Assuming 255 represents the thresholded pixels (white)
    #             rgb_image.putpixel((x, y), red_color)
    #         else:
    #             rgb_image.putpixel((x, y), (0, 0, 0))  # Set non-thresholded pixels to black
    # rgb_image

    # tmp_image = Image.fromarray(np.uint8(parse[0][3]) * 255)
    # rgb_image = tmp_image.convert("RGB")
    # red_color = (150, 150, 0)
    # for x in range(tmp_image.width):
    #     for y in range(tmp_image.height):
    #         pixel_value = tmp_image.getpixel((x, y))
    #         if pixel_value == 255:  # Assuming 255 represents the thresholded pixels (white)
    #             rgb_image.putpixel((x, y), red_color)
    #         else:
    #             rgb_image.putpixel((x, y), (0, 0, 0))  # Set non-thresholded pixels to black
    # rgb_image

    # tmp_image = Image.fromarray(np.uint8(parse[0][4]) * 255)
    # rgb_image = tmp_image.convert("RGB")
    # red_color = (255, 0, 0)
    # for x in range(tmp_image.width):
    #     for y in range(tmp_image.height):
    #         pixel_value = tmp_image.getpixel((x, y))
    #         if pixel_value == 255:  # Assuming 255 represents the thresholded pixels (white)
    #             rgb_image.putpixel((x, y), red_color)
    #         else:
    #             rgb_image.putpixel((x, y), (0, 0, 0))  # Set non-thresholded pixels to black
    # rgb_image

    # tmp_image = Image.fromarray(np.uint8(parse[0][5]) * 255)
    # rgb_image = tmp_image.convert("RGB")
    # red_color = (255, 0, 255)
    # for x in range(tmp_image.width):
    #     for y in range(tmp_image.height):
    #         pixel_value = tmp_image.getpixel((x, y))
    #         if pixel_value == 255:  # Assuming 255 represents the thresholded pixels (white)
    #             rgb_image.putpixel((x, y), red_color)
    #         else:
    #             rgb_image.putpixel((x, y), (0, 0, 0))  # Set non-thresholded pixels to black
    # rgb_image


    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)

    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))

    # # Part 2. Clothes Deformation
    # gmm = GMM(opt, inputA_nc=7, inputB_nc=3)

    agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
    pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
    c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
    gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

    _, warped_grid = gmm(gmm_input, c_gmm)
    warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
    warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

    warped_grid.shape, warped_c.shape, warped_cm.shape, warped_cm.reshape(1024,768).shape

    warped_c.sum(axis=1).reshape(1024,768)
    Image.fromarray(np.uint8(warped_c.sum(axis=1).reshape(1024,768).detach().numpy() * 255))

    Image.fromarray(np.uint8(warped_cm.reshape(1024,768).detach().numpy() * 255))


    

    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    # Part 3. Try-on synthesis
    misalign_mask = parse[:, 2:3] - warped_cm
    misalign_mask[misalign_mask < 0.0] = 0.0
    parse_div = torch.cat((parse, misalign_mask), dim=1)
    parse_div[:, 2:3] -= misalign_mask

    output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

    unpaired_names = []
    for img_name, c_name in zip(img_names, c_names):
        unpaired_names.append('{}_{}'.format(img_name.split('_')[0], c_name))

    # save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))

    # if (i + 1) % opt.display_freq == 0:
    #     print("step: {}".format(i + 1))



    op = save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))

    op.save('result.png')

