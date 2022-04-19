import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
import re
import cv2

from config import cfg


def loadImgs_plus(path_im, keyword="", grayscale=False):
    fs = []
    fullfs = []

    files = os.listdir(path_im)
    print(path_im)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for file in files:
        if file.find(keyword) != -1:
            fs.append(file)
            fullfs.append(path_im + "/" + file)

    dim = (224, 224)
    imgs = []
    for i in range(len(fullfs)):
        print("loading file:", fs[i], end='\r')
        if grayscale:
            im = cv2.imread(fullfs[i], 0)
            im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
            
        else:
            im = cv2.imread(fullfs[i], cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
            im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)

        imgs.append(im)
    return np.asarray(imgs)


encoder = Encoder(cfg)
decoder = Decoder(cfg)
refiner = Refiner(cfg)
merger = Merger(cfg)

if torch.cuda.is_available():
    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = torch.nn.DataParallel(decoder).cuda()
    refiner = torch.nn.DataParallel(refiner).cuda()
    merger = torch.nn.DataParallel(merger).cuda()

weight_path = "Pix2Vox-A-ShapeNet.pth"
checkpoint = torch.load(weight_path)
epoch_idx = checkpoint['epoch_idx']
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
refiner.load_state_dict(checkpoint['refiner_state_dict'])
merger.load_state_dict(checkpoint['merger_state_dict'])

encoder.eval()
decoder.eval()
refiner.eval()
merger.eval()

IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

imageTransformation = utils.data_transforms.Compose([
    utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    #utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE), 
    #utils.data_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    utils.data_transforms.ToTensor(),
])

with torch.no_grad():
    # Get data from data loader

    path_img = './test_car'
    ft = 'png'
    imgs = loadImgs_plus(path_img, ft, grayscale=False)
    
    print("imgs shape:", imgs.shape)
    rendering_images = torch.tensor(imgs)        # n x 224 x 224 x3
    #rendering_images = imageTransformation(imgs)
    rendering_images = rendering_images.expand(1, *rendering_images.shape)
    rendering_images = utils.network_utils.var_or_cuda(rendering_images)
    #ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

    # Test the encoder, decoder, refiner and merger
    print("rendering_images:", rendering_images.shape)
    
    image_features = encoder(rendering_images.permute(0, 1, 4, 2, 3))
    raw_features, generated_volume = decoder(image_features)
    generated_volume = merger(raw_features, generated_volume)
    generated_volume = refiner(generated_volume)

    
    img_dir = "./result"
    gv = generated_volume.cpu().numpy()
    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'test'),
                                                                  epoch_idx)
    #test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx)
    #gtv = ground_truth_volume.cpu().numpy()
    #rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test'),
                                                                  #epoch_idx)
    #test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx)


