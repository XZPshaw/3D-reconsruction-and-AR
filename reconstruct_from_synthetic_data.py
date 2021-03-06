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
from argparse import ArgumentParser

from datetime import datetime as dt

import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
import re
import cv2

from config import cfg

import pytorch3d.ops
import pytorch3d.io
import datetime




def get_args_from_command_line():
    parser = ArgumentParser(description='params for reconstructing')
    parser.add_argument('--path',
                        dest='path',
                        help='Path of image folder',
                        default='./test_recons',
                        type=str)

    parser.add_argument('--out_path',
                        dest='out_path',
                        help='Path of output folder(exsiting one)',
                        default='./result',
                        type=str)


    parser.add_argument('--weight_path',
                        dest='weight_path',
                        help='file of trained parameters(.pth)',
                        default='trained_3d_rec_param.pth',
                        type=str)

    parser.add_argument('--ft',
                        dest='ft',
                        help='the file type of input image(s), like png or jpg',
                        default='png',
                        type=str)
    
     
    args = parser.parse_args()
    return args

def loadImgs_plus(path_im, keyword="", grayscale=False):
    fs = []
    fullfs = []

    files = os.listdir(path_im)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for file in files:
        if file.find(keyword) != -1:
            fs.append(file)
            fullfs.append(path_im + "/" + file)
        
    imgs = []
    for i in range(len(fullfs)):
        print("loading file:", fs[i], end='\r')
        if grayscale:
            im = cv2.imread(fullfs[i], 0)   
            im = cv2.resize(im,(137,137))
            
        else:
            im = cv2.imread(fullfs[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            
            im = cv2.resize(im,(137,137))
            
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



args = get_args_from_command_line()

weight_path = args.weight_path

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
    utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE), 
    utils.data_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    utils.data_transforms.ToTensor(),
])



with torch.no_grad():
    # Get data from data loader
    output_dir = "./result"
    path_img = './test_recons'
    path_img = './test_table'
    ft = 'png'
    
    output_dir = args.out_path
    path_img = args.path
    print(path_img)
    ft = args.ft
    #ft = 'jpg'
    imgs = loadImgs_plus(path_img, ft, grayscale=False)
    
    print("imgs shape:", imgs.shape)
    rendering_images = torch.tensor(imgs)        # n x 224 x 224 x3
    rendering_images = imageTransformation(imgs)
    rendering_images = rendering_images.expand(1, *rendering_images.shape)
    rendering_images = utils.network_utils.var_or_cuda(rendering_images)

    # Test the encoder, decoder, refiner and merger
    print("rendering_images:", rendering_images.shape)
    
    image_features = encoder(rendering_images)
    raw_features, generated_volume = decoder(image_features)
    generated_volume = merger(raw_features, generated_volume)
    generated_volume = refiner(generated_volume)

    meshes = pytorch3d.ops.cubify(generated_volume, 0.2)
    verts = meshes.verts_list()[0]
    faces = meshes.faces_list()[0]
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    print("curr:",current_time)
    pytorch3d.io.obj_io.save_obj(os.path.join(output_dir, 'test/model%s.obj'%str(current_time)),verts,faces)

    gv = generated_volume.cpu().numpy()
    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(output_dir, 'test'), epoch_idx)




