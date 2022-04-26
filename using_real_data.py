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
from torchvision import models
import torchvision.transforms as T
from PIL import Image
from config import cfg
from argparse import ArgumentParser

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
        else:
            im = cv2.imread(fullfs[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        imgs.append(im)

    return np.asarray(imgs)


# from google.colab import files
# uncomment the above line tov use colab's files.upload function
def decode_segmap(image, nc=21):
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_object(image, mask):
    # idx = mask != 9 # chair
    # idx = mask != 2 # bicycle
    idx = mask != 20
    image[idx, 3] = 0


def crop(image, source, nc=21):
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    foreground = cv2.imread(source)

    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    background = 255 * np.ones_like(rgb).astype(np.uint8)

    foreground = foreground.astype(float)
    background = background.astype(float)

    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha = alpha.astype(float) / 255

    foreground = cv2.multiply(alpha, foreground)

    background = cv2.multiply(1.0 - alpha, background)

    outImage = cv2.add(foreground, background)

    return outImage / 255


def mask(img):
    ''' Use the below commented code to upload an image using colab.upload in colab notebook but then
    there is no need to use img parameter and source parameter while using crop function'''
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    trf = T.Compose([
        T.Resize(640),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    inp = trf(img).unsqueeze(0)

    inp_im = inp.resize(inp.shape[2], inp.shape[3], inp.shape[1]).detach().cpu().numpy()
    print("inp:", inp.resize(inp.shape[2], inp.shape[3], inp.shape[1]).detach().cpu().numpy().shape)

    # plt.imshow(inp_im);plt.show()
    out = dlab(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    # result = decode_segmap(om) # use result = crop(om,source) while using crop function discussed later
    # plt.imshow(result); plt.axis('off'); plt.show()
    return om



args = get_args_from_command_line()

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

dlab2 = models.segmentation.deeplabv3_resnet50(pretrained=1).eval()

img = Image.open('./real_world_testing/pinyon-jay-bird.jpg')
img = Image.open("./real_world_testing/cam.jpg")
# plt.imshow(img); plt.axis('off'); plt.show()
results = []
# for i in range(8):
#     img = Image.open("./Pix2Vox-master/test_real_world2/chairs/{}.jpg".format(i))
#     m = mask(img)
#     im = np.asarray(T.Resize(640)(img))
#     im = Image.fromarray(im)
#     im.putalpha(255)
#     im = np.asarray(im)
#     get_object(im, m)
#     # plt.imshow(im); plt.show()
#     results.append(im)
cam = cv2.VideoCapture('./Pix2Vox-master/test_real_world2/screen.mp4')
frame_count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    if frame_count % 30 != 0:
        frame_count += 1
        continue
    else:
        frame_count += 1

    frame = Image.fromarray(frame)
    m = mask(frame)

    im = np.asarray(T.Resize(640)(frame))
    im = Image.fromarray(im)
    im.putalpha(255)
    im = np.asarray(im)
    get_object(im, m)
    # plt.imshow(im); plt.show()
    # cv2.imshow("extracted object", im)
    # cv2.waitKey(1)
    results.append(im)

results = np.asarray(results)

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
    path_img = './test_real_world2/chairs'
    ft = 'png'
    # ft = 'jpg'
    imgs = loadImgs_plus(path_img, ft, grayscale=False)
    
    #if results.shape[0] > 0:
        #print("here")
        #imgs = results
    print("imgs shape:", imgs.shape)
    rendering_images = torch.tensor(imgs)  # n x 224 x 224 x3
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

    print("curr:", current_time)
    pytorch3d.io.obj_io.save_obj(os.path.join(output_dir, 'test/model%s.obj' % str(current_time)), verts, faces)

    gv = generated_volume.cpu().numpy()
    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(output_dir, 'test'), epoch_idx)