# 3D shape reconstruction and view through augmented-reality
3D shape reconsruction with silouette-based refinement and viewed thruogh tracking based Augmented reality

- By default, the reconstructed 3D mesh file are stored in ./result/test

## 3d reconstruction step

our silouette guidance implementation is in ./core/train.py

download the the trained model parameters using shapeNet

- https://drive.google.com/file/d/1WyX_saoFLxbwDv3X-suphDWGGQy0TUz_/view?usp=sharing


(optional) download the captured images from single or multi view about ShapeNet models

- http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz


to train your own parameters, download the ShapeNet rendering images and also ground truth voxlized model 
- http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

Specify the two lines below in config.py to your local path, and run 

```
python3 runner.py
```

- __C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'

- __C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'

1.

```

python3 reconstrcut_from_synthetic_data.py 

python3 reconstrcut_from_synthetic_data.py --path ./test_table --out_path ./result --weight_path trained_3d_rec_param.pth --ft png

```
--path

Path of single image/multiple images folder, make sure all images are named only using numbers, default using ./test_recons

--out_path

Path of output folder(should be an exsiting one), the output .obj model will be stored in ./test folder inside it.

--weight_path

The path of the trained model use to perform the 3D reconstruction task

--ft

the file type of input image(s), usually png or jpg, by default it will be jpg


2.

```
python3 reconstruct_from_real_data.py

```
The command arguments and usage are identical as previous one, the only difference is one foreground extraction is used in order to reconstruct from real-world images.

However, the performance is not good for real world ones.

## live Augmented reality step

- After running, you will be asked to press Esc and select the corner of projection planer in clockwise order. Then it will start tracking and render model as the camera is working.

```
python3 live_ar_render.py 
example runner with args:
python3 live_ar_render.py --obj ./Pix2Vox-master/result/test/model.obj --test_param 0 --intrinsic 715 715 320 240 --template_type 2  --camera_id 1
```
--obj specific the path of the obj mesh model to be rendered in the live ar

--test_user_param_before_rendering 

specify if a there would be a static test part for the user interactive parameters, default 0(no need), 1 for performing this step 

--template_type  

estimated template type,1 for square, 2 for rectangle with height greater than width, 3 for rectangle with width greater than height, default 1

--intrinsic

Composed of 4 integers,  for example, for a 3 by 3 camera intrinsic matrix 

715   0     320

0     710   240

0         0         1

The input argument should be --intrinsic 715 710 320 240

--camera_id     

the camera id used for live video capture, default 0

--show_planar_tracking'   

decide if draw lines for tracked planar, 0 for no, 1 for yes, default 0,

