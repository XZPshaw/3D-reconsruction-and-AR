# 3D shape reconstruction and view through augmented-reality
3D shape reconsruction with silouette-based refinement and viewed thruogh tracking based Augmented reality

## How to use our code

python3 live_ar_render.py 

example runner with args
python3 live_ar_render.py --obj ./Pix2Vox-master/result/test/model.obj --test_param 0 --intrinsic 715 715 320 240 --template_type 2  --camera_id 1

--obj specific the path of the obj mesh model to be rendered in the live ar
--test_param 
--template_type  estimated template type,1 for square, 2 for rectangle with height greater than width, 3 for rectangle with width greater than height, default 1
--camera_id     the camera id used for live video capture, default 0

When after running, you will be asked to press Esc and select the corner of projection planer in clockwise order. Then it will start tracking and render model as the camera is working.

##



##