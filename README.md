# multi-person-gaze-following

# overview
This repo provides PyTorch implementation of our paper: 'Multi-Person Gaze-Following with Numerical Coordinate Regression'

# dataset
GazeFollow:
You can download GazeFollow dataset from [here](http://gazefollow.csail.mit.edu/) and the additional annotation for the GazeFollow dataset from [Attention-Target-Detection](https://www.dropbox.com/s/1mhgpu0x2w5yto6/gazefollow_extended_txtonly.zip?dl=0)

VideoAttentionTarget:
You can download VideoAttentionTarget dataset from [Attention-Target-Detection](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0)

Please adjust the dataset path accordingly in every '.py' file.

#model
You can download resnet-GazeFollow,ghost-GazeFollow,ghost-VideoAttentionTarget, three gaze-following models, and a face-detector model from [Baidu Cloud Disk](https://pan.baidu.com/s/1hPF4IAysQ1iNfrZYZ0mj8Q)
password: wlsj

put the model in master directory

# run
After adjust the dataset path.

train:

>python train_gazefollow.py
or
>python train_videoattention.py

evaluation in dataset:
>python test_gazefollow.py
or
>python test_videoattention.py

test multi-person gaze-following:
>python videoattention_face_gaze_detection_shared_save.py

#reference
[Attention-Target-Detection](https://github.com/ejcgt/attention-target-detection)
[GhostNet](https://github.com/iamhankai/ghostnet.pytorch)
[yolov5-face](https://github.com/deepcam-cn/yolov5-face])
[softargmax](https://github.com/david-wb/softargmax/blob/master/softargmax.py)
