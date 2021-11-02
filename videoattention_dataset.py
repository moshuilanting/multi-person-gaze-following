import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as TF
from utils import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imresize
from skimage import transform as skimage_transform
import math
import time
import random
from math import sqrt

input_resolution = 224
output_resolution = 224
depth_resolution = 64

f = 5.8262448167737955e+02

class VideoAttention(Dataset):
    def __init__(self, data_dir, csv_path, transform, input_size=input_resolution, output_size=output_resolution,depth_size = depth_resolution,
                 test=False, imshow=False):

            column_names = ['path', 'face_x_min', 'face_y_min', 'face_x_max', 'face_y_max', 'gaze_x', 'gaze_y']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df.reset_index(inplace=True)

            self.X_train = df[['path', 'face_x_min', 'face_y_min', 'face_x_max', 'face_y_max']]
            self.Y_train = df[['gaze_x','gaze_y']]
            self.length = len(df)

            self.data_dir = data_dir
            self.transform = transform
            self.test = test
            self.input_size = input_size
            self.output_size = output_size
            self.imshow = imshow
            self.max_gaze_target_depth = 0

    def __getitem__(self, index):
        path,x_min,y_min,x_max,y_max = self.X_train.iloc[index]
        target_x,target_y = self.Y_train.iloc[index]
        if target_x==-1 and target_y==-1:
            gaze_inside = 0
        else:
            gaze_inside = 1

        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        img = Image.open(os.path.join(self.data_dir,path))
        img = img.convert('RGB')
        width,height = img.size

        if_flip = np.random.random_sample()
        gaze_x,gaze_y = target_x/width, target_y/height

        if self.test:
            imsize = torch.IntTensor([width, height])
            cont_gaze =torch.FloatTensor([[gaze_x,gaze_y]])
        else:
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                else:
                    crop_x_min = 0

                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)
                else:
                    crop_y_min = 0

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                # else:
                #     gaze_x = -1; gaze_y = -1

                width, height = crop_width, crop_height

            if if_flip<=0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)

        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                             3,
                                             type='Gaussian')

        gaze_location = torch.tensor([gaze_y, gaze_x])
        #gaze_target = torch.tensor([target_y,target_x])
        if self.test ==True:
            return img,face,head_channel,gaze_heatmap,path,gaze_inside,gaze_location,cont_gaze,imsize

        return img, face, head_channel, gaze_heatmap, path, gaze_inside, gaze_location

    def __len__(self):
        return self.length

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def DepthNorm(x, maxDepth):
    y = 1000/x  # * 741.2+ 351
    return y


if __name__ == '__main__':
    input_resolution = 224
    output_resolution = 64

    videoattention_train_data = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/images"
    videoattention_train_label = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/annotations/videoattention_train.txt"
    videoattention_val_data = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/images"
    videoattention_val_label = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/annotations/videoattention_test.txt"

    transform = _get_transform()

    # data = GazeFollow(gazefollow_train_data, gazefollow_train_label, transform, input_size=input_resolution,
    #                   output_size=output_resolution)
    data = VideoAttention(videoattention_train_data, videoattention_train_label, transform, input_size=input_resolution,
                      output_size=output_resolution, test=False)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=0)
    for i in range(1):
        start = time.time()
        for batch, (img, face, head_channel, gaze_heatmap, _ , gaze_inside,gaze_location) in enumerate(train_loader):
            # head_channel:[batch_size,1,input_resolution,output_resolution]
            # gaze_heatmap:[batch_size,output_resolution,output_resolution]
            # gaze_inside：是否在图像内部
            print(batch)
            #pass
            #print(depth.shape)
            plt.subplot(131)
            plt.imshow(np.transpose(img[0], (1, 2, 0)))
            plt.subplot(132)
            plt.imshow(np.transpose(face[0], (1, 2, 0)))
            # # #plt.imshow(gaze_heatmap[0])
            # # #plt.imshow(np.transpose(head_channel[0], (1, 2, 0)))
            plt.subplot(133)
            print(gaze_inside,gaze_location)
            plt.imshow(gaze_heatmap[0])
            plt.show()
            # if batch>10:
            #     break
        end = time.time()
        print('耗时：',end-start)
