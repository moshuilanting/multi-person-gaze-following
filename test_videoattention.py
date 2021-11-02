#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from videoattention_dataset import VideoAttention
from ghost_rock_model_share import ghostnet
import os
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
from utils import imutils, evaluation,misc
from scipy.misc import imresize
#from thop import profile
from skimage import transform as skimage_transform
import time
from soft_argmax import softargmax2d

input_resolution = 224
output_resolution = 64
batch_size = 48
gpu_id = 0
lr = 0.0005
epochs = 201
print_every = 200
eval_every = 1400
save_every = 2
videoattention_train_data = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/images"
videoattention_train_label = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/annotations/videoattention_train.txt"
videoattention_val_data = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/images"
videoattention_val_label = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/annotations/videoattention_test.txt"


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)



if __name__=='__main__':


    transform = _get_transform()

    train_data = VideoAttention(videoattention_train_data, videoattention_train_label, transform, input_size=input_resolution,
                      output_size=output_resolution)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=12)

    val_dataset = VideoAttention(videoattention_val_data, videoattention_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)
    print("read dataset")

    np.random.seed(1)
    # Define device
    device = torch.device('cuda', gpu_id)

    model = ghostnet()
    model_dict = model.state_dict()
    pretrained_dict = torch.load('videoattention_ghost.pt',map_location={'cuda:3':'cuda:0'})
    pretrained_dict = pretrained_dict['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda().to(device).to(torch.float64)
    model.train(True)
    print("read model")

    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    step = 0
    max_steps = len(train_loader)
    optimizer.zero_grad()
    calc_lr = 0
    min_loss = 1

    start = time.time()
    model.train(False)
    print('Validation in progress ...')
    right = 0
    count = 0
    class_count =0
    class_acc = 0
    AUC = []
    new_AUC = []
    avg_dist_heatmap = []
    min_dist_heatmap = []
    avg_dist_location = []
    min_dist_location = []
    in_vs_out_groundtruth = []
    in_vs_out_pred = []

    with torch.no_grad():
        for val_batch, (val_image, val_face, val_head_channel, val_label_heatmap, _ , val_gaze_inside,val_gaze_location,cont_gaze,imsize) in enumerate(val_loader):
            model.train(False)
            val_images = val_image.cuda().to(device).to(torch.float64)
            val_faces = val_face.cuda().to(device).to(torch.float64)
            val_head_channel = val_head_channel.cuda().to(device).to(torch.float64)

            val_pred_heatmap,val_pred_inout = model(val_images, val_faces,val_head_channel)
            val_pred_heatmap = val_pred_heatmap.squeeze(1)
            val_pred_inout = val_pred_inout.cpu()

            val_pred_location = softargmax2d(val_pred_heatmap,device=device).cpu()
            val_pred_heatmap = val_pred_heatmap.cpu()

            for b_i in range(len(val_gaze_inside)):
                class_count +=1
                if val_gaze_inside[b_i] == torch.argmax(val_pred_inout[b_i]):
                    class_acc +=1
                if int(val_gaze_inside[b_i])== 0:
                    continue

                            
                # AUC: area under curve of ROC
                multi_hot = torch.zeros(output_resolution,output_resolution)
                gaze_x, gaze_y = cont_gaze[b_i][0]
                multi_hot = imutils.draw_labelmap(multi_hot, [gaze_x * output_resolution, gaze_y * output_resolution], 3, type='Gaussian')
                multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
                multi_hot = misc.to_numpy(multi_hot)

                #multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                scaled_heatmap = imresize(val_pred_heatmap[b_i].squeeze().cpu(),(output_resolution, output_resolution), interp='bilinear')
                auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)

                #fusion_scaled_heatmap = imresize(head_pred_heatmap,(imsize[b_i][1], imsize[b_i][0]), interp='bilinear')
                #fusion_auc_score = evaluation.auc(fusion_scaled_heatmap, multi_hot)
                #fusion_AUC.append(fusion_auc_score)
                            

                # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                pred_x, pred_y = evaluation.argmax_pts(val_pred_heatmap[b_i])
                norm_p = [pred_y / float(output_resolution), pred_x / float(output_resolution)]
                gt_gaze = val_gaze_location[b_i]  #question????
                avg_distance = evaluation.L2_dist(gt_gaze, norm_p)
                avg_dist_heatmap.append(avg_distance)
                #min_dist_heatmap.append(min(all_distances))
                # average distance: distance between the predicted point and human average point

                pred_y_location,pred_x_location = val_pred_location[b_i]
                norm_p = [pred_y_location / float(output_resolution), pred_x_location / float(output_resolution)]
                gt_gaze = val_gaze_location[b_i]
                all_distances = []
                avg_distance = evaluation.L2_dist(gt_gaze, norm_p)
                avg_dist_location.append(avg_distance)
            in_vs_out_groundtruth.extend(val_gaze_inside.cpu().numpy())
            in_vs_out_pred.extend(val_pred_inout.cpu().numpy())

        end = time.time()
        print('耗时：',end-start)
        print('AUC: ', torch.mean(torch.tensor(AUC)))
        print("AP:{:.4f}\tmin dist heatmap:{:.4f}\tavg dist heatmap:{:.4f}\tmin dist location:{:.4f}\tavg dist location:{:.4f}".format(
                    evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred),
                    torch.mean(torch.tensor(min_dist_heatmap)),
                    torch.mean(torch.tensor(avg_dist_heatmap)),
                    torch.mean(torch.tensor(min_dist_location)),
                    torch.mean(torch.tensor(avg_dist_location))))

