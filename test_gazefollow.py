import torch
from torchvision import transforms
import torch.nn as nn

from gazefollow_dataset_gazedepth import GazeFollow
from ghost_rock_model_share import ghostnet
from resnet_test import get_resnet_rock_model

from utils import imutils, evaluation
from soft_argmax import softargmax2d
import argparse
import os
import numpy as np
from scipy.misc import imresize
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



input_resolution = 224
output_resolution = 64
batch_size = 64
gpu_id = 0

gazefollow_val_data = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/gazefollow"
gazefollow_val_label = "/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/gazefollow/test_annotations_release.txt"


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def test():

    transform = _get_transform()

    # Prepare data
    print("Loading Data")
    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)

    # Define device
    device = torch.device('cuda', gpu_id)

    # Load model
    print("Constructing model")
    model = ghostnet(shared=False, inout = False)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('gazefollow_ghost_1314.pt',map_location={'cuda:1':'cuda:0'})
    pretrained_dict = pretrained_dict['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda().to(device)

    print('Evaluation in progress ...')
    #model.train(True)
    AUC = []; min_dist = []; avg_dist = []
    avg_dist_location = []
    min_dist_location = []
    with torch.no_grad():
        for val_batch, (val_img, val_face, val_head_channel, val_label_heatmap, cont_gaze, imsize, _) in enumerate(val_loader):
            print(val_batch)
            model.train(False)
            val_images = val_img.cuda().to(device)
            val_faces = val_face.cuda().to(device)
            val_head_channel = val_head_channel.cuda().to(device)
            val_label_heatmap = val_label_heatmap.cuda().to(device)
            #val_depth = val_depth.cuda().to(device)

            val_pred_heatmap= model(val_images, val_faces, val_head_channel)
            #val_pred_heatmap, val_pred_inout = model(val_images, val_faces, val_head_channel)
            #val_pred_depthmap = val_pred_depthmap.squeeze(1)
            val_pred_heatmap = val_pred_heatmap.squeeze(1)
            val_pred_location = softargmax2d(val_pred_heatmap,device=device).cpu()
            #val_pred_depthmap = val_pred_depthmap.cpu()
            val_pred_heatmap = val_pred_heatmap.cpu()
            #val_depth = val_depth.cpu()
            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(cont_gaze)):
                # remove padding and recover valid ground truth points
                valid_gaze = cont_gaze[b_i]
                valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
                # AUC: area under curve of ROC
                multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                scaled_heatmap = imresize(val_pred_heatmap[b_i], (imsize[b_i][1], imsize[b_i][0]), interp = 'bilinear')
                auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)
                # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                pred_x, pred_y = evaluation.argmax_pts(val_pred_heatmap[b_i])
                norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
                all_distances = []
                for gt_gaze in valid_gaze:
                    all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                min_dist.append(min(all_distances))
                # average distance: distance between the predicted point and human average point
                mean_gt_gaze = torch.mean(valid_gaze, 0)
                avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                avg_dist.append(avg_distance)



                pred_y_location,pred_x_location = val_pred_location[b_i]
                norm_p = [pred_x_location / float(output_resolution), pred_y_location / float(output_resolution)]
                all_distances = []
                for gt_gaze in valid_gaze:
                    all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                min_dist_location.append(min(all_distances))
                # average distance: distance between the predicted point and human average point
                mean_gt_gaze = torch.mean(valid_gaze, 0)
                avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                avg_dist_location.append(avg_distance)



    print("\tAUC:{:.4f}\tavg dist argmax:{:.4f}\tmin dist argmax:{:.4f}\tavg dist NCR:{:.4f}\tmin dist NCR:{:.4f}".format(
          torch.mean(torch.tensor(AUC)),
          torch.mean(torch.tensor(avg_dist)),
          torch.mean(torch.tensor(min_dist)),
          torch.mean(torch.tensor(avg_dist_location)),
          torch.mean(torch.tensor(min_dist_location))))


if __name__ == "__main__":
    test()
