#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from videoattention_dataset import VideoAttention
from ghost_rock_model_base import ghostnet
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
log_dir='VideoAttention_ghost_logs'
gpu_id = 3
lr = 0.0005
epochs = 201
print_every = 200
eval_every = 1400
save_every = 2
videoattention_train_data = "/home/yours/videoattentiontarget/images"
videoattention_train_label = "/home/yours/videoattentiontarget/annotations/videoattention_train.txt"
videoattention_val_data = "/home/yours/videoattentiontarget/images"
videoattention_val_label = "/home/yours/videoattentiontarget/annotations/videoattention_test.txt"


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def inout_cross_entropy_loss(scene_pred: torch.Tensor,
                             scene_gt: torch.Tensor) -> torch.Tensor:
    """ Computes the cross entropy loss for scenes where scene_pred is a log softmax input using NLL loss

    Shape:
        - scene_pred: :math:`(N, C)` where N is the batch size and C is the number of scene types in the dataset
        - scene_gt: :math:`(N)`
        - Output: scalar

    """
    scene_pred = nn.LogSoftmax(dim=-1)(scene_pred)
    cross_entropy_loss = nn.NLLLoss(reduction='mean')
    loss = cross_entropy_loss(scene_pred, scene_gt)

    return loss


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

    logdir = os.path.join(log_dir,datetime.now().strftime('good_%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)
    np.random.seed(1)
    # Define device
    device = torch.device('cuda', gpu_id)

    model = ghostnet()
    model_dict = model.state_dict()
    model.cuda().to(device).to(torch.float64)
    model.train(True)
    print("read model")

    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()
    # l1_criterion = nn.L1Loss()
    # bcelogit_loss = nn.BCEWithLogitsLoss()
    # cross_loss = nn.CrossEntropyLoss()
    # smooth_crit = nn.SmoothL1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    step = 0
    max_steps = len(train_loader)
    optimizer.zero_grad()
    calc_lr = 0
    min_loss = 1
    print("Training in progress ...")
    for ep in range(epochs):
        print('lr: ',optimizer.state_dict()['param_groups'][0]['lr'],'calc_lr: ',calc_lr)
        for batch, (img, face, head_channel, gaze_heatmap, _ , gaze_inside,gaze_location) in enumerate(train_loader):
            #start = time.time()
            model.train(True)
            images = img.cuda().to(device).to(torch.float64)
            faces = face.cuda().to(device).to(torch.float64)
            head_channel = head_channel.cuda().to(device).to(torch.float64)
            gaze_location = gaze_location.cuda().to(device).to(torch.float64)
            gaze_heatmap = gaze_heatmap.cuda().to(device).to(torch.float64)
            #loss_weight = loss_weight.cuda().to(device)
            gaze_inside = gaze_inside.cuda(device).to(torch.float64)
            gaze_inout = gaze_inside.to(torch.float64)


            #pred
            #print(images.type(),faces.type(),head_channel.type())
            pred_heatmap,pred_inout = model(images,faces,head_channel)
            #print(pred_inout.shape,gaze_inout.shape)
            #pred_depthmap = pred_depthmap.squeeze(1)
            pred_heatmap = pred_heatmap.squeeze(1)
            gaze_heatmap = gaze_heatmap.squeeze(1)

            pred_location = softargmax2d(pred_heatmap,device=device)
            #print(pred_location/output_resolution,gaze_location)
            # loss_direction = mse_loss(pred_direction,gaze_direction)
            # loss_direction = torch.mean(loss_direction, dim=1)
            # loss_direction = torch.mul(loss_direction, gaze_inside)
            # loss_direction = torch.sum(loss_direction) / torch.sum(gaze_inside)*100

            #depth_map_loss = huber_loss(pred_depthmap,depth)*100

            #l2_loss_1 = torch.mean(l2_loss_1, dim=1)
            #l2_loss_1 = torch.mul(l2_loss_1, gaze_inside)
            #l2_loss_1 = torch.sum(l2_loss_1) / torch.sum(gaze_inside)*100

            #inout_loss = inout_cross_entropy_loss(pred_inout,gaze_inout)*100
            #inout_loss = 0
            inout_loss = bcelogit_loss(pred_inout.squeeze(), gaze_inside.squeeze()) * 100

            l2_loss_1 = mse_loss(gaze_location, pred_location/output_resolution)
            l2_loss_1 = torch.mean(l2_loss_1, dim=1)
            l2_loss_1 = torch.mul(l2_loss_1,gaze_inside)
            location_loss = torch.sum(l2_loss_1) / torch.sum(gaze_inside) * 10000 *0.5

            l2_loss_2 = mse_loss(gaze_heatmap,pred_heatmap)
            l2_loss_2 = torch.mean(l2_loss_2, dim=1)
            l2_loss_2 = torch.mean(l2_loss_2, dim=1)
            l2_loss_2 = torch.mul(l2_loss_2, gaze_inside)
            heatmap_loss = torch.sum(l2_loss_2) / torch.sum(gaze_inside)*10000 *0.5

            total_loss =  inout_loss + 2*location_loss#2*heatmap_loss#+location_loss

            
            total_loss.backward() # loss accumulation
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if batch % print_every ==0:

                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (total_loss){:.4f} (heatmap_loss){:.4f} (location_loss){:.4f} (inout_loss){:.4f}".format(ep, batch + 1,max_steps, total_loss,heatmap_loss,location_loss,inout_loss))
                # Tensorboard
                ind = np.random.choice(len(images), replace=False)
                writer.add_scalar("Train Loss", total_loss, global_step=step)
                writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], global_step=step)

            # eval
            if (batch != 0 and batch % eval_every == 0) or batch + 1 == max_steps:
            #print((ep+1) % 50)
            #if (batch != 0 and batch % eval_every == 0) or (ep+1) % 50 == 0:
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
                        #val_label_heatmap = val_label_heatmap.cuda().to(device)
                        #val_gaze_location = val_gaze_location.cuda().to(device).to(torch.float64)

                        val_pred_heatmap,val_pred_inout = model(val_images, val_faces,val_head_channel)
                        #val_pred_depthmap = val_pred_depthmap.squeeze(1)
                        val_pred_heatmap = val_pred_heatmap.squeeze(1)
                        val_pred_inout = val_pred_inout.cpu()

                        val_pred_location = softargmax2d(val_pred_heatmap,device=device).cpu()
                        #val_pred_depthmap = val_pred_depthmap.cpu()
                        val_pred_heatmap = val_pred_heatmap.cpu()
                        #val_depth = val_depth.cpu()

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
                print('need time：',end-start)
                print('AUC: ', torch.mean(torch.tensor(AUC)))
                print("AP:{:.4f}\tmin dist heatmap:{:.4f}\tavg dist heatmap:{:.4f}\tmin dist location:{:.4f}\tavg dist location:{:.4f}".format(
                    evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred),
                    torch.mean(torch.tensor(min_dist_heatmap)),
                    torch.mean(torch.tensor(avg_dist_heatmap)),
                    torch.mean(torch.tensor(min_dist_location)),
                    torch.mean(torch.tensor(avg_dist_location))))


                if min_loss>torch.mean(torch.tensor(avg_dist_heatmap)):
                      calc_lr = 0
                      min_loss = torch.mean(torch.tensor(avg_dist_heatmap))
                      checkpoint = {'model': model.state_dict()}
                      torch.save(checkpoint, os.path.join(logdir, 'light_best_weights.pt'))
                else:
                      calc_lr += 1
                      if calc_lr >= 6:
                            scheduler.step()
                            calc_lr = 2


        if ep % save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep + 1)))
