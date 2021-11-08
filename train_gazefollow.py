import torch
import torch.nn as nn
from torchvision import transforms
from gazefollow_dataset_gazedepth import GazeFollow
from ghost_rock_model import ghostnet
import os
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
from utils import imutils, evaluation
from scipy.misc import imresize
#from thop import profile
from skimage import transform as skimage_transform
import time
from soft_argmax import softargmax2d

input_resolution = 224
output_resolution = 64
batch_size = 64
log_dir='gazefollow_logs'
gpu_id = 1
lr = 0.001
epochs = 201
print_every = 200
eval_every = 1000
save_every = 2
gazefollow_train_data = "/home/yours/gazefollow"
gazefollow_train_label = "/home/yours/gazefollow/train_annotations_release.txt"
gazefollow_val_data = "/home/yours/gazefollow"
gazefollow_val_label = "/home/yours/gazefollow/test_annotations_release.txt"


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def huber_loss(x: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    """ Computes a Smooth-L1 loss (Huber loss) in log space

    Shape:
        - X: :math:`(N, C, H, W)` where N is the batch size and C is the number of channels (1 for depth)
        - Target: :math:`(N, C, H, W)`
        - Output: scalar
    """
    log_target = torch.log(torch.clamp(target, min=1e-3, max=20))
    #log_target = torch.clamp(target, min=1e-3, max=10)
    smooth_l1 = nn.SmoothL1Loss(reduction='mean')
    loss = smooth_l1(x, log_target)
    return loss


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

    train_data = GazeFollow(gazefollow_train_data, gazefollow_train_label, transform, input_size=input_resolution,
                      output_size=output_resolution)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_label,
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

    model = ghostnet(inout=True)
    model_dict = model.state_dict()
    model.cuda().to(device)
    model.train(True)
    print("read model")

    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

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
        for batch, (img, face, head_channel, gaze_heatmap, name, gaze_inside,gaze_location) in enumerate(train_loader):
            #start = time.time()
            model.train(True)
            images = img.cuda().to(device)
            faces = face.cuda().to(device)
            head_channel = head_channel.cuda().to(device)
            gaze_location = gaze_location.cuda().to(device)
            gaze_heatmap = gaze_heatmap.cuda().to(device)
            gaze_inside = gaze_inside.cuda(device)
            gaze_inout = gaze_inside.long()


            pred_heatmap,inout_pred = model(images,faces,head_channel)

            pred_heatmap = pred_heatmap.squeeze(1)
            gaze_heatmap = gaze_heatmap.squeeze(1)

            pred_location = softargmax2d(pred_heatmap,device=device).float()

            Xent_loss = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze())*100

            l2_loss_1 = mse_loss(gaze_location, pred_location/output_resolution)
            l2_loss_1 = torch.mean(l2_loss_1, dim=1)
            l2_loss_1 = torch.mul(l2_loss_1,gaze_inside)
            location_loss = torch.sum(l2_loss_1) / torch.sum(gaze_inside) * 10000
            
            l2_loss_2 = mse_loss(gaze_heatmap,pred_heatmap)
            l2_loss_2 = torch.mean(l2_loss_2, dim=1)
            l2_loss_2 = torch.mean(l2_loss_2, dim=1)
            l2_loss_2 = torch.mul(l2_loss_2, gaze_inside)
            heatmap_loss = torch.sum(l2_loss_2) / torch.sum(gaze_inside)*10000 

            total_loss = 0.6*heatmap_loss + 0.4 * location_loss#+Xent_loss

            total_loss.backward() # loss accumulation
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            #end = time.time()
            if batch % print_every ==0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (total_loss){:.4f} (heatmap_loss){:.4f} (location_loss){:.4f} (inout_loss){:.4f}".format(ep, batch + 1,max_steps, total_loss,heatmap_loss,location_loss,Xent_loss))
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
                AUC = []
                avg_dist_heatmap = []
                min_dist_heatmap = []
                avg_dist_location = []
                min_dist_location = []

                with torch.no_grad():
                    for val_batch, (val_image, val_face, val_head_channel, val_label_heatmap, cont_gaze, imsize, _) in enumerate(val_loader):
                        model.train(False)
                        val_images = val_image.cuda().to(device)
                        val_faces = val_face.cuda().to(device)
                        val_head_channel = val_head_channel.cuda().to(device)
                        #val_label_heatmap = val_label_heatmap.cuda().to(device)

                        val_pred_heatmap, val_pred_inout = model(val_images, val_faces,val_head_channel)
                        #val_pred_depthmap = val_pred_depthmap.squeeze(1)
                        val_pred_heatmap = val_pred_heatmap.squeeze(1)

                        val_pred_location = softargmax2d(val_pred_heatmap,device=device).cpu()
                        #val_pred_depthmap = val_pred_depthmap.cpu()
                        val_pred_heatmap = val_pred_heatmap.cpu()
                        #val_depth = val_depth.cpu()

                        for b_i in range(len(cont_gaze)):

                            # remove padding and recover valid ground truth points
                            valid_gaze = cont_gaze[b_i]
                            valid_gaze = valid_gaze[valid_gaze != -1].view(-1, 2)

                            '''
                            # AUC: area under curve of ROC
                            multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                            scaled_heatmap = imresize(val_pred_heatmap[b_i],(imsize[b_i][1], imsize[b_i][0]), interp='bilinear')
                            auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                            AUC.append(auc_score)

                            fusion_scaled_heatmap = imresize(head_pred_heatmap,(imsize[b_i][1], imsize[b_i][0]), interp='bilinear')
                            fusion_auc_score = evaluation.auc(fusion_scaled_heatmap, multi_hot)
                            fusion_AUC.append(fusion_auc_score)
                            '''

                            # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                            pred_x, pred_y = evaluation.argmax_pts(val_pred_heatmap[b_i])
                            norm_p = [pred_x / float(output_resolution), pred_y / float(output_resolution)]
                            all_distances = []
                            for gt_gaze in valid_gaze:
                                all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                            min_dist_heatmap.append(min(all_distances))
                            # average distance: distance between the predicted point and human average point
                            mean_gt_gaze = torch.mean(valid_gaze, 0)
                            avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                            avg_dist_heatmap.append(avg_distance)

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

                end = time.time()
                print('need time：',end-start)
                print("\tmin dist heatmap:{:.4f}\tavg dist heatmap:{:.4f}\tmin dist location:{:.4f}\tavg dist location:{:.4f}".format(
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
                      if calc_lr >= 10:
                            scheduler.step()
                            calc_lr = 5


        if ep % save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep + 1)))
