# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import json
import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import numpy as np
import copy
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils import imutils, evaluation
from utils.torch_utils import select_device, load_classifier, time_synchronized
import threading
import os

from PIL import Image
# from ghost_rock_model import ghostnet
from ghost_rock_model_share import ghostnet
import random


color_red = [random.randint(0,255) for i in range(100)]
color_blue =  [random.randint(0,255) for i in range(100)]
color_green =  [random.randint(0,255) for i in range(100)]


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def show_results(img, xywh, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
        # if point_x<x1 or point_y<y1:
        #     cv2.imwrite('1.jpg',img)
        #     print(landmarks[2 * i],landmarks[2 * i+1])
    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def _get_transform(input_resolution=224):
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def find_gaze_object(gaze_location, object_message):
    # object_message[x1,y1,x2,y2,'object_name']
    l = len(object_message)
    # print(l)
    use_distance = 2
    if use_distance == 1:
        _distance = []
        for i in range(l):
            x = (object_message[i][0] + object_message[i][2]) / 2
            y = (object_message[i][1] + object_message[i][3]) / 2
            d = (x - gaze_location[0]) ** 2 + (y - gaze_location[1]) ** 2
            _distance.append(d)
        if len(_distance) > 0:
            min_idx = _distance.index(min(_distance))
            return object_message[min_idx]
        else:
            return None
    # use bounding box
    else:
        in_box = []
        min_box = 500 * 500
        for i in range(l):
            if gaze_location[0] > object_message[i][0] and gaze_location[0] < object_message[i][2] and gaze_location[
                1] > object_message[i][1] and gaze_location[1] < object_message[i][3]:
                w_b = object_message[i][2] - object_message[i][0]
                h_b = object_message[i][3] - object_message[i][1]
                box_space = w_b * h_b
                if min_box > box_space:
                    min_box = box_space
                    in_box.append(object_message[i])

        if len(in_box) > 0:
            return in_box[0]
        else:
            return None


def video_detect():
    global object_msg
    global get_head_realsense_image

    # gaze
    input_resolution = 224
    output_resolution = 64
    color_list = [(color_red[i],color_blue[i],color_green[i]) for i in range(100)]
    gaze_model = ghostnet(shared=True)
    model_dict = gaze_model.state_dict()
    # pretrained_dict = torch.load('model_demo.pt')
    pretrained_dict = torch.load('videoattention_ghost.pt', map_location={'cuda:3': 'cuda:0'})
    pretrained_dict = pretrained_dict['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # pretrained_dict = torch.load('torch1-8_best.pt', map_location={'cuda:1': 'cuda:0'})

    model_dict.update(pretrained_dict)
    gaze_model.load_state_dict(model_dict)
    gaze_model.cuda()
    gaze_model.train(False)

    test_transforms = _get_transform(input_resolution)

    # yolov5-face detect
    video_dir = '/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/videoattentiontarget/annotations/video_1'
    _weights = './yolov5s-face.pt'
    img_size = 1920
    conf_thres = 0.7
    iou_thres = 0.5
    agnostic = False
    hide_labels = False
    hide_conf = False
    line_thickness = 8

    weights, imgsz = _weights, img_size
    print(weights)

    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # for i in tqdm(range(0, nb_frames)):

    videos = os.listdir(video_dir)
    save_path = 'Videoattention_video'
    for video_name in videos:
        print(video_name)

        video_start = time.time()
        face_number = 0
        gaze_use_time = 0
        detect_use_time = 0

        time_test = 0
        video_reader = cv2.VideoCapture(os.path.join(video_dir, video_name))
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        success, frame = video_reader.read()
        flag = True

        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #writer = cv2.VideoWriter(save_path + '/' + video_name[:-4]+'.mp4', fourcc, 16,(frame_w, frame_h), True)
        if os.path.exists(save_path+'/'+video_name[:-4]) == False:
            os.mkdir(save_path+'/'+video_name[:-4])
        count =0
        while flag:
            count+=1
            # image = get_head_realsense_image
            # need_message = object_msg
            _, image = video_reader.read()  # Padded resize
            # print(_)
            if _ == False:
                # flag = False
                break
            img = letterbox(image, imgsz, stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment='store_true')[0]

            # Apply NMS
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)


            # Process detections
            for i, det in enumerate(pred):  # detections per image
                face_number += len(det)
                faces = []
                head_channels = []

                gn = torch.tensor(image.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
                gn_lks = torch.tensor(image.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
                if len(det):

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()

                    current_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    width, height = current_image.size

                    detect_start = time.time()
                    current_frame = test_transforms(current_image)
                    current_frame = current_frame.unsqueeze(0).cuda()

                    face_heads = []
                    for j in range(det.size()[0]):

                        face_left = det[j][0]
                        face_top = det[j][1]
                        face_right = det[j][2]
                        face_bottom = det[j][3]
                        face_left -= (face_right - face_left) * 0.2
                        face_right += (face_right - face_left) * 0.2
                        face_top -= (face_bottom - face_top) * 0.2
                        face_bottom += (face_bottom - face_top) * 0.2

                        face_left = int(face_left)
                        face_top = int(face_top)
                        face_right = int(face_right)
                        face_bottom = int(face_bottom)

                        head_box = [face_left, face_top, face_right, face_bottom]
                        face_heads.append(head_box)

                    face_heads.sort()

                    for qq in range(len(face_heads)):
                        face_region = current_image.crop((face_heads[qq]))

                        head = test_transforms(face_region)
                        face_left, face_top, face_right, face_bottom = face_heads[qq]
                        head_channel = imutils.get_head_box_channel(face_left, face_top, face_right, face_bottom, width,
                                                                    height,
                                                                    resolution=224).unsqueeze(0)

                        faces.append(head.unsqueeze(0).numpy())
                        head_channels.append(head_channel.unsqueeze(0).numpy())


                    heads = np.array(faces)
                    head_channels = np.array(head_channels)
                    heads = torch.from_numpy(heads)
                    head_channels= torch.from_numpy(head_channels)

                    heads=heads.cuda()
                    head_channels = head_channels.cuda()
                    start = time.time()
                    raw_heatmap, raw_inout = gaze_model(current_frame, heads, head_channels)
                    gaze_end = time.time()

                    gaze_use_time += gaze_end - start
                    detect_use_time += start - detect_start


                    for nn in range(len(raw_inout)):
                        raw_hm = raw_heatmap[nn].squeeze().cpu().detach().numpy() * 255
                        raw_hm = raw_hm.squeeze()
                        if raw_inout[nn][0] > -2:
                            c = 0
                        else:
                            c = 1
                        # c = torch.argmax(raw_inout[0])

                        if c == 0:  # or c == 1 or c == 2 or c == 3:
                            cv2.rectangle(image, (face_heads[nn][0], face_heads[nn][1]), (face_heads[nn][2], face_heads[nn][3]), color_list[nn],
                                          line_thickness)
                            pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                            norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                            cv2.circle(image, (int(norm_p[0] * width), int(norm_p[1] * height)), 10, color_list[nn], -1)
                            cv2.line(image, ((face_heads[nn][0] + face_heads[nn][2]) // 2, (face_heads[nn][1] + face_heads[nn][3]) // 2),
                                     (int(norm_p[0] * width), int(norm_p[1] * height)), color_list[nn], line_thickness)

                            # if isinstance(need_message, list):
                            # need_message = json.loads(need_message)
                            # print(need_message)
                            # gaze_target_location = find_gaze_object([int(norm_p[0] * width), int(norm_p[1] * height)],need_message)
                            # print(gaze_target_location)

                        elif c == 1:
                            cv2.rectangle(image, (face_heads[nn][0], face_heads[nn][1]), (face_heads[nn][2], face_heads[nn][3]), color_list[nn],line_thickness)
                        # elif c == 2:
                        #     cv2.rectangle(image, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (225, 225, 0),
                        #                   5)
                        # elif c == 3:
                        #     cv2.rectangle(image, (head_box[0], head_box[1]), (head_box[2], head_box[3]), color_list[i],
                        #                   5)
                    cv2.imwrite(save_path+'/'+video_name[:-4]+'/'+str(count)+'.jpg', image)

                    # show
                    # for j in range(det.size()[0]):
                    #     xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    #     conf = det[j, 4].cpu().numpy()
                    #     landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    #     class_num = det[j, 15].cpu().numpy()
                    #     image = show_results(image, xywh, conf, landmarks, class_num)

            # pub_face = str(det.tolist())
            # print(pub_face)
            # client.publish('face', payload=pub_face, qos=0)
            # print(need_message)
            # cv2.imshow('1', image)
            # cv2.waitKey(1)
        video_end = time.time()

        print(
            "total_time:{:.4f}\tgaze_use_time:{:.4f}\tdetect_use_time:{:.4f}\tnb_frames:{:4d}\t mean_face:{:.2f}".format(
                video_end - video_start, gaze_use_time, detect_use_time, nb_frames - 1, face_number / (nb_frames - 1)))
        #print('detect_use_time 2 :',time_test)




if __name__ == "__main__":

    video_detect()
