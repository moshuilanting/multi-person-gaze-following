import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class auxiliary_inout(nn.Module):
    """Module for scene predictions
    """
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.channels = channels
        self.bn_out = nn.BatchNorm2d(64, momentum=0.01, track_running_stats=True)

        self.scene_in = nn.Conv2d(in_channels=128, out_channels=self.channels, kernel_size=1)
        self.scene_out = nn.Conv2d(in_channels=self.channels, out_channels=64, kernel_size=1)

        # Added test conv layers
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.01, track_running_stats=True)

    def forward(self, x):
        """
        Shape:
            - X: :math:`(N, C_in, H, W)` where :math:`C_in = 512`
            - Output: :math:`(N, C_out, H, W)` where :math:`C_out = 2048`
            - Scene pred: :math:`(N, num_scenes)` where :math:`num_scenes = 27`

        """
        # Added conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.scene_in(x)

        pred = torch.mean(torch.flatten(x, start_dim=2), dim=-1)

        x = self.scene_out(x)
        x = self.bn_out(x)
        x = F.relu(x)

        return x, pred





class resnet_model(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(resnet_model, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale

        self.scene_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.scene_bn1 = nn.BatchNorm2d(64)
        self.scene_layer1 = self._scene_make_layer(block, 64, layers[0])
        self.scene_layer2 = self._scene_make_layer(block, 128, layers[1], stride=2)
        self.scene_layer3 = self._scene_make_layer(block, 256, layers[2], stride=2)
        self.scene_layer4 = self._scene_make_layer(block, 512, layers[3], stride=2)
        self.scene_layer5 = self._scene_make_layer(block, 256, layers[4], stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.face_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.face_bn1 = nn.BatchNorm2d(64)
        self.face_layer1 = self._face_make_layer(block, 64, layers[0])
        self.face_layer2 = self._face_make_layer(block, 128, layers[1], stride=2)
        self.face_layer3 = self._face_make_layer(block, 256, layers[2], stride=2)
        self.face_layer4 = self._face_make_layer(block, 512, layers[3], stride=2)
        self.face_layer5 = self._face_make_layer(block, 256, layers[4], stride=1)

        self.auxiliary_inout = auxiliary_inout(1)

        #encoding for saliency
        self._conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self._bn1 = nn.BatchNorm2d(1024)
        self._conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self._bn2 = nn.BatchNorm2d(512)

        # decoding for saliency
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(128)

        # depth information fuse
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv_bn4 = nn.BatchNorm2d(64)

        # inout information fuse
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_bn6 = nn.BatchNorm2d(1)
        self.conv7 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _scene_make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _face_make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def forward(self, images,faces,head):

        #images path
        images = self.scene_conv1(images)
        images = self.scene_bn1(images)
        images = self.relu(images)
        images = self.maxpool(images)

        image_feature_1 = images = self.scene_layer1(images)
        image_feature_2 = images = self.scene_layer2(images)
        image_feature_3 = images = self.scene_layer3(images)
        image_feature_4 = images = self.scene_layer4(images)
        images = self.scene_layer5(images)

        #faces path
        faces = self.face_conv1(torch.cat((faces, head), dim=1))
        faces = self.face_bn1(faces)
        faces = self.relu(faces)
        faces = self.maxpool(faces)

        faces = self.face_layer1(faces)
        faces = self.face_layer2(faces)
        faces = self.face_layer3(faces)
        faces = self.face_layer4(faces)
        faces = self.face_layer5(faces)

        scene_face_feature = torch.cat((images, faces), 1)

        #scene + face feat -> encoding -> decoding
        conv_result = self._conv1(scene_face_feature)
        conv_result = self._bn1(conv_result)
        conv_result = self.relu(conv_result)
        conv_result = self._conv2(conv_result)
        conv_result = self._bn2(conv_result)
        conv_result = self.relu(conv_result)

        x = self.deconv1(conv_result)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)

        inout_featrue, inout_pred = self.auxiliary_inout(x)
        x = self.conv4(x)
        x = self.conv_bn4(x)
        x = x + inout_featrue       #feature can feed back or not
        x = self.conv5(x)
        x = self.conv_bn5(x)
        x = self.conv6(x)
        x = self.conv_bn6(x)
        x = self.conv7(x)


        return x,inout_pred

def get_resnet_rock_model(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = resnet_model(Bottleneck, [3, 4, 6, 3, 2], baseWidth = 26, scale = 4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model
