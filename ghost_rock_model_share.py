import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class auxiliary_inout(nn.Module):
    """Module for scene predictions
    """
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.channels = channels
        self.bn_out = nn.BatchNorm2d(64, momentum=0.01, track_running_stats=True)

        self.scene_in_class = nn.Conv2d(in_channels=128, out_channels=self.channels, kernel_size=1)
        self.scene_out_class = nn.Conv2d(in_channels=self.channels, out_channels=64, kernel_size=1)

        # Added test conv layers
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.01, track_running_stats=True)
        self.pred_linear = nn.Linear(self.channels*49,self.channels)

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

        x = self.scene_in_class(x)

        #pred = torch.mean(torch.flatten(x, start_dim=2), dim=-1)
        pred = self.pred_linear(torch.flatten(x,start_dim=1))

        x = self.scene_out_class(x)
        x = self.bn_out(x)
        x = F.relu(x)

        return x, pred




class GhostNet_rock(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2,shared = False, inout=True):
        super(GhostNet_rock, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.shared = shared
        self.inout = inout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        self.scene_conv1 = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.scene_bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)

        self.face_conv1 = nn.Conv2d(4, output_channel, 3, 2, 1, bias=False)
        self.face_bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        scene_stages = []
        face_stages = []
        scene_block = GhostBottleneck
        face_block = GhostBottleneck
        for cfg in self.cfgs:
            scene_layers = []
            face_layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                scene_layers.append(scene_block(input_channel, hidden_channel, output_channel, k, s,se_ratio=se_ratio))
                face_layers.append(face_block(input_channel, hidden_channel, output_channel, k, s,se_ratio=se_ratio))
                input_channel = output_channel
            scene_stages.append(nn.Sequential(*scene_layers))
            face_stages.append(nn.Sequential(*face_layers))

        #output_channel = _make_divisible(exp_size * width, 4)
        #stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        #input_channel = output_channel

        self.scene_layer1 = nn.Sequential(*scene_stages[0])
        self.scene_layer2 = nn.Sequential(*scene_stages[1:3])
        self.scene_layer3 = nn.Sequential(*scene_stages[3:5])
        self.scene_layer4 = nn.Sequential(*scene_stages[5:7])
        self.scene_layer5 = nn.Sequential(*scene_stages[7:])
        #self.blocks = nn.Sequential(*stages)

        self.face_layer1 = nn.Sequential(*face_stages[0])
        self.face_layer2 = nn.Sequential(*face_stages[1:3])
        self.face_layer3 = nn.Sequential(*face_stages[3:5])
        self.face_layer4 = nn.Sequential(*face_stages[5:7])
        self.face_layer5 = nn.Sequential(*face_stages[7:])

        if self.inout==True:
            self.auxiliary_inout = auxiliary_inout(1)

        self.relu = nn.ReLU(inplace=True)
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.compress_conv1 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(256)
        self.compress_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(128)

        # decoding for saliency
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(32)

        # inout information fuse
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_bn5 = nn.BatchNorm2d(1)
        self.conv7 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, images,faces,head):

        #x = torch.unsqueeze(x,0)

        images = self.scene_conv1(images)
        images = self.scene_bn1(images)
        images = self.act1(images)

        scene_channel = self.scene_layer1(images)
        scene_channel = self.scene_layer2(scene_channel)
        scene_channel = self.scene_layer3(scene_channel)
        scene_channel = self.scene_layer4(scene_channel)
        scene_channel = self.scene_layer5(scene_channel)

        if self.shared == True:  # [batch,n, weight,height,channel]
            heatmap_result = []
            inout_result = []
            for i in range(len(faces)):
                #print(i,'/',len(faces))
                face ,head_mask = faces[i],head[i]
                face_feature = self.face_conv1(torch.cat((face, head_mask),dim=1))
                face_feature = self.face_bn1(face_feature)
                face_feature = self.act1(face_feature)

                face_channel = self.face_layer1(face_feature)
                face_channel = self.face_layer2(face_channel)
                face_channel = self.face_layer3(face_channel)
                face_channel = self.face_layer4(face_channel)
                face_channel = self.face_layer5(face_channel)
                scene_face_feature = torch.cat((scene_channel, face_channel), 1)

                encoding = self.compress_conv1(scene_face_feature)
                encoding = self.compress_bn1(encoding)
                encoding = self.relu(encoding)
                encoding = self.compress_conv2(encoding)
                encoding = self.compress_bn2(encoding)
                encoding = self.relu(encoding)

                if self.inout == True:
                    inout_featrue, inout_pred = self.auxiliary_inout(encoding)

                x = self.deconv1(encoding)
                x = self.deconv_bn1(x)
                x = self.relu(x)
                x = self.deconv2(x)
                x = self.deconv_bn2(x)
                x = self.relu(x)
                x = self.deconv3(x)
                x = self.deconv_bn3(x)
                x = self.relu(x)

                x = self.conv5(x)
                x = self.conv_bn5(x)
                x = self.conv7(x)
                heatmap_result.append(x)
                inout_result.append(inout_pred)

            if self.inout == True:
                return heatmap_result,inout_result
            else:
                return heatmap_result

        if self.shared == False:
            face, head_mask = faces, head
            face_feature = self.face_conv1(torch.cat((face, head_mask), dim=1))
            face_feature = self.face_bn1(face_feature)
            face_feature = self.act1(face_feature)

            face_channel = self.face_layer1(face_feature)
            face_channel = self.face_layer2(face_channel)
            face_channel = self.face_layer3(face_channel)
            face_channel = self.face_layer4(face_channel)
            face_channel = self.face_layer5(face_channel)

            scene_face_feature = torch.cat((scene_channel, face_channel), 1)

            encoding = self.compress_conv1(scene_face_feature)
            encoding = self.compress_bn1(encoding)
            encoding = self.relu(encoding)
            encoding = self.compress_conv2(encoding)
            encoding = self.compress_bn2(encoding)
            encoding = self.relu(encoding)

            if self.inout == True:
                inout_featrue, inout_pred = self.auxiliary_inout(encoding)

            x = self.deconv1(encoding)
            x = self.deconv_bn1(x)
            x = self.relu(x)
            x = self.deconv2(x)
            x = self.deconv_bn2(x)
            x = self.relu(x)
            x = self.deconv3(x)
            x = self.deconv_bn3(x)
            x = self.relu(x)

            x = self.conv5(x)
            x = self.conv_bn5(x)
            x = self.conv7(x)

            if self.inout == True:
                return x, inout_pred
            else:
                return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet_rock(cfgs, **kwargs)


if __name__=='__main__':
    model = ghostnet()
    # input_1 = torch.randn(1,3,224, 224)
    # input_2 = torch.randn(1, 3, 224, 224)
    # input_3 = torch.randn(1, 1, 224, 224)
    # from thop import profile
    # flops, params = profile(model, inputs=(input_1,input_2,input_3))
    # print(flops,params)
    checkpoint = {'model': model.state_dict()}
    torch.save(checkpoint, 'ghost_gazetarget.pt')
    # model.eval()
    # print(model)
    # input = torch.randn(1,3,224,224)
    # y,d,i = model(input)
    # print(y.size())
