# import torch.nn as nn
# from math import sqrt
# import torch
#
#
#
# class DnCNN(nn.Module):
#     def __init__(self, channels, num_of_layers=17):
#         super(DnCNN, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         layers = []
#         layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
#                                 bias=False))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(num_of_layers - 2):
#             layers.append(
#                 nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
#                           bias=False))
#             layers.append(nn.BatchNorm2d(features))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
#                                 bias=False))
#         self.dncnn = nn.Sequential(*layers)
#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
#
#     def forward(self, x):
#         out = self.dncnn(x)
#         return out
#
#
# class GlobalMaxPool(torch.nn.Module):
#     def __init__(self):
#         super(GlobalMaxPool, self).__init__()
#
#     def forward(self, input):
#         output = torch.max(input, dim=1)[0]
#
#         return torch.unsqueeze(output, 1)
#
#
# class CA(nn.Module):
#     def __init__(self, channel, reduction=1):
#         super(CA, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
#
# class RB(nn.Module):
#     def __init__(self, features):
#         super(RB, self).__init__()
#         layers = []
#         kernel_size = 3
#         for _ in range(1):
#             layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
#                                     padding=kernel_size // 2, bias=True))
#             layers.append(nn.PReLU())
#             layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
#                                     padding=kernel_size // 2, bias=True))
#         self.res = nn.Sequential(*layers)
#         self.ca = CA(features)
#
#     def forward(self, x):
#         out = self.res(x)
#         out = self.ca(out)
#         out += x
#         return out
#
#
# class _down(nn.Module):
#     def __init__(self, channel_in):
#         super(_down, self).__init__()
#         self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2 * channel_in, kernel_size=3, stride=2, padding=1)
#         self.relu = nn.PReLU()
#
#     def forward(self, x):
#         out = self.relu(self.conv(x))
#         return out
#
#
# class _up(nn.Module):
#     def __init__(self, channel_in):
#         super(_up, self).__init__()
#         self.conv = nn.PixelShuffle(2)
#         self.relu = nn.PReLU()
#
#     def forward(self, x):
#         out = self.relu(self.conv(x))
#         return out
#
#
# class AB(nn.Module):
#     def __init__(self, features):
#         super(AB, self).__init__()
#         num = 2
#         self.DCR_block1 = self.make_layer(RB, features, num)
#         self.down1 = self.make_layer(_down, features, 1)
#         self.DCR_block2 = self.make_layer(RB, features * 2, num)
#         self.down2 = self.make_layer(_down, features * 2, 1)
#         self.DCR_block3 = self.make_layer(RB, features * 4, num)
#         self.up2 = self.make_layer(_up, features * 8, 1)
#         self.DCR_block22 = self.make_layer(RB, features * 4, num)
#         self.up1 = self.make_layer(_up, features * 4, 1)
#         self.DCR_block11 = self.make_layer(RB, features * 2, num)
#         self.conv_f = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
#         self.relu2 = nn.Sigmoid()
#
#     def make_layer(self, block, channel_in, num):
#         layers = []
#         for _ in range(num):
#             layers.append(block(channel_in))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         conc1 = self.DCR_block1(x)
#         out = self.down1(conc1)
#         conc2 = self.DCR_block2(out)
#         conc3 = self.down2(conc2)
#         out = self.DCR_block3(conc3)
#         out = torch.cat([conc3, out], 1)
#         out = self.up2(out)
#         out = torch.cat([conc2, out], 1)
#         out = self.DCR_block22(out)
#         out = self.up1(out)
#         out = torch.cat([conc1, out], 1)
#         out = self.DCR_block11(out)
#         out = self.relu2(self.conv_f(out))
#         # out += x
#         return out



import torch
import torch.nn as nn


class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class AB(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]
        #nb_filter = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = CONV_Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = CONV_Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = CONV_Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = CONV_Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = CONV_Block(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv3_1 = CONV_Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = CONV_Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = CONV_Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = CONV_Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 3, kernel_size=1)
        self.output_layer = nn.Sigmoid()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels):
        return block(dilation_series, padding_series, NoLabels)

    def forward(self, input):
        x0_0 = self.conv0_0(input)   #64 256 256
        x1_0 = self.conv1_0(self.pool(x0_0)) # 128 128 128
        x2_0 = self.conv2_0(self.pool(x1_0)) # 256 64
        x3_0 = self.conv3_0(self.pool(x2_0)) # 512 32
        x4_0 = self.conv4_0(self.pool(x3_0)) # 1024 16

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1)) #32
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = self.output_layer(output)
        return output