from collections import namedtuple
import torch
from torchvision import models as tv
from torch.nn import functional as F
import torch.nn as nn
from torchvision.models import vgg19

class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if (num == 18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif (num == 34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif (num == 50):
            self.net = tv.resnet50(pretrained=pretrained)
        elif (num == 101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif (num == 152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        # outputs = namedtuple("Outputs", ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = [h_relu1, h_conv2, h_conv3, h_conv4, h_conv5]
        return out


# class PerceptualLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super(PerceptualLoss, self).__init__()
#         self.resnet_model = resnet()
#         self.layer_weights = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]
#         self.perceptual_weight = loss_weight
#
#     def forward(self, x, gt):
#         x_features = self.resnet_model(x)
#         gt_features = self.resnet_model(gt.detach())
#         percep_loss = 0
#         for k in range(len(self.layer_weights)):
#             percep_loss += F.l1_loss(x_features[k], gt_features[k]) * self.layer_weights[k]
#         #percep_loss *= self.perceptual_weight
#         return percep_loss


class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        #model = model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.selected_feature_index = [2, 7, 12, 21, 30]
        self.layer_weight = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]

    def extract_feature(self, x):
        selected_features = []
        for i, model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, source, target):
        source_feature = self.extract_feature(source)
        target_feature = self.extract_feature(target)
        len_feature = len(source_feature)
        perceptual_loss = 0
        for i in range(len_feature):
            perceptual_loss += self.mae_loss(source_feature[i], target_feature[i]) * self.layer_weight[i]
        return self.loss_weight * perceptual_loss


if __name__ == '__main__':
    image_in = torch.randn((1, 3, 225, 225))
    image_gt = torch.randn((1, 3, 225, 225))
    model = PerceptualLoss()
    model_output = model(image_in, image_gt)
    print(model_output)
