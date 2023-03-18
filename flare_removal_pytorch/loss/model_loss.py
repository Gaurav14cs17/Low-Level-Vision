from collections import namedtuple
import torch
from torchvision import models as tv
from torch.nn import functional as F
import torch.nn as nn


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


class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.resnet_model = resnet()
        self.layer_weights = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]
        self.perceptual_weight = loss_weight

    def forward(self, x, gt):
        x_features = self.resnet_model(x)
        gt_features = self.resnet_model(gt.detach())
        percep_loss = 0
        for k in range(len(self.layer_weights)):
            percep_loss += F.mse_loss(x_features[k], gt_features[k], reduction='mean') * self.layer_weights[k]
        percep_loss *= self.perceptual_weight
        return percep_loss


if __name__ == '__main__':
    image_in = torch.randn((1, 3, 225, 225))
    image_gt = torch.randn((1, 3, 225, 225))
    model = PerceptualLoss()
    model_output = model(image_in, image_gt)
    print(model_output)
