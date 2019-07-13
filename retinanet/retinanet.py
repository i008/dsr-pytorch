import multiprocessing as mp
from math import sqrt

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models


class FPN(nn.Module):
    def __init__(self, encoder_depth, pretrained=True):
        super().__init__()

        if encoder_depth == 34:
            self.encoder = models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise Exception('ResNet depth must be in {34, 50, 101, 152}')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.conv6 = nn.Conv2d(bottom_channel_nr, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(bottom_channel_nr, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(bottom_channel_nr // 2, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(bottom_channel_nr // 4, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))

        p5 = self.toplayer1(p5)
        p4 = self.toplayer2(p4)
        p3 = self.toplayer3(p3)

        return p3, p4, p5, p6, p7

class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, num_classes=20):
        super(RetinaNet, self).__init__()
        self.fpn = FPN(50)
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,224,224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()
