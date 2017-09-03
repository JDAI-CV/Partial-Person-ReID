import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50




class Model(nn.Module):
  def __init__(self, last_conv_stride=1):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
    self.AvgPool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.AvgPool2 = nn.AvgPool2d(kernel_size=4, stride=2, padding=0)
    self.AvgPool3 = nn.AvgPool2d(kernel_size=6, stride=2, padding=0)
    self.AvgPool4 = nn.AvgPool2d(kernel_size=8, stride=2, padding=0)
    self.AvgPool5 = nn.AvgPool2d((16, 8), stride=16)
    #self.conv1 = nn.Conv2d(2048, 256, 1, 1, 0)
    self.bn1 = nn.BatchNorm2d(2048)
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base(x)
    #x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    #x1 = self.AvgPool1(x)
    #x2 = self.AvgPool2(x)
    #x3 = self.AvgPool3(x)
    x4 = self.AvgPool4(x)
    #x5 = self.AvgPool5(x)

    #x1 = x1.view(x1.size(0), x1.size(1), x1.size(2) * x1.size(3))
    #x2 = x2.view(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))
    #x3 = x3.view(x3.size(0), x3.size(1), x3.size(2) * x3.size(3))
    x4 = x4.view(x4.size(0), x4.size(1), x4.size(2) * x4.size(3))
    #x5 = x5.view(x5.size(0), x5.size(1), x5.size(2) * x5.size(3))

    x = F.avg_pool2d(x, x.size()[2:])

    x = x.view(x.size(), -1)  # (b*t, 2048)
    x = x.view(x.size(0) // t, t, -1)
    x = x.mean(1)

    return x
    #spatialFeature = torch.cat((x1, x2), 2)
    # shape [N, C]
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50




class PartialModel(nn.Module):
  def __init__(self, last_conv_stride=1):
    super(PartialModel, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
    self.AvgPool1 = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
    self.AvgPool2 = nn.AvgPool2d(kernel_size=6, stride=3, padding=0)
    self.AvgPool3 = nn.AvgPool2d(kernel_size=9, stride=3, padding=0)
    self.AvgPool4 = nn.AvgPool2d(kernel_size=12, stride=3, padding=0)
    self.AvgPool5 = nn.AvgPool2d((24, 12), stride=24)

    self.bn1 = nn.BatchNorm2d(2048)
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base(x)
    #x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x1 = self.AvgPool1(x)
    x2 = self.AvgPool2(x)


    x1 = x1.view(x1.size(0), x1.size(1), x1.size(2) * x1.size(3))
    x2 = x2.view(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))

    spatialFeature = torch.cat((x1, x2), 2)  # shape [N, C, M] M: the number of multi-scale spatial feature
    x = F.avg_pool2d(x, x.size()[2:]) # Global feature
    x = x.view(x.size(0), -1)
    return x, spatialFeature

