# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch 
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def __init__(self, input_shape): 
    super(FiLM, self).__init__()
    self.gammas = nn.parameter.Parameter(torch.randn(input_shape))
    self.betas = nn.parameter.Parameter(torch.randn(input_shape))

  def forward(self, x):
    assert x.dim() == 4 or x.dim() == 2
    if (x.dim() == 4):
        gammas = self.gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = self.betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    elif (x.dim() == 2):
        gammas = self.gammas.expand_as(x)
        betas = self.betas.expand_as(x)
    return (gammas * x) + betas

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_tasks=2):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.films = nn.ModuleList([FiLM(input_shape = (1, self.expansion*planes)) for _ in range(num_tasks)])
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self, x, task):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.films[task](self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetTaskLayer(nn.Module):
    def __init__(self, layers):
        super(ResNetTaskLayer, self).__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x, task):
        for l in self.layers: 
            x = l(x, task)
        return x

class ResNetFilm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetFilm, self).__init__()
        self.in_planes = 64
        self.num_tasks=2

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, num_tasks=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, num_tasks=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, num_tasks=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, num_tasks=2)

    def _make_layer(self, block, planes, num_blocks, stride, num_tasks):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_tasks))
            self.in_planes = planes * block.expansion
        return ResNetTaskLayer(layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out0 = self.layer1(out, 0)
        out0 = self.layer2(out0, 0)
        out0 = self.layer3(out0, 0)
        out0 = self.layer4(out0, 0)
        out0 = F.avg_pool2d(out0, 4)
        out0 = out0.view(out0.size(0), -1)
        out1 = self.layer1(out, 1)
        out1 = self.layer2(out1, 1)
        out1 = self.layer3(out1, 1)
        out1 = self.layer4(out1, 1)
        out1 = F.avg_pool2d(out1, 4)
        out1 = out1.view(out1.size(0), -1)
        return out0, out1


class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(2048, 2)
    
    def forward(self, x):
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out