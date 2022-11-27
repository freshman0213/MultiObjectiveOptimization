import torch
import torch.nn as nn
from model.resnet import Bottleneck

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

class Bottleneck_filmed(nn.Module):
    def __init__(self, layers, film_size):
        super(Bottleneck_filmed, self).__init__()
        if len(layers) == 7:
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = layers[0]
            self.bn1 = layers[1]
            self.conv2 = layers[2]
            self.bn2 = layers[3]
            self.conv3 = layers[4]
            self.bn3 = layers[5]
            self.film_task1 = FiLM(input_shape = (1,film_size))
            self.film_task2 = FiLM(input_shape = (1,film_size))
            self.relu = layers[6]
            self.downsample = None
        elif len(layers) == 9:
            self.conv1 = layers[0]
            self.bn1 = layers[1]
            self.conv2 = layers[2]
            self.bn2 = layers[3]
            self.conv3 = layers[4]
            self.bn3 = layers[5]
            self.film_task1 = FiLM(input_shape = (1,film_size))
            self.film_task2 = FiLM(input_shape = (1,film_size))
            self.relu = layers[6]
            self.downsample = nn.Sequential(*[layers[7], layers[8]])
        else:
            assert len(layers) == 7 or len(layers) == 9

    def forward(self, x):
        (x1, x2) = x

        identity1 = x1
        identity2 = x2

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        out2 = self.conv1(x2)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)

        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out2 = self.conv3(out2)
        out2 = self.bn3(out2)


        out1 = self.film_task1(out1)
        out2 = self.film_task2(out2)

        if self.downsample is not None:
            identity1 = self.downsample(x1)
            identity2 = self.downsample(x2)

        out1 += identity1
        out2 += identity2
        out1 = self.relu(out1)
        out2 = self.relu(out2)

        return out1, out2



class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pre-defined ResNet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    
    def forward_stage(self, x, stage):
        assert(stage in ['conv','layer1','layer2','layer3','layer4', 'layer1_without_conv'])
        
        if stage == 'conv':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x

        elif stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x
        
        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer(x)

class ResnetDilated_film(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated_film, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.film_size = [256, 512, 1024, 2048]

        # take pre-defined ResNet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        
        self.maxpool = orig_resnet.maxpool
        self.layer1 = self._filmed(orig_resnet.layer1, 0)
        self.layer2 = self._filmed(orig_resnet.layer2, 1)
        self.layer3 = self._filmed(orig_resnet.layer3, 2)
        self.layer4 = self._filmed(orig_resnet.layer4, 3)
        
        

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def _filmed(self, m, layer_number):
        res_block = []
        for module in m.modules():
            layers = []
            if isinstance(module, Bottleneck):
                for mod in module.modules():
                    if not isinstance(mod, Bottleneck) and not isinstance(mod, nn.Sequential):
                        layers.append(mod)
                res_block.append(Bottleneck_filmed(layers, self.film_size[layer_number]))
      
                
        # print(nn.Sequential(*res_block))
        return nn.Sequential(*res_block)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x1, x2 = self.layer1((x, x)) 
        x1, x2 = self.layer2((x1, x2))
        x1, x2 = self.layer3((x1, x2))
        x1, x2 = self.layer4((x1, x2))
        return x1, x2

    
    def forward_stage(self, x, stage):
        assert(stage in ['conv','layer1','layer2','layer3','layer4', 'layer1_without_conv'])
        
        if stage == 'conv':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x

        elif stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x
        
        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer(x)