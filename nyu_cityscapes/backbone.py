import torch, sys
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
from model.resnet_dilated import ResnetDilated, ResnetDilated_film
from model.aspp import DeepLabHead
from model.resnet import Bottleneck, conv1x1

sys.path.append('../utils')
from basemodel import BaseModel

def build_model(dataset, model, weighting, random_distribution=None):
    if model == 'DMTL_film':                # film appended network
        model = DeepLabv3_film(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'DMTL':
        model = DeepLabv3(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'MTAN':
        model = MTANDeepLabv3(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'NDDRCNN':
        model = NDDRCNN(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    elif model == 'Cross_Stitch':
        model = Cross_Stitch(dataset=dataset, weighting=weighting, random_distribution=random_distribution)
    return model

class DeepLabv3(BaseModel):
    def __init__(self, dataset='NYUv2', weighting=None, random_distribution=None):
        
        ch = [256, 512, 1024, 2048]
        
        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        self.task_num = len(self.tasks)
        
        super(DeepLabv3, self).__init__(task_num=self.task_num,
                                        weighting=weighting,
                                        random_distribution=random_distribution)
        
        self.backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        self.rep = x
        if self.rep_detach:
            for tn in range(self.task_num):
                self.rep_i[tn] = self.rep.detach().clone()
                self.rep_i[tn].requires_grad = True
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](self.rep_i[i] if self.rep_detach else x), 
                                   img_size, mode='bilinear', align_corners=True)
            if t in ['segmentation', 'segment_semantic']:
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()


class DeepLabv3_film(BaseModel):
    def __init__(self, dataset='NYUv2', weighting=None, random_distribution=None):
        
        ch = [256, 512, 1024, 2048]
        
        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        self.task_num = len(self.tasks)
        
        super(DeepLabv3_film, self).__init__(task_num=self.task_num,
                                        weighting=weighting,
                                        random_distribution=random_distribution)
        
        self.backbone = ResnetDilated_film(resnet.__dict__['resnet50'](pretrained=True))
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        if self.rep_detach:
            for tn in range(self.task_num):
                self.rep_i[tn] = self.rep.detach().clone()
                self.rep_i[tn].requires_grad = True
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](self.rep_i[i] if self.rep_detach else x[i]), 
                                   img_size, mode='bilinear', align_corners=True)
            if t in ['segmentation', 'segment_semantic']:
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()