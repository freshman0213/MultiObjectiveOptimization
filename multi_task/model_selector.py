from models.multi_lenet import MultiLeNetO, MultiLeNetR, MultiFilmLeNetR
from models.segnet import SegnetEncoder, SegnetInstanceDecoder, SegnetSegmentationDecoder, SegnetDepthDecoder
from models.pspnet import SegmentationDecoder, get_segmentation_encoder
from models.multi_faces_resnet import ResNet, FaceAttributeDecoder, BasicBlock
import torchvision.models as model_collection
import torch.nn as nn
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(params):
    data = params['dataset']
    if 'mnist_film' in data:
        print("CALL MultiFilmLeNetR")
        model = {}
        model['rep'] = MultiFilmLeNetR()
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        model['L'] = MultiLeNetO()
        if params['parallel']:
            model['L'] = nn.DataParallel(model['L'])
        model['L'].to(DEVICE)
        model['R'] = model['L']
        return model
    if 'mnist' in data:
        print("CALL MultiLeNetR")
        model = {}
        model['rep'] = MultiLeNetR()
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        if 'L' in params['tasks']:
            model['L'] = MultiLeNetO()
            if params['parallel']:
                model['L'] = nn.DataParallel(model['L'])
            model['L'].to(DEVICE)
        if 'R' in params['tasks']:
            model['R'] = MultiLeNetO()
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['R'].to(DEVICE)
        return model
    if 'cityscapes' in data:
        model = {}
        model['rep'] = get_segmentation_encoder() # SegnetEncoder()
        #vgg16 = model_collection.vgg16(pretrained=True)
        #model['rep'].init_vgg16_params(vgg16)
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        if 'S' in params['tasks']:
            model['S'] = SegmentationDecoder(num_class=19, task_type='C')
            if params['parallel']:
                model['S'] = nn.DataParallel(model['S'])
            model['S'].to(DEVICE)
        if 'I' in params['tasks']:
            model['I'] = SegmentationDecoder(num_class=2, task_type='R')
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['I'].to(DEVICE)
        if 'D' in params['tasks']:
            model['D'] = SegmentationDecoder(num_class=1, task_type='R')
            if params['parallel']:
                model['D'] = nn.DataParallel(model['D'])
            model['D'].to(DEVICE)
        return model

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2,2,2,2])
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].to(DEVICE)
        return model

