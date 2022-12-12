from models.multi_lenet import MultiLeNetO, MultiLeNetR, MultiFilmLeNetR
from models.multi_faces_resnet import ResNet, ResNetFilm, FaceAttributeDecoder, BasicBlock, BasicBlockFilm
from models.vgg5 import MultiVgg5R, MultiVgg5R_film, MultiVgg5O
import torchvision.models as model_collection
import torch.nn as nn
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(params):
    data = params['dataset']
    if 'mnist_film' in data:
        print("CALL MultiFilmLeNetR")
        model = {}
        model['rep'] = MultiFilmLeNetR(params)
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        if 'L' in params['tasks']:
            model['L'] = MultiLeNetO(params)
            if params['parallel']:
                model['L'] = nn.DataParallel(model['L'])
            model['L'].to(DEVICE)
        if 'R' in params['tasks']:
            model['R'] = MultiLeNetO(params)
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['R'].to(DEVICE)
        return model
    if 'mnist' in data:
        print("CALL MultiLeNetR")
        model = {}
        model['rep'] = MultiLeNetR(params)
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        if 'L' in params['tasks']:
            model['L'] = MultiLeNetO(params)
            if params['parallel']:
                model['L'] = nn.DataParallel(model['L'])
            model['L'].to(DEVICE)
        if 'R' in params['tasks']:
            model['R'] = MultiLeNetO(params)
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['R'].to(DEVICE)
        return model

    if 'celeba_film' in data:
        model = {}
        model['rep'] = ResNetFilm(BasicBlockFilm, [2,2,2,2])
        print(model['rep'])
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].to(DEVICE)
        return model
    
    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2,2,2,2])
        print(model['rep'])
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].to(DEVICE)
        return model

    if 'cifar_svhn_film' in data:
        model = {}
        model['rep'] = MultiVgg5R_film()
        print(model['rep'])
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        for t in params['tasks']:
            model[t] = MultiVgg5O()
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].to(DEVICE)
        return model
    if 'cifar_svhn' in data:
        model = {}
        model['rep'] = MultiVgg5R()
        print(model['rep'])
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(DEVICE)
        for t in params['tasks']:
            model[t] = MultiVgg5O()
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].to(DEVICE)
        return model
    
