import os
import errno
import sys
import torch
import click
import json
import datetime
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision
import types

from tqdm import tqdm
from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector

NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_multi_task(params):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    exp_identifier = []
    for (key, val) in params.items():
        if 'tasks' in key:
            continue
        if 'scales' in key:
            for task, scale in val.items():
                exp_identifier += ['scale_{}={}'.format(task, scale)]
        else:
            exp_identifier += ['{}={}'.format(key,val)]

    exp_identifier = '|'.join(exp_identifier)
    params['exp_id'] = exp_identifier

    writer = SummaryWriter(log_dir='runs/{}_{}'.format(params['exp_id'], datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    if 'RMSprop' in params['optimizer']:
        optimizer = torch.optim.RMSprop(model_params, lr=params['lr'])
    elif 'Adam' in params['optimizer']:
        optimizer = torch.optim.Adam(model_params, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=params['lr'], momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.85)

    tasks = params['tasks']
    all_tasks = params['tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    scale = {}
    for t in tasks:
        scale[t] = float(params['scales'][t])
    n_iter = 0
    best_val_loss = np.Inf
    for epoch in tqdm(range(NUM_EPOCHS)):
        start = timer()
        print('Epoch {} Started'.format(epoch))

        for m in model:
            model[m].train()

        ##### Training #####
        for batch in train_loader:
            n_iter += 1
            # First member is always images
            images = batch[0]
            images = Variable(images.to(DEVICE))

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i+1]
                labels[t] = Variable(labels[t].to(DEVICE))

            loss_data = {}
            # Scaled back-propagation
            optimizer.zero_grad()
            if 'mnist' in params['dataset']: 
                rep = model['rep'](images)
                if len(rep) == 2: 
                    for i, t in enumerate(tasks):
                        out_t = model[t](rep[i])
                        loss_t = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss_t.item() 
                        if i > 0:
                            loss = loss + scale[t]*loss_t
                        else:
                            loss = scale[t]*loss_t
                    loss.backward()
                    optimizer.step()
                else:
                    for i, t in enumerate(tasks):
                        out_t = model[t](rep)
                        loss_t = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss_t.item() 
                        if i > 0:
                            loss = loss + scale[t]*loss_t
                        else:
                            loss = scale[t]*loss_t
                    loss.backward()
                    optimizer.step()
            elif 'celeb' in params['dataset']: 
                if 'film' in params['dataset']: 
                    for i, t in enumerate(tasks):
                        rep = model['rep'](images, i)
                        out_t = model[t](rep)
                        loss_t = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss_t.item() 
                        if i > 0:
                            loss = loss + scale[t]*loss_t
                        else:
                            loss = scale[t]*loss_t
                    loss.backward()
                    optimizer.step()
                else: 
                    rep = model['rep'](images, i)
                    for i, t in enumerate(tasks):
                        out_t = model[t](rep)
                        loss_t = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss_t.item() 
                        if i > 0:
                            loss = loss + scale[t]*loss_t
                        else:
                            loss = scale[t]*loss_t
                    loss.backward()
                    optimizer.step()
            else: 
                print('errorrrrrrrr')
            writer.add_scalar('training_loss', loss.item(), n_iter)
            for t in tasks:
                writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

        scheduler.step()

        for m in model:
            model[m].eval()

        ##### Validation #####
        tot_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_val in val_loader:
                val_images = batch_val[0].to(DEVICE)
                labels_val = {}
                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i+1]
                    labels_val[t] = labels_val[t].to(DEVICE)

                if 'mnist' in params['dataset']: 
                    val_rep = model['rep'](val_images)
                    if len(val_rep) == 2:
                        for i, t in enumerate(tasks):
                            out_t_val = model[t](val_rep[i])
                            loss_t = loss_fn[t](out_t_val, labels_val[t], val=True)
                            tot_val_loss += scale[t]*loss_t.item()
                    else:
                        for t in tasks:
                            out_t_val = model[t](val_rep)
                            loss_t = loss_fn[t](out_t_val, labels_val[t], val=True)
                            tot_val_loss += scale[t]*loss_t.item()
                elif 'celeb' in params['dataset']: 
                    if 'film' in params['dataset']: 
                        for i, t in enumerate(tasks):
                            val_rep = model['rep'](val_images, i)
                            out_t_val = model[t](val_rep)
                            loss_t = loss_fn[t](out_t_val, labels_val[t], val=True)
                            tot_val_loss += scale[t]*loss_t.item()
                    else: 
                        val_rep = model['rep'](val_images)
                        for i, t in enumerate(tasks):
                            out_t_val = model[t](val_rep)
                            loss_t = loss_fn[t](out_t_val, labels_val[t], val=True)
                            tot_val_loss += scale[t]*loss_t.item()
                num_val_batches+=1
        writer.add_scalar('validation_loss', tot_val_loss/len(val_dst), n_iter)

        # Early Stopping
        if (tot_val_loss < best_val_loss):
            best_val_loss = tot_val_loss
            state = {'epoch': epoch+1,
                    'model_rep': model['rep'].state_dict(),
                    'optimizer_state' : optimizer.state_dict()}
            for t in tasks:
                key_name = 'model_{}'.format(t)
                state[key_name] = model[t].state_dict()

            

            try:
                os.makedirs(os.path.join('./saved_models/'))
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
            torch.save(state, "./saved_models/{}_model.pkl".format(params['exp_id']))
        
        end = timer()
        print('Epoch ended in {}s'.format(end - start))


def test_multi_task(params, trial_identifier):
    with open('configs.json') as config_params:
        configs = json.load(config_params)
    _, _, _, _, test_loader, test_dst = datasets.get_dataset(params, configs)

    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    state = torch.load("./saved_models/{}_model.pkl".format(trial_identifier))

    model = model_selector.get_model(params)
    model['rep'].load_state_dict(state['model_rep'])

    tasks = params['tasks']
    all_tasks = params['tasks']

    scale = {}
    for t in tasks:
        scale[t] = float(params['scales'][t])

    for t in tasks:
        key_name = 'model_{}'.format(t)
        model[t].load_state_dict(state[key_name])

    # ##### Testing #####
    tot_loss = {}
    tot_loss['all'] = 0.0
    for t in tasks:
        tot_loss[t] = 0.0

    num_test_batches = 0 
    
    testing_metric = {}
    testing_loss = {}

    for m in model:
        model[m].eval()

    with torch.no_grad():
        for batch_test in test_loader:
            test_images = batch_test[0].to(DEVICE)
            labels_test = {}
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels_test[t] = batch_test[i+1]
                labels_test[t] = labels_test[t].to(DEVICE)

            
            if 'mnist' in params['dataset']: 
                test_rep = model['rep'](test_images)
                if len(test_rep) == 2:
                    for i, t in enumerate(tasks):
                        out_t_test = model[t](test_rep[i])
                        loss_t = loss_fn[t](out_t_test, labels_test[t])
                        tot_loss['all'] += scale[t]*loss_t.item()
                        tot_loss[t] += loss_t.item()
                        metric[t].update(out_t_test, labels_test[t])
                else:
                    for t in tasks:
                        out_t_test = model[t](test_rep)
                        loss_t = loss_fn[t](out_t_test, labels_test[t])
                        tot_loss['all'] += scale[t]*loss_t.item()
                        tot_loss[t] += loss_t.item()
                        metric[t].update(out_t_test, labels_test[t])
            
            elif 'celeb' in params['dataset']: 
                if 'film' in params['dataset']: 
                    for i, t in enumerate(tasks):
                        test_rep = model['rep'](test_images, i)
                        out_t_test = model[t](test_rep)
                        loss_t = loss_fn[t](out_t_test, labels_test[t])
                        tot_loss['all'] += scale[t]*loss_t.item()
                        tot_loss[t] += loss_t.item()
                        metric[t].update(out_t_test, labels_test[t])
                else: 
                    test_rep = model['rep'](test_images, i)
                    for i, t in enumerate(tasks):
                        out_t_test = model[t](test_rep)
                        loss_t = loss_fn[t](out_t_test, labels_test[t])
                        tot_loss['all'] += scale[t]*loss_t.item()
                        tot_loss[t] += loss_t.item()
                        metric[t].update(out_t_test, labels_test[t])
            num_test_batches+=1
        
        for t in tasks:
            testing_loss[t] = tot_loss[t]/num_test_batches
            metric_results = metric[t].get_result()
            for metric_key in metric_results:
                testing_metric[t] = metric_results[metric_key].item() 
                # writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter) 
            metric[t].reset()
        testing_loss['all'] = tot_loss['all']/len(test_dst)

        return testing_loss, testing_metric
