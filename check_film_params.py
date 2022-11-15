import os
import click
import json
import torch
import numpy as np
# from train_multi_task import train_multi_task

@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')




def check_parameters(param_file):
    with open(param_file) as json_params:
        params = json.load(json_params)

    hyper_parameters = params['hyper_parameters']
    tuned_params = {}
    num_trials = 1
    fixed_params = {}
    for (key, val) in params.items():
        if key == 'hyper_parameters':
            continue
        elif key in hyper_parameters:
            tuned_params[key] = val
            num_trials *= len(val)    
        else:
            fixed_params[key] = val

    film_params = {}
    for trial_idx in range(num_trials):
        trial_params = fixed_params.copy()
        # Setup the params for a new trial
        for (key, val) in tuned_params.items():
            trial_params[key] = val[trial_idx % len(val)]
        trial_identifier = []
        for (key, val) in trial_params.items():
            if 'tasks' in key:
                continue
            if 'scales' in key:
                for task, scale in val.items():
                    trial_identifier += ['scale_{}={}'.format(task, scale)]
            else:
                trial_identifier += ['{}={}'.format(key,val)]
        trial_identifier = '|'.join(trial_identifier)

        tasks = fixed_params['tasks']
        state = torch.load("./saved_models/{}_model.pkl".format(trial_identifier))


        model_rep = state['model_rep']
        for layer, params in model_rep.items():
            if ('module.film' in layer):
                film_params[layer] = params
            
        print("film1 task1 gammas:", film_params['module.film3_task1.gammas'])
        print("film1 task2 gammas:", film_params['module.film3_task2.gammas'])
        print("film1 task1 betas:", film_params['module.film3_task1.betas'])
        print("film1 task2 betas:", film_params['module.film3_task2.betas'])


if __name__ == '__main__':
    check_parameters()