import os
import click
import json
import torch
import numpy as np
from train_multi_task import train_multi_task

@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')

RESULT_FOLDER = './results/mnist/'
def train_multi_task_scalarization(param_file):
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

    pareto_frontier = []
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
        if not os.path.exists("./saved_models/{}_model.pkl".format(trial_identifier)):
            # Trian the multi-task model with specific params
            train_multi_task(trial_params)

        # Retrieve the best model's performance
        if not os.path.exists("./saved_models/{}_model.pkl".format(trial_identifier)):
            continue
        tasks = fixed_params['tasks']
        state = torch.load("./saved_models/{}_model.pkl".format(trial_identifier))
        testing_loss = {}
        for t in tasks:
            testing_loss[t] = state['testing_loss_{}'.format(t)]

        # Check whether it is on the Pareto Frontier
        non_dominated = True
        for x in pareto_frontier:
            if pareto_dominate(x, testing_loss):
                non_dominated = False 
                break
        if not non_dominated:
            continue
        else:
            for x in pareto_frontier:
                if pareto_dominate(testing_loss, x):
                    pareto_frontier.remove(x)
            pareto_frontier.append(testing_loss)

    # Save the pareto frontier
    exp_identifier = []
    for (key, val) in params.items():
        if 'tasks' in key or 'hyper_parameters' in key:
            continue
        if 'scales' in key:
            for task, scale in val.items():
                exp_identifier += ['scale_{}={}'.format(task, scale)]
        else:
            exp_identifier += ['{}={}'.format(key,val)]
    exp_identifier = '|'.join(exp_identifier)
    exp_identifier = RESULT_FOLDER+exp_identifier
    np.save(exp_identifier, pareto_frontier)

def pareto_dominate(a, b):
    for k in a.keys():
        if (a[k] > b[k]):
            return False
    return True

if __name__ == '__main__':
    train_multi_task_scalarization()