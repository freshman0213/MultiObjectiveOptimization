import json
import click
from multi_task import train_multi_task

@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')
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

    for trial_idx in range(num_trials):
        # Setup the params for a new trial
        for (key, val) in tuned_params:
            fixed_params[key] = val[trial_idx % len(val)]
        
        # Trian the multi-task model with specific params
        train_multi_task(fixed_params)

    # output the best model (Q: how to define best since we are dealing with multiple tasks at the same time?)

if __name__ == '__main__':
    train_multi_task_scalarization()