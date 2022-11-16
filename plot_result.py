import numpy as np
import matplotlib.pyplot as plt
import os


# if a dominate b
def pareto_dominate(a, b):
    a_has_min = False
    for loss_a, loss_b in zip(a, b):
        if (loss_a > loss_b):
            return False
        if (loss_a < loss_b):
            a_has_dominated = True
    return a_has_dominated


# find npy files in the folder
result_file_list = []
for file in os.listdir("./results/mnist_film_2head"): 
    filename, extension = os.path.splitext(file)
    if extension == '.npy':
        result_file_list.append("./results/mnist_film_2head/" + file)

# load npy files

all_results = {}
datasets = []
pareto_frontier = {}
pareto_dominated = {}
for file in result_file_list: 
    results = np.load(file, allow_pickle=True)
    file_split = file.split('|')
    for s in file_split: 
        if 'dataset=' in s: 
            dataset = s[9:]
    if dataset in all_results: 
        all_results[dataset] += [[pt['L'], pt['R']] for pt in results]
    else:
        datasets.append(dataset)
        all_results[dataset] = [[pt['L'], pt['R']] for pt in results]
        pareto_frontier[dataset] = []
        pareto_dominated[dataset] = []

print(all_results)
for dataset in datasets:
    for i, target in enumerate(all_results[dataset]): 
        non_dominated = True
        for j, temp in enumerate(all_results[dataset]): 
            if i != j and pareto_dominate(temp, target): 
                non_dominated = False
        if non_dominated: 
            pareto_frontier[dataset].append(target)
        else: 
            pareto_dominated[dataset].append(target)

colors = ['tab:blue', 'tab:orange']
for i, dataset in enumerate(datasets):
    frontier = np.array(pareto_frontier[dataset])
    other = np.array(pareto_dominated[dataset])
    sort_indices = np.argsort(frontier[:, 0])
    frontier = frontier[sort_indices]
    plt.scatter(other[:, 0], other[:, 1], color=colors[i], marker='.')
    plt.plot(frontier[:, 0], frontier[:, 1], color=colors[i], marker='.', label=dataset)

plt.legend()
plt.show()
    