import numpy as np
import matplotlib.pyplot as plt
import os

# if a dominate b
def pareto_dominate(a, b):
    a_has_dominated = False
    for loss_a, loss_b in zip(a, b):
        if (loss_a > loss_b):
            return False
        if (loss_a < loss_b):
            a_has_dominated = True
    return a_has_dominated

# load npy files

all_results = {}
pareto_frontier = {}
pareto_dominated = {}

# datasets = ['mnist', 'mnist_film_1head', 'mnist_film_2head']
datasets = ['mnist', 'mnist_film_2head', 'mnist_film1', 'mnist_film2']
for dataset in datasets:
    for file in os.listdir("./results/%s"%(dataset)): 
        print(file)
        results = np.load("./results/%s/%s"%(dataset, file), allow_pickle=True)
        filename, extension = os.path.splitext(file)
        if dataset in all_results: 
            all_results[dataset] += [[pt['L'], pt['R']] for pt in results]
        else:
            all_results[dataset] = [[pt['L'], pt['R']] for pt in results]
            pareto_frontier[dataset] = []
            pareto_dominated[dataset] = []

# print(all_results)
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

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey']
for i, dataset in enumerate(datasets):
    frontier = np.array(pareto_frontier[dataset])
    other = np.array(pareto_dominated[dataset])
    sort_indices = np.argsort(frontier[:, 0])
    frontier = frontier[sort_indices]
    # plt.scatter(other[:, 0], other[:, 1], color=colors[i], marker='.')
    plt.plot(frontier[:, 0], frontier[:, 1], color=colors[i], marker='.', label=dataset)

plt.ylim([0.1, 0.3])
plt.xlim([0.05, 0.3])
plt.ylabel('Task R Loss')
plt.xlabel('Task L Loss')
plt.legend()
plt.show()
    