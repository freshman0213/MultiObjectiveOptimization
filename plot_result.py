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
for file in os.listdir("../"): 
    filename, extension = os.path.splitext(file)
    if extension == '.npy':
        result_file_list.append("../" + file)

# load npy files
pareto_frontier = []
pareto_dominated = []

all_results = []
for file in result_file_list: 
    results = np.load(file, allow_pickle=True)
    all_results += [[pt['L'], pt['R']] for pt in results]

pareto_frontier = []
for i, target in enumerate(all_results): 
    non_dominated = True
    for j, temp in enumerate(all_results): 
        if i != j and pareto_dominate(temp, target): 
            non_dominated = False
    if non_dominated: 
        pareto_frontier.append(target)
    else: 
        pareto_dominated.append(target)

pareto_frontier = np.array(pareto_frontier)
pareto_dominated = np.array(pareto_dominated)


sort_indices = np.argsort(pareto_frontier[:, 0])
pareto_frontier = pareto_frontier[sort_indices]

plt.scatter(pareto_dominated[:, 0], pareto_dominated[:, 1], label='pareto_dominated', color='r', marker='.')
plt.plot(pareto_frontier[:, 0], pareto_frontier[:, 1], label='pareto_frontier', color='r', marker='.')
plt.legend()
plt.show()
    