import numpy as np
import torch
from matplotlib import pyplot as plt

import os
p = os.path.dirname(os.path.abspath(__file__))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

def plot_decision_boundry(net, X, line='g--'):
    W = net.fc.weight[0].detach().cpu().numpy()
    b = net.fc.bias.detach().cpu().numpy()
    f = lambda x: (-W[0]/W[1]) * x +  (-b/W[1])
    dziedz = np.arange(-1, 1, 0.01)
    plt.plot(dziedz, f(dziedz), line) 
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    
    
def plot_decision_space(net, X, Y):
    net.to(DEVICE)
    
    xx, yy = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = torch.sigmoid(net(torch.from_numpy(grid).float().to(DEVICE))).view(-1, 1).view(xx.shape)
    f, ax = plt.subplots(figsize=(8, 6))


    contour = ax.contourf(xx, yy, probs.detach().cpu().numpy(), 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:,0], X[:, 1], c=Y, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-1, 1), ylim=(-1, 1),
           xlabel="$X_1$", ylabel="$X_2$")
    
    return f
    
    
def visualize_data(X, Y):
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:,0], X[:, 1], c=Y, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlabel="$X_1$", ylabel="$X_2$")    
    

def idtoname(i):

    with open(os.path.join(p, 'imagenet_synsets.txt'), 'r') as f:
        synsets = f.readlines()


    with open(os.path.join(p, 'imagenet_classes.txt'), 'r') as f:
        class_id_to_key = f.readlines()


    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}


    class_id_to_key = [x.strip() for x in class_id_to_key]
    name = key_to_classname[class_id_to_key[i]]
    print(name)
    return name