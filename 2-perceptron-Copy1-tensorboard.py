# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python [default]
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import seaborn
import torch
from IPython.display import HTML
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.tensorboard import writer

from utils.vis import plot_decision_space, visualize_data


RANDOM_STATE = 45

np.random.seed(RANDOM_STATE)

# %matplotlib inline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(7)

# DONT WORRY ABOUT THIS CODE VISUALIZATION ONLY
# -


w = writer.SummaryWriter()

# # Prepare and Visualize data

# +
N_FEATURES = 2

X, Y = make_blobs(centers=2, random_state=RANDOM_STATE, n_features=N_FEATURES)
X, Y = make_moons(n_samples=1000)
X = X/np.abs(X).max()
visualize_data(X,Y)

# -

data = pd.DataFrame(X, columns=['x1','x2'])
data['target'] = Y
data

pd.DataFrame(Y).head()

# # Define the perceptron
#
#
# ![alt text](https://cdn-images-1.medium.com/max/1600/1*-JtN9TWuoZMz7z9QKbT85A.png "Title")
#

# +

input_tensor = torch.randn((1,2))

linear_layer = nn.Linear(2, 1, bias=False)

linear_layer(input_tensor)

print(linear_layer.weight)
print(input_tensor)
output = linear_layer(input_tensor)
print(output)

print(input_tensor.shape)
manual_output = input_tensor.mm(linear_layer.weight.transpose(0, 1))
print(manual_output)
assert output == manual_output


# +

class Perceptron(nn.Module):
    def __init__(self, n_in):

        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=True)
        self.fc2 = nn.Linear(10, 10, bias=True)
        self.fc3 = nn.Linear(10, 10, bias=True)
        self.fc4 = nn.Linear(10, 10, bias=True)
        self.fc5 = nn.Linear(10, 10, bias=True)
        self.fc6 = nn.Linear(10, 1, bias=True)
        
    
    def forward(self, x):
        
        activ = F.tanh
        
    
        return self.fc6(activ(self.fc5(activ(self.fc4(activ(self.fc3(activ(self.fc2(activ(self.fc1(x)))))))))))

def custom_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)
#         m.weight.fill_(0)
        m.bias.data.fill_(0)
        
perceptron = Perceptron(2)

# Linear layers get their weights initialized by default, but you can reinitialize them if needed.
# perceptron.apply(custom_weights)

# print("fc weight", perceptron.fc.weight)
# print("bias weight", perceptron.fc.bias)
# -

[p for p in dir(perceptron) if not p.startswith('_')]

# # Train Loop Using CrossEntropy
#
#
#
# While using crossentropy loss our model needs to return (BS, n_classes) output tensor, the target has to be a coresponding dense label vector of shape (BS, 1)

# list(net.pa)
net = Perceptron(2)
list(net.parameters())
optimizer = SGD(net.parameters(), lr=0.01)
optimizer.__dict__

# +
N_EPOCHS = 50

# init the model, loss and optimizer
net = Perceptron(2)

optimizer = SGD(net.parameters(), lr=0.03)
criterion = nn.BCEWithLogitsLoss()
net = net.to(DEVICE)
loss_history = []

w = writer.SummaryWriter()


xs = torch.tensor(X[0]).unsqueeze(0).float().cuda()
w.add_graph(net, xs)
# w.add_image_with_boxes()

# neural_network.fit(X, y)
for epoch in range(N_EPOCHS):
    print("training epoch {}".format(epoch))
    
    
    
    for name, param in net.named_parameters():
        w.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    
    for xsample,ysample in zip(X, Y):
        optimizer.zero_grad() 
        # zero the gradients
        
        # batch preparation

        x = torch.Tensor(xsample).unsqueeze(0) # tensor([[0.8745, 0.5205]]) torch.Size([1, 2])
        y = torch.Tensor([ysample]).unsqueeze(0) # tensor([[1.]]) torch.Size([1, 1])
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
    
        
#         if True:
#             ix = np.random.randint(0, size=(4,), high=len(X))
#             x = torch.tensor((X[ix, :])).float()
#             y = torch.tensor(torch.Tensor(Y[ix]).reshape(-1,1)).float()
#             x = x.to(DEVICE)
#             y = y.to(DEVICE)
        
        # forward pass
        out = net(x)

        # loss calculation
        loss = criterion(out, y)

        # calculating gradients
        loss.backward()
        

        # changing the weights by specified(in the optimizer init) learning rate
        optimizer.step()
        
        #append calculated loss to the history
        loss_history.append(loss.detach().cpu().numpy())
        
    for name, param in net.named_parameters():
        w.add_histogram(name + 'grad', param.grad.clone().cpu().data.numpy(), epoch)
    
    w.add_scalar('train/loss', loss.detach().data, epoch)
#     calculate_acc(net, X, Y)
        
#     plot_decision_boundry(net, X)
# at the end plot final solution in red
# plot_decision_boundry(net, X, 'r-')
seaborn.scatterplot(x='x1',y='x2', hue='target', data=data)
plt.xlim((-1, 1))
plt.ylim((-1, 1))

f = plot_decision_space(net, X, Y)

w.add_figure('ddlkd', f)
w.close()

        
print("input shape (BS, n_classes):", x.shape)
print("target shape (BS, 1):", y.shape)
print("output shape", out.shape)
f
# def score_model(net, X, Y)
# -

plt.plot(loss_history)


def calculate_acc(net, X, Y):
    
    net.eval()
    

    Xtorch = torch.from_numpy(X).float().cuda()
    ytorch = torch.from_numpy(Y).float().cuda()

    with torch.no_grad():
        predictions = net(Xtorch)
    predictions = (predictions.sigmoid() > .5).squeeze(-1)
    predictions = np.array(predictions.cpu())
    
    real = np.array(ytorch.cpu())
    acc = sum(predictions==real)/len(predictions)
    print("model acc is: ", acc)
    
    net.train()

# +
history = pd.DataFrame(loss_history, columns=['loss'])
history.loss[:].rolling(10).mean().plot()
plt.title("loss")
plt.xlabel("batch number")
plt.ylabel("loss (CE)")

# history[::10].plot()

# -

# ## Exercises
# 1) Play with the training loop, enojoy the fact that you can inspect all the values dynamically. Consider using pdb.set_trace() for instance  
# 2) Can you edit the Perceptron class to create a Multi Layer Perceptron? (ie having more then 0 hidden layers)  
# 3) Initialize the the initial weights to 0. What do you think will happen? Can we still train the perceptron?  
# 4) What kind of gradient descnet are we using here? Stochastic? Batch? or Vanilla?  
# 5) What does detach do and why do we have to call it? (use google)  
# 6) Try adding a RELU activation after Linear unit - Will it train?
# 7) Try to implement a progress-bar (it might come in handy in our future exercises to)  
# 8) We can see the loss never going to 0, but the accuracy probably is reaching 100% - calculate ACC each epoch
# 9) Implement 

HTML('<iframe src=https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=&seed=0.19214&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false width=1000 height=600></iframe>')

f = plt.gcf()

# %matplotlib inline
f.show()

list(net.named_parameters())[0][1].grad


