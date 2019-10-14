
### 1.
F.max_pool2d(torch.rand((1, 64, 64, 32)), kernel_size=2)

### 2. 
tensor = torch.randn((32, 10))
tensor.argmax(dim=1)


### 3.
# Change X shape into (10, 3)
X = torch.ones(5, 6)
Y = X.view(10, 3)

# Remove all the dimensions of size 1 in X.

X = torch.randn(10, 10, 1, 1)
Y = X.squeeze()

### 4.
# stack x, y, and z vertically.
x = torch.Tensor([1, 4])
y = torch.Tensor([2, 5])
z = torch.Tensor([3, 6])
O = torch.stack([x, y, z])
# print(O)

### 5.  Counting 
# Get the indices of all nonzero elements in X.
X = torch.Tensor([[0,1,7,0,0],[3,0,0,2,19]])
X.nonzero()

### 5
# Why is the output of this cell a tensor with a shape 1,64,6,6 - what can we do to get the same input and output shape
v =torch.rand(1, 32, 8, 8)
print(nn.Conv2d(32, 64, 3)(v).shape)

### 6
# Given
y_true = np.array([0,1,2,3,4,5]).astype(float)
y_true = torch.tensor(y_true, requires_grad=True)
y_pred = y_true * 0.8
loss = F.mse_loss(y_pred, y_true)# calculate the mse loss for those arrays and the corresponding gradients
