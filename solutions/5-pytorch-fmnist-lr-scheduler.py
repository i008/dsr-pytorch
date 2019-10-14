


dftrain = pd.read_csv(PATH_TO_FMNIST_TRAIN).sample(frac=1)
fmnist_train = FashionMnist(dftrain, transform=transform_train)
data_loader = DataLoader(fmnist_train, batch_size=32)


INITIAL_LR = 10e-7
losses = []
lrs = []
cnn = SimpleCNN(10).cuda()
optimizer = getattr(torch.optim, OPTIMIZER)(cnn.parameters(), lr=0.001)

for param_group in optimizer.param_groups:
    param_group['lr'] = INITIAL_LR
    

for i, batch in enumerate(data_loader):
    if i % 100 == 0: 
        print(i)
        print(loss)
        
    optimizer.zero_grad()
    X, y = batch
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    y_pred = cnn(X)

    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if loss > 10:
        break

    current_lr = lr = optimizer.param_groups[0]['lr']
    losses.append(loss)
    lrs.append(current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr * 1.03
        
    if i == 1000:
        break


    
df = pd.DataFrame([l.detach().cpu().numpy() for l in losses])
df['lrs'] = lrs
ax = plt.plot(df['lrs'], df[0])
plt.xscale('log')
