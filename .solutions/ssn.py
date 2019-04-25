
class SSN(nn.Module):
    def __init__(self):
        super(SSN, self).__init__()

        
        # Create 4 conv layers with 16, 32, 64, 16 inputs
        # Create a Linear layer that will map this into a 192x192 vector
        # Assume input shape to be (1, 1, 192, 192)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        
        self.linear = nn.Linear(1 * 12 * 12, 36864)

        
    def forward(self, x):
        after_conv =self.layer4(self.layer3(self.layer2(self.layer1(x))))
        flatten = after_conv.view(after_conv.shape[0], -1)

        lin = self.linear(flatten)
        
    
        return lin.view(x.shape[0],1,192,192)
        