
def cnn_factory(n_in, n_out, kernel_size):
    
    return nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    
    

class SSN(nn.Module):
    def __init__(self):
        super(SSN, self).__init__()

        
        # Create 4 conv layers with 16, 32, 64, 16 inputs
        # Create a Linear layer that will map this into a 192x192 vector
        # Assume input shape to be (1, 1, 192, 192)
        
        self.layer1 = cnn_factory(1, 16, 3)

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
        
        
        self.linear = nn.Linear(1 * 12 * 12, 192 * 192)

        
    def forward(self, x):
        after_conv =self.layer4(self.layer3(self.layer2(self.layer1(x))))
        flatten = after_conv.view(after_conv.shape[0], -1)

        lin = self.linear(flatten)
        
    
        return lin.view(x.shape[0],1,192,192)

class AlaUnet(nn.Module):
    def __init__(self):
        super(AlaUnet, self).__init__()

        
        # Create 4 conv layers with 16, 32, 64, 16 inputs
        # Create a Linear layer that will map this into a 192x192 vector
        # Assume input shape to be (1, 1, 192, 192)
        
        
        self.layer0 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, padding=1))
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.layer1_up_stream = nn.Sequential(nn.Conv2d(48, 16, kernel_size=1), nn.ReLU())  
        self.layer2_up_stream = nn.Sequential(nn.Conv2d(24, 20, kernel_size=1), nn.ReLU())
        self.layer3_up_stream = nn.Sequential(nn.Conv2d(24, 16, kernel_size=1), nn.ReLU())
        self.layer4_up_stream = nn.Sequential(nn.Conv2d(17, 1, kernel_size=1), nn.ReLU())

    def forward(self, x):
        down0 = self.layer0(x)
        down1 = self.layer1(down0)
        down2 = self.layer2(down1)
        down3 = self.layer3(down2)
        down4 = self.layer4(down3)

        up1 = self.upsample(down4)
        
        first_up = torch.cat([up1, down3], dim=1)
        up2 = self.upsample(self.layer1_up_stream(first_up))
        
        second_up = torch.cat([up2, down2], dim=1)
        up3 = self.upsample(self.layer2_up_stream(second_up))
        
        third_up = torch.cat([up3, down1],dim=1)
        up4 = self.upsample(self.layer3_up_stream(third_up))

        x = self.layer4_up_stream(torch.cat([x, up4], dim=1))
        return x        
