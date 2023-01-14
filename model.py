import torch.nn as nn

# Define the CNN architecture
class SmileCNN(nn.Module):
    def __init__(self):
        super(SmileCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 9)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.conv3 = nn.Conv2d(8, 16, 5)
        self.linear = nn.Linear(256,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = self.softmax(x)
        return x
