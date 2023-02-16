import torch.nn as nn
# Define the CNN architecture
# class SmileCNN(nn.Module):
#     def __init__(self):
#         super(SmileCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 9)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(16, 8, 5)
#         self.conv3 = nn.Conv2d(8, 16, 5)
#         self.linear = nn.Linear(256,1)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.relu(x))
#         x = self.conv2(x)
#         x = self.pool2(self.relu(x))
#         x = self.conv3(x)
#         x = self.dropout(x)
#         x = self.pool3(self.relu(x))
#         #x = x.view(-1, 128 * 4 * 4)
#         x = self.fc(x)
#         #x = self.out(x)
#         return x


class SmileCNN(nn.Module):
  def __init__(self):
    super(SmileCNN, self).__init__()
    # Define the convolutional layers
    self.conv1 = nn.Conv2d(1, 16, kernel_size=9, padding=1)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=1)
    self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=1)

    # Define the max pooling layers
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    # Define the fully connected layer
    self.fc = nn.Linear(400, 400)
    self.out = nn.Linear(400, 2)

    self.relu = nn.ReLU()
    #self.flatten - nn.Flatten()
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool1(self.relu(x))
    x = self.conv2(x)
    x = self.pool2(self.relu(x))
    x = self.conv3(x)
    x = self.dropout(x)
    x = self.pool3(self.relu(x))
    x = x.view(-1, 400)
    x = self.fc(x)
    x = self.out(x)
    return x

  
class SmileCNNSVM(nn.Module):
  def __init__(self):
    super(SmileCNNSVM, self).__init__()
    # Define the convolutional layers
    self.conv1 = nn.Conv2d(1, 16, kernel_size=9)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=5)
    self.conv3 = nn.Conv2d(8, 16, kernel_size=5)

    # Define the max pooling layers
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    # Define the fully connected layer
    #self.fc = nn.Linear(400, 400)
    self.out = nn.Linear(256, 2)
    self.flatten = nn.Flatten()

    self.relu = nn.ReLU()
    #self.flatten - nn.Flatten()
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool1(self.relu(x))
    x = self.conv2(x)
    x = self.pool2(self.relu(x))
    x = self.conv3(x)
    x = self.dropout(x)
    x = self.pool3(self.relu(x))    

    x = self.flatten(x)
    w = x
    x = self.out(x)
    return x, w