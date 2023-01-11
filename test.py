import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from mtcnn import MTCNN
import cv2

class Genki4kDataset(Dataset):
    def __init__(self, labels_path, image_name_path, images_path):
        self.labels = self._load_labels(labels_path)
        self.image_names = self._load_image_names(image_name_path)
        self.images_path = images_path
        self.detector = MTCNN()

    def _load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip()[0] for line in f]
        return labels

    def _load_image_names(self, image_name_path):
        with open(image_name_path, 'r') as f:
            image_names = [line.strip() for line in f]
        return image_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        label = self.labels[i]
        image_names= self.image_names[i]
        img_path = self.images_path + f'/{image_names}'
        image = Image.open(img_path)
        detector.detect_faces(image)
        image = transforms.ToTensor()(image)
        return image, label
    def create_bounding_box(image):
        faces = detector.detect_faces(image)
        bounding_box = faces[0]["box"][:4] # to obtain the only 1 image in our case
        keypoints = faces[0]["keypoints"]

  cv2.rectangle(image, (bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0,155,255), 2)
  return image, bounding_box

# Define the CNN architecture
class SmileCNN(nn.Module):
    def __init__(self):
        super(SmileCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network and move it to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SmileCNN()
net.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Load in the Dataset
dataset = Genki4kDataset('C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', 'C:/Code/461_data/GENKI-R2009a/files')

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define the DataLoader for the train and validation dataset
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)

print("Training Start!")
# Train the network
for epoch in range(2):
    running_loss = 0.0
    # train the model
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # validate the model
    val_loss = 0.0
    val_acc = 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in val_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    print(f'epoch {epoch + 1} train_loss: {running_loss / len(train_data_loader)} val_loss: {val_loss / len(val_data_loader)} val_acc: {val_acc / len(val_dataset)}')
    net.train()


print("Training Finished")

print("Testing Start!")
# test the model
test_loss = 0.0
test_acc = 0.0
net.eval()
with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        test_acc += (preds == labels).sum().item()
print(f'test_loss: {test_loss / len(test_data_loader)} test_acc: {test_acc / len(test_dataset)}')
# # Define a DataLoader for the dataset
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# print("Training Start")


# # Train the network
# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(data_loader):
#         # get the inputs
#         inputs, labels = data

#         # move data to device
#         inputs, labels = inputs.to(device), labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

