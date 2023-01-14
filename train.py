import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Genki4kDataset
from model import SmileCNN

torch.manual_seed(42)
BATCH_SIZE = 4
EPOCHS = 10

# Initialize the network and move it to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SmileCNN()
net.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Load in the Dataset
#dataset = Genki4kDataset('C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', 'C:/Code/461_data/GENKI-R2009a/files')
dataset = Genki4kDataset('/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files')
#dataset = Genki4kDataset('/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels_dummy.txt','/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images_dummy.txt', '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files')
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define the DataLoader for the train and validation dataset
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_data_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

print("Training Start!")
# Train the network
for epoch in range(EPOCHS):
    running_loss = 0.0
    # train the model
    for i, data in enumerate(train_data_loader):
        inputs, labels = data

        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)

        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = torch.t(net(inputs))[0]
        loss = criterion(outputs, labels.type(torch.float64))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # validate the model
    val_loss = 0.0
    val_acc = 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in val_data_loader:
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.t(net(inputs))[0]
            loss = criterion(outputs, labels.type(torch.float64))
            val_loss += loss.item()
            preds = outputs
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
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.t(net(inputs))[0]
        loss = criterion(outputs, labels.type(torch.float64))
        test_loss += loss.item()
        preds = outputs
        test_acc += (preds == labels).sum().item()
print(f'test_loss: {test_loss / len(test_data_loader)} test_acc: {test_acc / len(test_dataset)}')
torch.save(net.state_dict(), "/Users/jen/Documents/Code/smileCNN_test1.pt")


