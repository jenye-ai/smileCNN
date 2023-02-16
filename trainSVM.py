from tqdm import tqdm
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from dataset import Genki4kDataset
from model import SmileCNN
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

losses = []
val_losses = []
test_losses = [] 

epoch_train_losses = []
epoch_test_losses = []

n_epochs = 100
early_stopping_tolerance = 10
early_stopping_threshold = 0.0
BATCH_SIZE = 50
lrate = 0.01
momentum = 0.9
weight_decay = 0.0005
NUM_CLASSES = 2

torch.set_grad_enabled(True)

def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)

    #enter train mode
    model.train()

    #compute loss
    loss = loss_fn(yhat, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step

def convert_one_hot(num_classes, data):
    return torch.nn.functional.one_hot(data, num_classes).squeeze(dim=1)

#Load in the Dataset
dataset = Genki4kDataset('C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', 'C:/Code/461_data/GENKI-R2009a/Subsets/preprocessed')
#dataset = Genki4kDataset('/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files')
#dataset = Genki4kDataset('/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels_dummy.txt','/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images_dummy.txt', '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files')
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define the DataLoader for the train and validation dataset
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
#loss
loss_fn = torch.nn.CrossEntropyLoss() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmileCNN()
model.to(device)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

optimizer = torch.optim.SGD(model.parameters(), lr=lrate, momentum=momentum, weight_decay = weight_decay)

#train step
train_step = make_train_step(model, optimizer, loss_fn)

for epoch in range(n_epochs):
  epoch_loss = 0
  print(optimizer.param_groups[-1]['lr'])
  for i ,data in tqdm(enumerate(train_data_loader), total = len(train_data_loader)): #iterate ove batches
    x_batch , y_batch = data
    x_batch = x_batch.to(device) #move to gpu
    y_batch = y_batch.to(device) #move to gpu
    

    loss = train_step(x_batch, y_batch)
    epoch_loss += loss/len(train_data_loader)
    losses.append(loss)
    
  epoch_train_losses.append(epoch_loss)
  print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

  #validation doesnt requires gradient
  with torch.no_grad():
    cum_loss = 0
    for x_batch, y_batch in val_data_loader:
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      #model to eval mode
      model.eval()

      yhat = model(x_batch)
      val_loss = loss_fn(yhat,y_batch)
      cum_loss += loss/len(val_data_loader)
      val_losses.append(val_loss.item())


    epoch_test_losses.append(cum_loss)
    print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  
    
    best_loss = min(epoch_test_losses)
    
    #save best model
    if cum_loss <= best_loss:
      best_model_wts = model.state_dict()
    
    #early stopping
    early_stopping_counter = 0
    if cum_loss > best_loss:
      early_stopping_counter +=1

    if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
      print("/nTerminating: early stopping")
      break #terminate training
    
#load best model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "C:/Code/CNN_models/smileCNN_iter1.pt")

print("Testing Start!")
# test the model
model.eval()
total_loss = 0
total_correct = 0
total_samples = 0

with torch.no_grad():
    
    for x_batch, y_batch in test_data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        
        yhat = model(x_batch)
        test_loss = loss_fn(yhat,y_batch)
        _, yhat = torch.max(yhat, dim=1)
        total_loss += loss.item() * x_batch.size(0)
        total_correct += torch.sum(yhat == y_batch)
        total_samples += x_batch.size(0)
            
    test_loss = total_loss / total_samples
    test_acc = total_correct.double() / total_samples
print(f'test_loss: {test_loss} test_acc: {test_acc}')