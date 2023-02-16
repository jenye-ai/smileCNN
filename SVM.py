import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from dataset import Genki4kDataset
from model import SmileCNNSVM

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd  

torch.manual_seed(42)

#checking model seperability
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmileCNNSVM()
model.load_state_dict(torch.load("C:/Code/CNN_models/smileCNN_iter2.pt"))
model.eval()

dataset = Genki4kDataset('C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', 'C:/Code/461_data/GENKI-R2009a/Subsets/preprocessed')
#dataset = Genki4kDataset('/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files')
#dataset = Genki4kDataset('/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels_dummy.txt','/Users/jen/Documents/Code/Datasets/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images_dummy.txt', '/Users/jen/Documents/Code/Datasets/GENKI-R2009a/files')
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define the DataLoader for the train and validation dataset
train_data_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

print("SVM Training Start!")
with torch.no_grad():
    
    for x_batch, y_batch in train_data_loader:
        #x_batch = x_batch.to(device)
        #y_batch = y_batch.to(device)

        
        yhat, svm_features = model(x_batch)
           

clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(gamma='auto')))
clf.fit(svm_features, y_batch)
print("SVM Testing Start!")
# test the model
total_loss = 0
total_correct = 0
total_samples = 0
loss_fn = torch.nn.CrossEntropyLoss() 
with torch.no_grad():
    
    for x_batch, y_batch in test_data_loader:
        #x_batch = x_batch.to(device)
        #y_batch = y_batch.to(device)

        
        yhat, svm_features = model(x_batch)
       
print("")
print(f"Accuracy: {clf.score(svm_features, y_batch)}")

from sklearn.metrics import classification_report

target_names = ['No Smile', 'Smile']
y_pred = clf.predict(svm_features)
print(classification_report(y_batch, y_pred, target_names=target_names))
print(clf)

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(svm_features) 

df = pd.DataFrame()
df["y"] = y_pred
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="SmileCNN T-SNE projection") 
plt.show()
# p = np.array(clf.steps[1][1].decision_function(svm_features)) # decision is a voting function
# prob = np.exp(p)/np.sum(np.exp(p),axis=0, keepdims=True) # softmax after the voting
# idx0 =  np.where(y_pred == 0)[0]
# idx1 =  np.where(y_pred == 1)[0]
# x = svm_features[:, [255]]

# prob0 = [prob[i] for i in idx0]
# x0 = [x[i] for i in idx0]

# prob1 = [prob[i] for i in idx1]
# x1 = [x[i] for i in idx1]

# plt.scatter(np.log(x0),prob0)
# plt.scatter(np.log(x1),prob1)
# plt.show()

#get probabilities of all rows
#get indices of all the 0's, get indices of all the 1's
#plot it for the most influential feature
#graph both on one graph
