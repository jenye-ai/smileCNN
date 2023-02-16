from tqdm import tqdm
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from dataset import Genki4kDataset
from model import SmileCNN

#checking model seperability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.load("C:/Code/CNN_models/smileCNN_iter1.pt", map_location=device)

dataset = Genki4kDataset('C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt','C:/Code/461_data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt', 'C:/Code/461_data/GENKI-R2009a/Subsets/preprocessed')
train_data_loader = DataLoader(dataset, batch_size=1)