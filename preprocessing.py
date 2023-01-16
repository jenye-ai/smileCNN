import torch
from PIL import Image


class ImagePreprocessing (torch.utils.data.Dataset):
  def __init__(self, file_paths, labels, transform=None):
    self.file_paths = file_paths
    self.labels = labels
    self.transform = transform
  
  def __len__(self):
    return len(self.file_paths)
  
  def __getitem__(self, idx):
    # Load the image and label
    image = Image.open(self.file_paths[idx])
    label = self.labels[idx]
    
    # Apply the transform (if any)
    if self.transform:
      image = self.transform(image)
    
    return image, label