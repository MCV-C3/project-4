#from __future__ import print_function
import torch.nn as nn
import os
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms.v2  as F

class InMemoryDataset(Dataset):
    def __init__(self, source: Dataset, device, transform=None):
        self.images = [] 
        self.targets = []
        self.transform = transform
        self.device = device
        
        imgs = []
        targets = []

        for img, target in source:
            t = F.functional.pil_to_tensor(img)
            imgs.append(t)
            targets.append(target)

        self.images = torch.stack(imgs).to(device)
        self.targets = torch.tensor(targets).to(device)
        self.classes = source.classes
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx]), self.targets[idx]
        return self.images[idx], self.targets[idx]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print(im.size)
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')


def get_patches(img_tensor, patch_size, stride):
    """
    Extracts patches using Unfold.
    Input: (B, C, H, W) -> e.g. (256, 3, 224, 224)
    Output: (B * Num_Patches, C, patch_size, patch_size) -> e.g. (4096, 3, 64, 64)
    """
    # Unfold extracts sliding local blocks from a batched input tensor
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
    
    # Output shape: (Batch, C * PatchH * PatchW, Num_Patches)
    unfolded = unfold(img_tensor)
    
    # Reshape to be (Total_Patches, Channels, PatchH, PatchW)
    # patches = unfolded.transpose(1, 2).reshape(-1, 3, patch_size, patch_size)
    patches = unfolded.permute(0, 2, 1).contiguous().view(-1, 3, patch_size, patch_size)

    return patches


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss