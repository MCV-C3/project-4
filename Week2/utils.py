#from __future__ import print_function
import torch.nn as nn
import os
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image

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