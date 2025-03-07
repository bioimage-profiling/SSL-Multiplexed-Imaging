from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import h5py

class hdf5_multichannel_singleh5():
    def __init__(self, file_path):
        h5_file = h5py.File(file_path, 'r')
        self.images = h5_file['Images']
        self.image_names = h5_file['Names']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,i):
        image = self.images[i] # [7, 256, 256]
        # image = np.transpose(image, (2, 0, 1)) # [256, 256, 7]
        name = self.image_names[i]
        return image, name
    
class h5_chunk_wrapper(Dataset):
    def __init__(self, h5_dir, transform=None):
        # get files
        files = sorted(list(h5_dir.glob('*.h5')))
        # open h5 files
        self.single_h5s = [hdf5_multichannel_singleh5(pfile) for pfile in files]
        # register
        self.register = {}
        idx = 0
        for h5id, h5class in enumerate(self.single_h5s):
            for imgid in range(h5class.__len__()):
                self.register[idx] = [h5id, imgid]
                idx += 1
              
        # define transform
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([])
            
    def __len__(self):
        return len(self.register)
    
    def __getitem__(self, i):
        h5id, imgid = self.register[i]
        image, name = self.single_h5s[h5id].__getitem__(imgid)
        image = torch.from_numpy(image)
        image = self.transform(image)
        return image, name[0].decode('utf-8')
        # return image, name.decode('utf-8')
    
prostate_log1p_meanstds = {
                                'round_1_DAPI'  : [7.5224,0.9350],
                                'round_1_AF488' : [6.8404,0.7217],
                                'round_1_AF555' : [6.9394,0.3670],
                                'round_1_AF647' : [5.8485,0.4203],
                                'round_1_AF750' : [6.4374,0.5327],
                                'round_2_AF647' : [7.7221,0.8249],
                                'round_2_AF750' : [7.1350,0.9929]
                                }

HBPTMA_log1p_meanstds = {  
                                'round_1_DAPI' : [7.4133,0.8771],
                                'round_1_AF488': [5.8039,0.6103],
                                'round_1_AF555': [6.0221,1.0753],
                                'round_1_AF647': [5.2786,0.4496],
                                'round_1_AF750': [6.3647,0.9954],
                                'round_2_AF647': [7.2614,0.9578],
                                }

BOMI1_log1p_meanstds = {  
                                'round_1_DAPI' : [7.5625,0.9301],
                                'round_1_AF488': [6.5928,0.9784],
                                'round_1_AF555': [5.9220,1.0435],
                                'round_1_AF647': [5.0343,0.2865],
                                'round_1_AF750': [6.0670,0.7288],
                                'round_2_AF647': [6.5871,1.1323]
                                }

def get_mean_std(args):
    if args.dataset == 'prostate':
        channel_names = ['round_1_DAPI', 'round_1_AF488', 'round_1_AF555',
                        'round_1_AF647', 'round_1_AF750', 'round_2_AF647', 'round_2_AF750'] # Should start with DAPI
    else:
        channel_names = ['round_1_DAPI', 'round_1_AF488', 'round_1_AF555',
                        'round_1_AF647', 'round_1_AF750', 'round_2_AF647',] # Should start with DAPI
    n_channels = len(channel_names)
    
    if args.dataset == 'BOMI1':
        ds_mean = np.array(list(BOMI1_log1p_meanstds.values()))[:,0]
        ds_std  = np.array(list(BOMI1_log1p_meanstds.values()))[:,1]
    elif args.dataset == 'HBPTMA':
        ds_mean = np.array(list(HBPTMA_log1p_meanstds.values()))[:,0]
        ds_std  = np.array(list(HBPTMA_log1p_meanstds.values()))[:,1]
    elif args.dataset == 'prostate':
        ds_mean = np.array(list(prostate_log1p_meanstds.values()))[:,0]
        ds_std  = np.array(list(prostate_log1p_meanstds.values()))[:,1]
    else:
        exit('Unknown dataset name')
    mean_std = {'mean':ds_mean, 'std' :ds_std}
    return mean_std, n_channels