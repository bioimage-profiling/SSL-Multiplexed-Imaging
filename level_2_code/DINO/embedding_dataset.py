from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class embedding_dataset(Dataset):
    def __init__(self, args, transform=None):
        path_embeddings = args.path_embeddings
        path_names = args.path_names
        self.ceil_size = 256
        
        embeddings = np.load(path_embeddings)
        names = np.load(path_names)
        
        self.ds_mean, self.ds_std = np.mean(embeddings, axis=0), np.std(embeddings, axis=0)
        args.n_channels = embeddings.shape[-1]
        
        self.roi_embeddings, self.roi_names = self.get_grouped_roi(embeddings, names)
        
        # define transform
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([])
    
    def __len__(self):
        return len(self.roi_names)
    
    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.roi_embeddings[idx]).float()
        embedding = self.transform(embedding)
        name = self.roi_names[idx]
        return embedding, name
    
    def get_names_dataframe(self, names):
        
        df = pd.DataFrame({'name':names})
        df['name'] = df['name'].astype('string')
    
        df['slide'] = [nm.split('_')[0] for nm in df['name']]
        df['slide'] = df['slide'].astype('string')
        df['roi'] = [nm.split('_')[1] for nm in df['name']]
        df['roi'] = df['roi'].astype('string')
        
        return df
    
    def repeat_to_size(self, original_array, ceil_size):
        # Calculate the number of times to repeat along the first axis
        num_repetitions = int(np.ceil(ceil_size / original_array.shape[0]))

        # Repeat the array along the first dimension to make it (256, 768)
        repeated_array = np.tile(original_array, (num_repetitions, 1))
        repeated_array = repeated_array[:ceil_size]  # Trim to exactly 256
        
        return repeated_array

    def get_grouped_roi(self, embeddings, names):
        
        df = self.get_names_dataframe(names)
        
        roi_names = []
        roi_embeddings = []
        for grpn, grpdf in df.groupby(by=['slide', 'roi']):
            roi_names.append('_'.join(grpn))
            emb = self.repeat_to_size(embeddings[grpdf.index], self.ceil_size)
            emb = np.swapaxes(emb, 0, 1).reshape(-1, 16, 16)
            roi_embeddings.append(emb)
            
        return roi_embeddings, roi_names
    
    def get_mean_std(self):
        return {'mean':self.ds_mean, 'std' :self.ds_std}