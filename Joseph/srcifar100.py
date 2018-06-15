import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class SRCIFAR100(Dataset):

    def __init__(self, folder_path, transform):
        """
        Args:
            csv_file (string): Path to the image file with included label file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder_path = folder_path
        self.labels = np.load(self.folder_path + '/labels.npy')
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path,str(idx)+'.npy')
        img_arr = np.uint8(np.load(img_name)*255)
        img_arr = np.rollaxis(img_arr,0,3)
        image = Image.fromarray(img_arr)

        if self.transform:
            image = self.transform(image)
            
        target = torch.LongTensor(np.array(self.labels[idx]))

        return image, target