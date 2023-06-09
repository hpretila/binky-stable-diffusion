# Data Loader for Unsplash Lite Dataset

import csv
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from util.hf_model_helper import HFDiffuserModelHelper  

class PersimmonDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_dim=512):    
        self.image_paths = []
        self.image_captions = []

        self.im_dimension = img_dim

        # Get max length
        self.text_max = 512

        # Open the CSV file and read the image path from it
        with open(root_dir + '/manifest.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                image_path = root_dir + '/' + row[0]
                image_caption = row[1]
                self.image_paths.append(image_path)
                self.image_captions.append(image_caption)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        if (not os.path.exists(path)):
            return None, None
        else:
            img = Image.open(path)
            img = img.resize((self.im_dimension, self.im_dimension))

            # Convert to tensor
            img = HFDiffuserModelHelper.preprocess_image(img)

        q = self.image_captions[idx]
        return img, q