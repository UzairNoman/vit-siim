import torch
from PIL import Image
from torchvision import transforms
#import torchvision
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import math
from sklearn.model_selection import train_test_split
class HAM(Dataset):
    """HAM dataset class"""

    def __init__(self, root, purpose, seed, split, transforms=None, tfm_on_patch=None):
        self.root_path = root
        self.purpose = purpose
        self.seed = seed
        self.split = split
        self.img_part1 = os.listdir(f'{root}/HAM10000_images_part_1/')
        self.img_part2 = os.listdir(f'{root}/HAM10000_images_part_2/')
        self.images, self.labels = self._make_dataset(directory=self.root_path, purpose=self.purpose, seed=self.seed, split=self.split)
        self.transforms = transforms
        self.tfm_on_patch = tfm_on_patch

    def _make_dataset(self,directory, purpose, seed, split):
        """
        Create the image dataset by preparing a list of samples
        :param directory: root directory of the dataset
        :returns: (images, labels) where:
            - images is a numpy array containing all images in the dataset
            - labels is a list containing one label per image
        """

        data_path = os.path.join(directory, "HAM10000_metadata.csv")
        meta_df = pd.read_csv(data_path)
        # meta_df.rename(columns={'image_id': 'image_name'})
        meta_df['target'] = pd.Categorical(meta_df['dx']).codes
        no_of_classes = meta_df['target'].unique()
        print(f'No. of Class in HAM: {no_of_classes}')
        meta_df['image_name'] = meta_df.apply(lambda row: self.extract_path_img(directory,row.image_id), axis=1)
        print(meta_df.head())

        
    
        #(33126, 8)
        
        train, val = train_test_split(meta_df, test_size=split, random_state=seed)
        #do we want to apply stratification here?
        # train, val, test = np.split(meta_df.sample(frac=1, random_state=seed), 
        #                                 [int(split*meta_df.shape[0]), int(((1.0-split)/2.0+split)*meta_df.shape[0])])

        #train -> 24844
        #val -> 8282
        # trueRows = train[train['target'] == 1] # 434
        # falseRows = train[train['target'] == 0] # 24410
        # # # print(len(trueRows))
        # # # print(f" = > {len(falseRows) - len(trueRows)}")
        # trueReplicas = pd.concat([trueRows]*(math.ceil(len(falseRows)/len(trueRows)))) # 434*57 = 24738


        
        
        # oversampled = falseRows.append(trueReplicas[:len(falseRows) - len(trueRows)], ignore_index=True) # 24410 + 23976  = 48386
        ######################

        if purpose=='train':
            return train['image_name'].tolist(), train['target'].tolist()
        elif purpose=='val':
            return val['image_name'].tolist(), val['target'].tolist()
        elif purpose=='test':
            data_path = os.path.join(directory, "test.csv")
            test_df = pd.read_csv(data_path, sep=',')

            return test_df['image_name'].tolist(), []

    def extract_path_img(self,directory,x):
        file = x + '.jpg'
        
        if file in self.img_part1:
            
            return os.path.join(f'{directory}/HAM10000_images_part_1', file)
        
        elif file in self.img_part2:
            
            return os.path.join(f'{directory}/HAM10000_images_part_2', file)
    def __len__(self):
        """Return number of images in the dataset"""
        return(len(self.images))
        
    def get_labels(self): return self.labels

    def __getitem__(self, index):
        """
        Creates a dict of the data at the given index:
            {"image": <i-th image>,                                              #
             "label": <label of i-th image>} 
        """
        img_root = self.images[index]
        
        #img = Image.open(img_root)
        #trans = transforms.ToTensor()
        #img = trans(img)
        #img = torchvision.io.read_image(img_root)
        img = read_image(img_root)
        if self.transforms is not None:
            img = self.transforms(img)


        if self.labels[index] == 1:
            transformForReplicas = transforms.RandomChoice([
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),
                transforms.RandomAutocontrast(),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomEqualize()
            ])

            transformForReplicas2 = transforms.RandomChoice([
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomPosterize(bits=2),
                
            ])

            img = transformForReplicas(img)
            img = transformForReplicas2(img)
        img = img.float()
     
        if self.purpose == 'test':
            return img
        else:
            return img, self.labels[index]