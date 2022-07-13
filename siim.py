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
class SIIM(Dataset):
    """SIIM dataset class"""

    def __init__(self, root, purpose, seed, split, transforms=None, tfm_on_patch=None):
        self.root_path = root
        self.purpose = purpose
        self.seed = seed
        self.split = split
        self.images, self.labels = self._make_dataset(directory=self.root_path, purpose=self.purpose, seed=self.seed, split=self.split)
        self.transforms = transforms
        self.tfm_on_patch = tfm_on_patch

    @staticmethod
    def _make_dataset(directory, purpose, seed, split):
        """
        Create the image dataset by preparing a list of samples
        :param directory: root directory of the dataset
        :returns: (images, labels) where:
            - images is a numpy array containing all images in the dataset
            - labels is a list containing one label per image
        """

        data_path = os.path.join(directory, "train.csv")
        meta_df = pd.read_csv(data_path, sep=',')
        #(33126, 8)
        
        train, val = train_test_split(meta_df, test_size=split, random_state=seed)
        #do we want to apply stratification here?
        # train, val, test = np.split(meta_df.sample(frac=1, random_state=seed), 
        #                                 [int(split*meta_df.shape[0]), int(((1.0-split)/2.0+split)*meta_df.shape[0])])

        #train -> 24844
        #val -> 8282
        trueRows = train[train['target'] == 1] # 434
        falseRows = train[train['target'] == 0] # 24410
        # print(len(trueRows))
        # print(f" = > {len(falseRows) - len(trueRows)}")
        trueReplicas = pd.concat([trueRows]*(math.ceil(len(falseRows)/len(trueRows)))) # 434*57 = 24738
        
        
        oversampled = falseRows.append(trueReplicas[:len(falseRows) - len(trueRows)], ignore_index=True) # 24410 + 23976  = 48386


        ############### this here needs to go
        '''
        if purpose=='train':
            return ['ISIC_0015719','ISIC_0052212','ISIC_0068279','ISIC_0074268',
            'ISIC_0074311','ISIC_0074542','ISIC_0075663','ISIC_0075914',
            'ISIC_0076262',
            'ISIC_0076545',
            'ISIC_0076742',
            'ISIC_0076995',
            'ISIC_0077472',
            'ISIC_0077735',
            'ISIC_0078703',
            'ISIC_0078712',
            'ISIC_0079038',
            'ISIC_0080512',
            # 'ISIC_0080752',
            # 'ISIC_0080817',
            # 'ISIC_0081956',
            # 'ISIC_0082348',
            # 'ISIC_0082543',
            # 'ISIC_0082934',
            # 'ISIC_0083035',
            # 'ISIC_0084086',
            # 'ISIC_0084270',
            # 'ISIC_0084395'     
            ], [0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0]#,0,0,0,0,0,0,1,0,0,0]
        elif purpose=='val':
            return [
                'ISIC_0085172',
                'ISIC_0085718',
                'ISIC_0085902',
                'ISIC_0086349',
                'ISIC_0086462',
                'ISIC_0086632',
                'ISIC_0086709',
                'ISIC_0087290',
                'ISIC_0087297',
                'ISIC_0088137',
                'ISIC_0088489',
                'ISIC_0089401',
                'ISIC_0089569',
                'ISIC_0089738',
                'ISIC_0090279'], [0,0,1,0,0,0,0,1,0,0,0,1,0,0,0]
        elif purpose=='test':
            return [
                'ISIC_0085172',
                'ISIC_0085718',
                'ISIC_0085902',
                'ISIC_0086349',
                'ISIC_0086462',
                'ISIC_0086632',
                'ISIC_0086709',
                'ISIC_0087290',
                'ISIC_0087297',
                'ISIC_0088137',
                'ISIC_0088489',
                'ISIC_0089401',
                'ISIC_0089569',
                'ISIC_0089738',
                'ISIC_0090279'], []
        '''
        ######################

        if purpose=='train':
            return train['image_name'].tolist(), train['target'].tolist()
        elif purpose=='val':
            return val['image_name'].tolist(), val['target'].tolist()
        elif purpose=='test':
            data_path = os.path.join(directory, "test.csv")
            test_df = pd.read_csv(data_path, sep=',')

            return test_df['image_name'].tolist(), []


    def __len__(self):
        """Return number of images in the dataset"""
        return(len(self.images))


    def __getitem__(self, index):
        """
        Creates a dict of the data at the given index:
            {"image": <i-th image>,                                              #
             "label": <label of i-th image>} 
        """
        if self.purpose == 'test':
            img_root = os.path.join(self.root_path, f"jpeg/test/{self.images[index]}.jpg")
        else:
            img_root = os.path.join(self.root_path, f"jpeg/train/{self.images[index]}.jpg")
        
        #img = Image.open(img_root)
        #trans = transforms.ToTensor()
        #img = trans(img)
        #img = torchvision.io.read_image(img_root)
        img = read_image(img_root)
        img = img.float()
        if self.transforms is not None:
            img = self.transforms(img)
        if self.purpose == 'test':
            return img
        else:
            return img, self.labels[index]
        return img, torch.tensor([self.labels[index]])


        data_dict = {
            'image': img,
            'label': torch.tensor([self.labels[index]])
        }

        


        # if self.tfm_on_patch is None: return data_dict

        # for tfm in self.tfm_on_patch:
        #     data_dict = tfm(data_dict)

        

        return data_dict