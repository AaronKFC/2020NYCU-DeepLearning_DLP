import pandas as pd
from torch.utils import data
import numpy as np
from os.path import isfile, join

import torch
from torchvision import transforms#, datasets
from PIL import Image
import matplotlib.pyplot as plt

import torch
from skimage import io#, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)
            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))
        
        self.transform = transform

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, idx):
        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           step2. Get the ground truth label from self.label
           step3. A.Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                    rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                  B.In the testing phase, if you have a normalization process during the training phase, 
                    you only need to normalize the data.
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]       
           step4. Return processed image and label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #step1.Get the image path from 'self.img_name' and load it.
        img_fn = self.img_name[idx]+'.jpeg'   # img_fn = [self.root+f+'.jpeg'  for f in self.img_name]# if isfile(join(self.root, f, '.jpeg'))]
        img_fn = join(self.root,img_fn) 
        image_fn = Image.open(img_fn).convert('RGB')
        
        #step2. Get the ground truth label from self.label
        label_fn = torch.Tensor([self.label[idx]]).long()
        
        #step3. Transform the images during the training phase
        #       Do normalization only during testing phase
        if self.transform:
            image_fn = self.transform(image_fn)
        else:
            image_fn = io.imread(img_fn)
            image_fn = np.array(image_fn)/ 255
            image_fn = torch.FloatTensor(image_fn)
        
        #step4. Return processed image and label
        sample = {'image': image_fn, 'label': label_fn}
        
        return sample



#################################################################################
#################################################################################

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                  std=[0.5, 0.5, 0.5])
        
# data_transform = transforms.Compose([
#         # transforms.Resize((224,224)),
#         transforms.RandomSizedCrop(224),
#         transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # normalize
#         ])

# # img = RetinopathyLoader(root='data/', mode='train', transform=None)
# img_dataset = RetinopathyLoader(root='data/', mode='train', transform=data_transform)
# # image = img_dataset[0]['image'][:,:,0]
# # label = img_dataset[0]['label']
# # print(img_dataset[0]['image'][:,:,1])
# # print(img_dataset[0]['image'].size())
# # print(torch.max(img_dataset[0]['image']))
# # print(torch.min(img_dataset[0]['image']))
# # print((img_dataset[0]['label']))

# dataloader = DataLoader(dataset=img_dataset, 
#                         batch_size=2000,
#                         shuffle=True, 
#                         # num_workers=4
#                         )

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['label'].size())

      
