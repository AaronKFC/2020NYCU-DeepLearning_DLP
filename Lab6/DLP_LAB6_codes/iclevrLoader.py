# import pandas as pd
from torch.utils import data
import numpy as np
from os.path import isfile, join
import torch
from torchvision import transforms#, datasets
from PIL import Image
from skimage import io#, transform
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils, datasets
import json

info_path = 'dataset/'

def getData(mode):
    with open(info_path+'objects.json', 'r', encoding='utf-8') as f:
            objects_info = json.load(f)
    if mode == 'train':
        with open(info_path+'train.json', 'r', encoding='utf-8') as f:
            train_info = json.load(f)
        inputs = []
        labels = []
        for fn, obj_ls in train_info.items():
            inputs.append(fn)
            objects = np.zeros(24)
            for obj in obj_ls:
                objects[objects_info[obj]]=1
            labels.append(objects)
        return inputs, labels
    else:
        with open(info_path+'test.json', 'r', encoding='utf-8') as f:
            test_info = json.load(f)
        inputs = []
        labels = []
        for obj_ls in test_info:
            objects = []
            for obj in obj_ls:
                objects.append(objects_info[obj])
            labels.append(objects)
        return inputs, labels
    
# a,b = getData('train')


class iclevrLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #step1.Get the image path from 'self.img_name' and load it.
        img_fn = self.img_name[idx]
        img_fn = join(self.root,img_fn) 
        image_fn = Image.open(img_fn).convert('RGB')
        
        #step2. Get the ground truth label from self.label
        label_fn = torch.Tensor([self.label[idx]]).long().squeeze()
        
        #step3. Transform the images during the training phase
        if self.transform:
            image_fn = self.transform(image_fn)
        else:
            image_fn = io.imread(img_fn)
            image_fn = np.array(image_fn)
            image_fn = torch.FloatTensor(image_fn)
        
        #step4. Return processed image and label
        sample = {'image': image_fn, 'label': label_fn}
        return sample



#################################################################################
#################################################################################

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                   std=[0.5, 0.5, 0.5])
        
# data_transform = transforms.Compose([
#         transforms.Resize((64,64)),
#         # transforms.RandomSizedCrop(224),
#         # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#         ])


# img_dataset = iclevrLoader(root='iclevr/', mode='train', transform=data_transform)

# image = img_dataset[0]['image']#[:,:,0]
# print(img_dataset[0]['image'])
# print(img_dataset[0]['image'].size())

# label = img_dataset[55]['label']
# print(label)
# print(label.size())


# # import random
# # random.shuffle(train_loader)
# # train_loader = zip(img_dataset)
# train_loader = img_dataset
# for idx, sample_batch in enumerate(train_loader):
#         # print(f'{epoch}, {phase}, {i_batch}')
#         x_ = sample_batch['image']#.cuda()#.to(device)
#         print(x_.size())
#         y_ = sample_batch['label']#.squeeze().squeeze()
#         print(y_.size())
#         print(idx)



# x = torch.arange(5)
# torch.nonzero(x)
# torch.nonzero(x).squeeze()
# torch.nonzero(x).squeeze().size()

# test_loader = iclevrLoader(root='iclevr/', mode='test', transform=data_transform)


# print(torch.max(img_dataset[0]['image']))
# print(torch.min(img_dataset[0]['image']))
# print((img_dataset[0]['label']))

# dataloader = DataLoader(dataset=img_dataset, 
#                         batch_size=2000,
#                         shuffle=True, 
#                         # num_workers=4
#                         )

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['label'].size())

      
