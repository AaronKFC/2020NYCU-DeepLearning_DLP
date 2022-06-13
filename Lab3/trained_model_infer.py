# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:47:09 2020

@author: AllenPC
"""
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models#, datasets
from torch.utils.data import DataLoader#, Dataset
import copy
import numpy as np
import pandas as pd

torch.cuda.empty_cache()

###### Inference settings ######
parser = argparse.ArgumentParser(description='PyTorch DLP_Lab2_BCI_training')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (but lr will decay from default 0.01)')
parser.add_argument('--ResNet18', action='store_true', default=False, 
                    help='True=>use ResNet18; False => use ResNet50')
parser.add_argument('--pretrain', action='store_true', default=True, 
                    help='True=>use pretrain; None or False => w/o pretrain')
parser.add_argument('--feature_ext', action='store_true', default=False, 
                    help='True=>Only train FC-layer; False => Finetuing all weights')
parser.add_argument('--add-fc', type=int, default=256, metavar='N',
                    help='add additional FC-layer (default: 256)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from RetinopathyLoader import RetinopathyLoader

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
img_size=512
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize((img_size,img_size)),
            # transforms.RandomResizedCrop((224,224)),
            transforms.RandomRotation(60, resample=False, expand=False, center=None),
            transforms.RandomAffine(0, shear=5, scale=(0.95,1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
            ]),
    'test':
        transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            # normalize
            ])}

image_datasets = {
    'train':RetinopathyLoader(root='data/', mode='train', transform=data_transforms['train']),
     'test':RetinopathyLoader(root='data/', mode='test',  transform=data_transforms['test'])
                  }
 
dataloaders = {'train':DataLoader(image_datasets['train'],
                       batch_size=args.batch_size, shuffle=True, # num_workers=4
                       ),
                'test':DataLoader(image_datasets['test'],
                       batch_size=args.batch_size, shuffle=False, # num_workers=4
                       )}    
 
if args.cuda:
        device = torch.device('cuda')
        
def inference(model, phase='test'):
    model.eval()
    Loss = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    exception_count = 0
    y_pred = []
    for i_batch, sample_batch in enumerate(dataloaders[phase]):

        inputs = sample_batch['image'].to(device)
        labels = sample_batch['label'].squeeze()
        labels = labels.to(device)
        try:
            outputs = model(inputs)
            loss = Loss(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # y_preds = preds.cpu().numpy()
            preds = preds.cpu().numpy()
            for p in preds:
                y_pred.append(p)
        except:
            exception_count += 1
            print(exception_count)
            continue
    num_data = len(image_datasets[phase])
    test_loss = running_loss / num_data
    test_acc = 100 * running_corrects.double() / num_data
    # test_acc_history.append(test_acc.cpu().numpy())
    print('test_avg_loss: {:.4f}, test_acc: {:.3f}%'.format(test_loss, test_acc))
    
    y_pred = np.array(y_pred)
    # y_pred.reshape(1,-1)
    return test_loss, running_corrects.cpu().numpy(), num_data, test_acc, y_pred

model_infer = models.resnet18(pretrained=False).to(device)
# model_infer = models.resnet50(pretrained=False).to(device)
in_feat = model_infer.fc.in_features
num_classes = 5
mid_feat=args.add_fc
# model_infer.fc = nn.Linear(in_feat, num_classes).to(device)
model_infer.fc = nn.Sequential(nn.Linear(in_feat, mid_feat),
                            nn.ReLU(inplace=True),
                            nn.Linear(mid_feat, num_classes)).to(device)
# model_infer.fc = nn.Sequential(
#                  nn.Linear(2048, 256),
#                  nn.ReLU(inplace=True),
#                  nn.Linear(256, 5)).to(device)
model_infer.load_state_dict(torch.load(
      'save_model/bestweights_ResNet18_pretrn1_finetune_addFC256_batch32_totEpo20bestEpo13.h5'))

test_loss, running_corrects, num_data, test_acc, y_pred = inference(model_infer, phase='test')



from itertools import product

class ConfusionMatrixDisplay:
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):

        # check_matplotlib_support("ConfusionMatrixDisplay.plot")
        # import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
                
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")
        ax.set_title('Normalized confusion matrix')
        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_map(y_true, y_pred, classes, cmap=plt.cm.Blues, normalize=None):

    cm = confusion_matrix(y_true, y_pred)#, labels=labels, normalize=normalize)
    # cm = cm/len(y_true)
    print(cm)
    true_cls_num = np.sum(cm, axis=1)
    cm = np.float32(cm)
    for i in range(len(true_cls_num)):
        cm[i,:] = cm[i,:] / true_cls_num[i]
    cm = np.around(cm, decimals=3)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    return disp.plot(include_values=True, cmap=cmap, ax=None, 
                     xticks_rotation='horizontal', values_format=None), true_cls_num

label = pd.read_csv('test_label.csv')
y_true = np.squeeze(label.values)
classes = ['0', '1', '2', '3', '4']
_, true_cls_num = plot_confusion_map(y_true, y_pred, classes, cmap=plt.cm.Blues, normalize=None)


# from sklearn.metrics import plot_confusion_matrix
# plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues, normalize=False)
