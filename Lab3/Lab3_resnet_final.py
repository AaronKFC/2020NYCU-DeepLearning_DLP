from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models#, datasets
from torch.utils.data import DataLoader#, Dataset
import copy
# import torch.utils.data as Data
# import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
import pandas as pd
from timeit import default_timer as timer

torch.cuda.empty_cache()

###### Training settings ######
parser = argparse.ArgumentParser(description='PyTorch DLP_Lab2_BCI_training')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (but lr will decay from default 0.005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

###### Model settings ######
parser.add_argument('--image-size', type=int, default=512, metavar='N',
                    help='Resize the image (original=512x512) (default: 400)')
parser.add_argument('--ResNet18', action='store_true', default=True, 
                    help='True=>use ResNet18; False => use ResNet50')
parser.add_argument('--pretrain', action='store_true', default=True, 
                    help='True=>use pretrain; None or False => w/o pretrain')
parser.add_argument('--feature_ext', action='store_true', default=False, 
                    help='True=>Only train FC-layer; False => Finetuing all weights')
parser.add_argument('--add-fc', type=int, default=256, metavar='N',
                    help='add additional FC-layer (default: 256)')

###### Other settings ######
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parser.add_argument("-v", "--activation", type=int, choices=[1, 2, 3], default=2,
#                     help="Select Activation:(1=ELU, 2=ReLU, 3=LeakyReLU)")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


############### train & test data preprocessing ##################
from RetinopathyLoader import RetinopathyLoader

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_size=args.image_size
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize((img_size,img_size)),
            # transforms.RandomResizedCrop((224,224)),
            transforms.RandomRotation(60, resample=False), #expand=False, center=None),
            # transforms.RandomAffine(0, shear=5, scale=(0.95,1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # normalize
            ]),
    'test':
        transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(), # normalize
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


###############################################################################
########## Create Model object and set it ot GPU ###########
########## Define optimizer/loss function ##############
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def select_model(ResNet18=args.ResNet18, pretrain=args.pretrain, feat_ext=args.feature_ext, mid_feat=args.add_fc):
    if args.cuda:
        device = torch.device('cuda')  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if ResNet18:
            DLmodel = models.resnet18(pretrained=pretrain, progress=True)
            model_name = 'ResNet18'
            in_feat = DLmodel.fc.in_features 
            #(224,224)input的話=512，(512,512)input的話=2048
        else:
            DLmodel = models.resnet50(pretrained=pretrain, progress=True)
            model_name = 'ResNet50'
            in_feat = DLmodel.fc.in_features  
            #(224,224)input的話=2048，(512,512)input的話=8912
        
        if pretrain:
            model_name += '_pretrn1'
        else:
            model_name += '_pretrn0'
        
        if feat_ext:
            model_name += '_featExt'
        else:
            model_name += '_finetune'
        
        DLmodel.to(device)
        ##### Decide to do finetuing or feature_extraction #####
        ##### finetuing (update all weights) && feature_extraction (Only train the FC-layer)
        set_parameter_requires_grad(DLmodel, feat_ext)


        ##### Replace the last fully-connected layer #####
        ### Parameters of newly constructed modules have requires_grad=True by default
        num_classes = 5
        mid_feat = mid_feat      
        model_name = model_name +'_addFC' +str(mid_feat)
        if mid_feat == 0:
            DLmodel.fc = nn.Linear(in_feat, num_classes).to(device)
        else:
            DLmodel.fc = nn.Sequential(nn.Linear(in_feat, mid_feat),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(mid_feat, num_classes)).to(device)
    Loss = nn.CrossEntropyLoss()
    
    # Gather the parameters to be optimized/updated in this run. If we are finetuning we will be updating all parameters. 
    # However, if we are doing feature extract method, we will only update the parameters that we have just initialized, i.e. the parameters with requires_grad is True.
    params_to_update = DLmodel.parameters()
    print("Params to learn:")
    if feat_ext:
        params_to_update = []
        for name,param in DLmodel.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in DLmodel.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=args.lr, 
                          momentum=args.momentum, weight_decay=5e-4)
    # optimizer = optim.SGD(DLmodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    # optimizer = optim.Adam(DLmodel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return device, DLmodel, Loss, optimizer, model_name
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########### learning rate scheduling ##############
def adjust_learning_rate(optimizer, epoch):
    if epoch < 3:
       lr = args.lr
    else: #epoch < 20:
       lr = 0.001
    # else: 
    #    lr = 0.0005
       
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

########### Model training function ##################
def train_model(model, criterion, optimizer, num_epochs):
    train_acc_history = []
    test_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  #启用 BatchNormalization 和 Dropout
                adjust_learning_rate(optimizer, epoch)
                running_loss = 0.0
                running_corrects = 0
                for i_batch, sample_batch in enumerate(dataloaders[phase]):
	                # print(f'{epoch}, {phase}, {i_batch}')
                    inputs = sample_batch['image'].to(device)
                    labels = sample_batch['label'].squeeze()
                    labels = labels.to(device)
                    outputs = model(inputs)
	                # assert output.shape[0] == train_Y.shape[0]
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
	     
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                train_loss = running_loss / len(image_datasets[phase])
                train_acc = 100 * running_corrects.double() / len(image_datasets[phase])
                train_acc_history.append(train_acc.cpu().numpy())
                print('train_avg_loss: {:.4f}, train_acc: {:.2f}%'.format(train_loss, train_acc))

            else:
                with torch.no_grad():
                    model.eval()  #不启用 BatchNormalization 和 Dropout
                    running_loss = 0.0
                    running_corrects = 0
                    for i_batch, sample_batch in enumerate(dataloaders[phase]):
                        # print(f'{epoch}, {phase}, {i_batch}')
                        inputs = sample_batch['image'].to(device)
                        labels = sample_batch['label'].squeeze()
                        labels = labels.to(device)
                        outputs = model(inputs)
                        # assert output.shape[0] == train_Y.shape[0]
                        loss = criterion(outputs, labels)
		     
                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    test_loss = running_loss / len(image_datasets[phase])
                    test_acc = 100 * running_corrects.double() / len(image_datasets[phase])
                    test_acc_history.append(test_acc.cpu().numpy())
                    print('test_avg_loss: {:.4f}, test_acc: {:.2f}%'.format(test_loss, test_acc))
            
            if phase == 'test' and test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    ### load the best model weights 
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, test_acc_history

device, DLmodel, Loss, optimizer, model_name = select_model(ResNet18=args.ResNet18)
# print(DLmodel)  # net architecture
start_time = timer()
model_trained, train_acc_history, test_acc_history = train_model(DLmodel, Loss, optimizer, num_epochs=args.epochs)

train_acc_ary = np.array(train_acc_history)
test_acc_ary = np.array(test_acc_history)
best_test_acc = np.max(test_acc_history)
best_epoch = np.argmax(test_acc_ary)+1
print('\nResult of ' +model_name +':')
print(f'The best test accuracy is {best_test_acc:.2f}% at epoch={best_epoch}')

######## save the best model weights
##### sample code for save all models 
# savefilename = 'ResNet_'+str(epoch)+'.tar'
# torch.save({'epoch':epoch, 'state_dict':model.state_dict() }, savefilename)
##### code for save the best model
model_fn = 'save_model/bestweights_' +model_name +'_batch' +str(args.batch_size) +'_totEpo' +str(args.epochs) +'bestEpo' +str(best_epoch) +'.h5'
torch.save(model_trained.state_dict(), model_fn)

end_time = timer()
minutes, seconds = (end_time - start_time)//60, (end_time - start_time)%60
print(f"\nTraining time taken: {minutes} minutes {seconds:.1f} seconds")

############ save acc history ###############
path = 'results/'
train_acc_his = pd.DataFrame(train_acc_ary)
train_acc_fn = path + model_name +'_trainAcc' +'_batch' +str(args.batch_size) +'_totEpoch' + str(args.epochs) 
train_acc_his.to_csv(train_acc_fn +'.csv', index=False)

test_acc_his = pd.DataFrame(test_acc_ary)
test_acc_fn = path +model_name +'_testAcc' +'_batch' +str(args.batch_size) +'_totEpoch' + str(args.epochs) 
test_acc_his.to_csv(test_acc_fn +'.csv', index=False)

##################################################
####### Training History Plot ############
# %matplotlib inline
import matplotlib.pyplot as plt

def acc_history(train_acc, test_acc, model_name):
    epoch = args.epochs
    epo = np.linspace(1, epoch, epoch)
    plt.plot(epo, train_acc)
    plt.plot(epo, test_acc)
    plt.title('Result Comparison--' +model_name)  
    plt.ylabel('Accuracy(%)')
    plt.xlabel('Epochs')
    plt.legend(['train_acc', 'test_acc'], loc='lower right')#, fontsize='small')
    plt.grid()
    plt.show()

acc_history(train_acc_ary, test_acc_ary, model_name)
