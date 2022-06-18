from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.autograd import Variable

import numpy as np
import pandas as pd
from timeit import default_timer as timer


###### Training settings ######
parser = argparse.ArgumentParser(description='PyTorch DLP_Lab2_BCI_training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (but lr will decay from default 0.001)')
parser.add_argument('--useEEG', action='store_true', default=True, 
                    help='True=>use EEGnet; None or False => use DeepConvNet')
parser.add_argument("-v", "--activation", type=int, choices=[1, 2, 3], default=2,
                    help="Select Activation:(1=ELU, 2=ReLU, 3=LeakyReLU)")
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


############### train & test data preprocessing ##################
def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)
    
    ##### tunr to torch tensor and form train_loader for batch training
    train_data_tensor = torch.FloatTensor(train_data)
    train_label_tensor = torch.from_numpy(train_label).long()
    torch_dataset = Data.TensorDataset(train_data_tensor, train_label_tensor)
    
    train_loader = Data.DataLoader(
                        dataset=torch_dataset,      # torch TensorDataset format
                        batch_size=args.batch_size,      # mini batch size
                        shuffle=True,               # True代表打亂dataset
                        # num_workers=1,              # 多線程讀取data(但在windows會報錯，乾脆取消掉)
                        )

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label, train_loader
##############################################################################
################## Create Deep Learning Model ################################
'''############# Define EEGnet here ##############'''
class EEGNet(nn.Module):
    def __init__(self, Activate):
        super(EEGNet, self).__init__()
        self.alpha = 1
        self.cvks1 = 61
        self.cvks2 = (2,1)
        self.cvks3 = 25
        self.out_ch1 = int(16*self.alpha)
        self.out_ch2 = int(32*self.alpha)
        self.out_ch3 = int(32*self.alpha)
        # self.Activate = Activate
        
        ##### layer1 #####
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.out_ch1, 
                      kernel_size=(1,self.cvks1), stride=(1,1), 
                      padding=(0,int((self.cvks1-1)/2)), bias=False),
            nn.BatchNorm2d(num_features=self.out_ch1, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True)
        )
        ##### layer2 #####
        self.depthwiseConv = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch2,
                      kernel_size=self.cvks2, stride=(1,1), 
                      groups=self.out_ch1, bias=False),
            nn.BatchNorm2d(num_features=self.out_ch2, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            Activate,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        ##### layer3 #####
        self.separableConv = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(in_channels=self.out_ch2, out_channels=self.out_ch3, 
                      kernel_size=(1,self.cvks3), stride=(1,1), 
                      padding=(0,int((self.cvks3-1)/2)), bias=False),
            nn.BatchNorm2d(num_features=self.out_ch3, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            Activate,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25)
        )
        ##### FC layer #####
        self.classify = nn.Sequential(
            nn.Linear(in_features=self.out_ch3*23, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstConv(x)       # Layer 1
        x = self.depthwiseConv(x)   # Layer 2
        x = self.separableConv(x)   # Layer 3
        # FC Layer
        x = x.view(x.size(0), -1)   #原dim=[64, 32, 1, 23]flatten成[64,32*1*23]
        x = self.classify(x)
        return x

'''############# Define DeepConvNet here ##############'''
class DeepConvNet(nn.Module):
    def __init__(self, Activate):
        super(DeepConvNet, self).__init__()
        self.alpha = 1
        self.cvks0 = 5
        self.cvks1 = (2,1)
        self.cvks2 = 5
        self.cvks3 = 5
        self.cvks4 = 5
        self.out_ch0 = int(25*self.alpha)
        self.out_ch1 = int(25*self.alpha)
        self.out_ch2 = int(50*self.alpha)
        self.out_ch3 = int(100*self.alpha)
        self.out_ch4 = int(200*self.alpha)
        # self.Activate = Activate
        
        ##### layer0 #####
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.out_ch0, 
                      kernel_size=(1,self.cvks0), stride=(1,1), 
                      padding=0, bias=False),
            )
        ##### layer1 #####
        self.block1 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(in_channels=self.out_ch0, out_channels=self.out_ch1,
                      kernel_size=self.cvks1, stride=(1,1), 
                      groups=self.out_ch1, bias=False),
            nn.BatchNorm2d(num_features=self.out_ch1, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            Activate,
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=0.5)
            )
        ##### layer2 #####
        self.block2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch2, 
                      kernel_size=(1,self.cvks2), stride=(1,1), 
                      padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.out_ch2, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            Activate,
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=0.5)
            )
        ##### layer3 #####
        self.block3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(in_channels=self.out_ch2, out_channels=self.out_ch3, 
                      kernel_size=(1,self.cvks3), stride=(1,1), 
                      padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.out_ch3, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            Activate,
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=0.5)
            )
        ##### layer4 #####
        self.block4 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(in_channels=self.out_ch3, out_channels=self.out_ch4, 
                      kernel_size=(1,self.cvks4), stride=(1,1), 
                      padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.out_ch4, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            Activate,
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=0.5)
            )
        ##### FC layer #####
        self.classify = nn.Sequential(
            nn.Linear(in_features=self.out_ch4*43, out_features=2, bias=True)
            )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)      
        # FC Layer
        x = x.view(x.size(0), -1) #原dim=[64, 32, 1, 23]flatten成[64,32*1*23]
        x = self.classify(x)
        return x

###############################################################################
########## 1. Create Model object and set it ot GPU ###########
########## 2. Define optimizer/loss function ##############
def select_model(useEEG=args.useEEG, acti = args.activation):
    if acti == 1:
        Activate = nn.ELU()
        acti_name = 'ELU'
    elif acti == 2:
        Activate = nn.ReLU()
        acti_name = 'ReLU'
    else:
        Activate = nn.LeakyReLU()
        acti_name = 'LeakyReLU'
        
    if args.cuda:
        device = torch.device('cuda')
        if args.useEEG:
            DLmodel = EEGNet(Activate = Activate)
            model_name = 'EEGnet'
        else:
            DLmodel = DeepConvNet(Activate = Activate)
            model_name = 'DeepConvNet'
        DLmodel.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(DLmodel.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return device, DLmodel, Loss, optimizer, acti_name, model_name
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########### learning rate scheduling ##############
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
       lr = 0.001
    elif epoch < 20:
       lr = 0.001
    # elif epoch < 160:
        # lr = 0.001
    else: 
       lr = 0.001
       
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

########### Model training function ####################
def train(epoch, model, train_loader):
    model.train()   #Set model to training mode
    adjust_learning_rate(optimizer, epoch)
    
    running_loss = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if args.cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()   #Clean gradient
        output = model(inputs)
        train_loss = Loss(output, labels)
        train_loss.backward()    #Backward gradient
        optimizer.step()   #Update weight
        
        # running_loss += train_loss.data#[0]
        running_loss.append(train_loss.cpu().detach().numpy())#[0]
    running_loss = np.average(np.array(running_loss))
    # print(f'train_avg_oss:{running_loss:.4f}')
 

########## Inference function for trained model ##################
def inference(model, X, Y):
    model.eval()   #Set model to evaluation mode
    # correct = 0
    
    data = torch.FloatTensor(X)
    target = torch.tensor(Y).long()

    if args.cuda:
        data, target = data.to(device), target.to(device)
        
    with torch.no_grad():
        # print(model.requires_grad)
        output = model(data)
    
    ##### 注意用torch.nn.CrossEntropyLoss的model output是log-probability
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct = pred.eq(target.data).cpu().sum()
    num_data = len(data)
    acc = 100*correct.numpy()/num_data
    
    avg_loss = Loss(output, target)  #注意target需要unsqueeze_(dim=0)是逐筆inference才要用
    avg_loss = avg_loss.cpu().numpy() # loss function already averages over batch size

    return avg_loss, correct, num_data, acc


########### run and save model #################
train_data, train_label, test_data, test_label, train_loader = read_bci_data()
device, DLmodel, Loss, optimizer, acti_name, model_name = select_model(useEEG=args.useEEG)
print(DLmodel)  # net architecture

start_time = timer()
train_acc_ls = []
test_acc_ls = []
for epoch in range(1, args.epochs + 1):
    print(f'\nEpoch{epoch:3d}')
    
    # train(epoch,DLmodel, train_data, train_label)
    train(epoch, DLmodel, train_loader)
    train_avg_loss, train_correct, train_num_data, train_acc = inference(DLmodel, train_data, train_label)
    train_acc_ls.append(train_acc)
    print('train_ave_loss: {:.4f}, train_acc: {}/{} ({:.3f}%)'.format(
          train_avg_loss, train_correct, train_num_data, train_acc))
    
    test_avg_loss, test_correct, test_num_data, test_acc = inference(DLmodel, test_data, test_label)
    test_acc_ls.append(test_acc)
    
    print('test_ave_loss: {:.4f}, test_acc: {}/{} ({:.3f}%)'.format(
          test_avg_loss, test_correct, test_num_data, test_acc))
     
    ######## save each model at each epoch
    # savefilename = 'EEGNet_'+str(epoch)+'.tar'
    # torch.save({'epoch': epoch,
    #             'state_dict': model.state_dict(),
    #            }, savefilename)
train_acc_ary = np.array(train_acc_ls)
test_acc_ary = np.array(test_acc_ls)
best_test_acc = np.max(test_acc_ary)
best_test_acc_epo = np.argmax(test_acc_ary)
print('\nResult of ' +model_name+'_'+acti_name +':')
print(f'The best test accuracy is {best_test_acc:.2f}% at epoch={best_test_acc_epo}')

end_time = timer()
minutes = (end_time - start_time)//60
seconds = (end_time - start_time)%60
print(f"\nTraining time taken: {minutes} minutes {seconds:.1f} seconds")

############ save acc history ###############
path = 'results/'
train_acc_his = pd.DataFrame(train_acc_ary)
train_acc_fn = path + model_name +'_'+ acti_name +'_trainAcc'+'_batch' +str(args.batch_size) +'_Epoch' + str(args.epochs) 
train_acc_his.to_csv(train_acc_fn +'.csv', index=False)

test_acc_his = pd.DataFrame(test_acc_ary)
test_acc_fn = path +model_name+'_'+ acti_name +'_testAcc'+'_batch' +str(args.batch_size) +'_Epoch' + str(args.epochs) 
test_acc_his.to_csv(test_acc_fn +'.csv', index=False)

##################################################
####### Training History Plot ############
# %matplotlib inline
import matplotlib.pyplot as plt

def acc_history(train_acc, test_acc):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Activation function Comparison--'+model_name)  #(DeepConv)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend([(acti_name+'_train'), (acti_name+'_test')], loc='lower right')
    # ['elu_train', 'elu_test', 'relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test']
    plt.show()

acc_history(train_acc_ary, test_acc_ary)
