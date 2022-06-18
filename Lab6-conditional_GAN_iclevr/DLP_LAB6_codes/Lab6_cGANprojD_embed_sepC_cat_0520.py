# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:33:03 2020

@author: AllenPC
"""
import random
import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
# import itertools
import pickle
import imageio
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms#, datasets
# from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader#, Dataset
import torchvision
from iclevrLoader import iclevrLoader, getData
from iclevr_Evaluator import evaluation_model

from sngan_projD.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from sngan_projD.generators.resnet64 import ResNetGenerator
import sngan_projD.losses as L

# device = torch.device('cuda')

torch.cuda.empty_cache()

z_dim = 100
num_label = 24
img_size = 64
# data_loader

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])

data_path = 'iclevr/'
train_datasets = iclevrLoader(root=data_path, mode='train', transform=data_transform)
train_loader = DataLoader(train_datasets, batch_size=32, shuffle=True) #, num_workers=4)
# test_loader = iclevrLoader(root='iclevr/', mode='test', transform=data_transform)
temp  = train_datasets[0]['image']#[:,:,0]
if (temp.shape[1] != img_size) or (temp.shape[2] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)


##### label preprocess for evaluator input
Evaluator = evaluation_model()

num_label = 24
onehot = torch.zeros(num_label,num_label)
one_ls = list(np.linspace(0,(num_label-1),num_label))
## input_object.scatter_(dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中。
onehot = onehot.scatter_(1, torch.LongTensor(one_ls).view(num_label, 1), 1).view(num_label, num_label, 1, 1)

_, test_label = getData('test')
eval_y_test_lb = []
for i in range(len(test_label)):
    lb = torch.zeros(num_label, 1, 1)
    # lb = torch.zeros(num_label)
    for n in test_label[i]:
        lb += onehot[n]
    eval_y_test_lb.append(lb.unsqueeze(0))
eval_y_test_lb = torch.cat(eval_y_test_lb, axis=0).squeeze().squeeze()
# print(eval_y_test_lb)


### fixed noise & label
################### fixed noise & label for Generation #####################
##### set conditial indices for testing img generation
_, test_label = getData('test')
y_test = []
for b in range(len(test_label)):
    y_lb = torch.LongTensor(test_label[b]) +1
    cy = torch.zeros([1, 3]).cuda() #[batch, channel, img_size, img_size]
    i = 0
    for n in y_lb:
        cy[0,i] += n
        i+=1
    y_test.append(cy)
fixed_y_test_c = torch.cat(y_test, axis=0).type(torch.LongTensor)
# print(fixed_y_test_c)

###### 產生test data的32組的gaussian noise
fixed_z_ = torch.randn((32, z_dim)).view(-1, z_dim)

with torch.no_grad():
    fixed_z_, fixed_y_test_c = Variable(fixed_z_.cuda()), Variable(fixed_y_test_c.cuda())

def Gen_eval():
    G.eval()
    test_images_raw = G(fixed_z_, fixed_y_test_c)
    G.train()
    
    ### img & label for Evaluator
    eval_acc = Evaluator.eval(test_images_raw, eval_y_test_lb)
    # print(f'Classification accuracy: {eval_acc:.4f}')
    return eval_acc

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    G.eval()
    test_images_raw = G(fixed_z_, fixed_y_test_c)
    # print(test_images_raw.size())
    test_images = (test_images_raw.data + 1) / 2.0
    G.train()
    
    ### img & label for Evaluator
    eval_acc = Evaluator.eval(test_images_raw, eval_y_test_lb)
    print(f'Classification accuracy: {eval_acc:.4f}')
    
    ### grid_img plot
    grid_img = torchvision.utils.make_grid(test_images.cpu(), nrow=8)
    grid_img = grid_img.detach()#.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))
    
    label = 'Epoch {0}'.format(num_epoch)
    plt.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    return eval_acc

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def test_acc_hist(hist, show = False, save = False, path = 'TestAcc_hist.png'):
    x = range(len(train_hist['Test_acc']))
    y = train_hist['Test_acc']
    plt.plot(x, y, label='Test_acc')

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

def rand_objects(mini_batch):
    rand_objs = []
    for i in range(mini_batch):
        obj_idx = np.linspace(0,23,24)
        randint_obj = random.randint(1, 3)
        rand_o = random.choices(obj_idx, k=randint_obj)
        rand_o = np.sort(rand_o)
        rand_objs.append(torch.LongTensor(rand_o))
    return rand_objs
# rand_objs = rand_objects(mini_batch)

# ###### gradient penalty ######
# # Loss weight for gradient penalty
# lambda_gp = 10
# def compute_gradient_penalty(D, real_samples, condition_y, fake_samples, mini_batch):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = torch.from_numpy(np.random.random((mini_batch, 1, 1, 1))).cuda()
    
#     # Get random interpolation between real and fake samples
#     intplo_real = torch.mul(alpha, real_samples)
#     intplo_fake = torch.mul((1 - alpha), fake_samples)
#     interpolates = (intplo_real + intplo_fake).requires_grad_(True)
#     d_interpolates = D(interpolates.type(torch.FloatTensor).cuda(), condition_y)
    
#     fake = Variable(torch.ones(mini_batch, 1), requires_grad=False)
    
#     # Get gradient w.r.t. interpolates
#     gradients = autograd.grad(outputs=d_interpolates.cuda(),
#                               inputs=interpolates.cuda(),
#                               grad_outputs=fake.cuda(), create_graph=True,
#                               retain_graph=True, only_inputs=True)[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

##### training parameters
batch_size = 32
lr = 0.0002
train_epoch = 60
n_critic = 2 #train G every 5 iters
# clip_value = 0.03


##### Create Network Object
num_obj_state = 24+1
G = ResNetGenerator(num_features=64, dim_z=100, bottom_width=4, 
                    activation=F.relu, num_classes=num_obj_state, distribution='normal')
D = SNResNetProjectionDiscriminator(64, num_obj_state, F.relu)

# D.load_state_dict(torch.load(
#         'save_model/iclevr_cWGAN_discriminator_param.pkl'))
# G.load_state_dict(torch.load(
#         'save_model/iclevr_cWGAN_generator_param.pkl'))
G.cuda()
D.cuda()

##### Selecty Loss
# adver_loss = nn.BCELoss()   # Binary Cross Entropy loss
# adver_loss = nn.MSELoss().cuda()
dis_loss = L.DisLoss(loss_type='hinge')
gen_loss = L.GenLoss(loss_type='hinge')

##### Adam optimizer
# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.3, 0.9))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.3, 0.9))

##### results save folder
root = 'iclevr_cGANprojD_results3/'
model = 'iclevr_cGANprojD_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['Test_acc'] = []
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# lr_decay = 4
lr_decay = np.linspace(lr, lr/500, train_epoch-6)
lr_decay = np.round(lr_decay,8)

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    
    # fixed learning rate
    if (epoch+1) < 8:
        G_optimizer.param_groups[0]['lr'] = lr
        D_optimizer.param_groups[0]['lr'] = lr
        print(f'learning_rate={lr}')
    # learning rate decay
    if (epoch+1) >= 8:
        # lr_decay = lr_decay[epoch-7]
        G_optimizer.param_groups[0]['lr'] = lr_decay[epoch-7]
        D_optimizer.param_groups[0]['lr'] = lr_decay[epoch-7]
        print(f'learning_rate={lr_decay[epoch-7]}')
    
    
    # # Adversarial ground truths
    # y_real_ = torch.ones(batch_size,1)
    # y_fake_ = torch.zeros(batch_size,1)
    # y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    epoch_start_time = time.time()
    num_iter = 0
    
    for i_batch, sample_batch in enumerate(train_loader):
        x_ = sample_batch['image']#.cuda()#.to(device)
        y_ = sample_batch['label']#.squeeze()
        
        mini_batch = x_.size()[0]
        ##### 為了最後batch_size無法整除data總數時而設的一步
        # if mini_batch != batch_size:
        #     # y_real_ = torch.ones(mini_batch)
        #     # y_fake_ = torch.zeros(mini_batch)
        #     y_real_ = torch.ones(mini_batch,1)
        #     y_fake_ = torch.zeros(mini_batch,1)
        #     y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                
        
        '''######## train discriminator D ######'''
        D.zero_grad()
        #####----- input Real img for D -----#####
        ##### y_fill_: conditional input to C
        ##### 注意沒有object的index為0
        y_label = []
        for b in range(len(y_)):
            y_lb = np.nonzero(y_[b]) +1
            # print(y_lb)
            cy = torch.zeros([1, 3]).cuda() #[batch, channel, img_size, img_size]
            i = 0
            for n in y_lb:
                cy[0,i] += n[0]
                i+=1
            y_label.append(cy)
        y_label_c = torch.cat(y_label, axis=0).type(torch.LongTensor)
        # print(y_label_c)
        
        x_, y_label_c = Variable(x_.cuda()), Variable(y_label_c.cuda())
        
        D_result_real = D(x_, y_label_c)#.squeeze()
        # D_real_loss = adver_loss(D_result, y_real_)
        

        #####----- G generate fake img for D -----#####
        # z_ = torch.randn((mini_batch, z_dim)).view(-1, z_dim, 1, 1)
        z_ = torch.randn((mini_batch, z_dim)).view(-1, z_dim)
        # y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
        rand_objs = rand_objects(mini_batch)
        
        ##### y_label_: conditional input to G
        y_rand = []
        for b in range(len(rand_objs)):
            y_lb = rand_objs[b] +1
            cy = torch.zeros([1, 3]).cuda() #[batch, channel, img_size, img_size]
            i = 0
            for n in y_lb:
                cy[0,i] += n
                i+=1
            y_rand.append(cy)
        y_rand_c = torch.cat(y_rand, axis=0).type(torch.LongTensor)#.cuda()
        # print(y_rand_c)
        
        z_, y_rand_c = Variable(z_.cuda()), Variable(y_rand_c.cuda())
        
        ##### G and D forward
        G_result = G(z_, y_rand_c)
        D_result_fake = D(G_result, y_rand_c)#.squeeze()
        # D_fake_loss = adver_loss(D_result, y_fake_)
        # D_fake_score = D_result.data.mean()
        
        
        ##### G and D backward
        # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(D, x_, y_rand_c, G_result, mini_batch)
        # Adversarial loss
        # D_train_loss = -torch.mean(D_result_real) + torch.mean(D_result_fake) + lambda_gp * gradient_penalty
        ### Hinge loss
        D_train_loss = dis_loss(D_result_fake, D_result_real)# + lambda_gp * gradient_penalty
        # D_train_loss = D_real_loss + D_fake_loss + D_real_but_fake_loss
        # D_train_loss = D_real_loss + D_fake_loss
        
        D_train_loss.backward()
        D_optimizer.step()
        
        # # Clip weights of discriminator
        # # clip_value = 0.05
        # for p in D.parameters():
        #     p.data.clamp_(-clip_value, clip_value)
        
        D_losses.append(D_train_loss.item())


        '''######## train Generator G ######'''
        if (epoch+1) >= 20:
            n_critic=3
        if (epoch+1) >= 40:
            n_critic=4
        # print(n_critic)
        if i_batch % n_critic == 0:
            G.zero_grad()
            
            z_ = torch.randn((mini_batch, z_dim)).view(-1, z_dim)
            rand_objs = rand_objects(mini_batch)
            
            ##### y_label_c: conditional input to G
            y_rand = []
            for b in range(len(rand_objs)):
                y_lb = rand_objs[b] +1
                # print(y_lb)
                cy = torch.zeros([1, 3]).cuda() #[batch, channel, img_size, img_size]
                i = 0
                for n in y_lb:
                    cy[0,i] += n
                    i+=1
                y_rand.append(cy)
            y_rand_c = torch.cat(y_rand, axis=0).type(torch.LongTensor)#.cuda()
            # print(y_rand_c)
            
                        
            z_, y_rand_c = Variable(z_.cuda()), Variable(y_rand_c.cuda())
    
            G_result = G(z_, y_rand_c)
            D_result_fake = D(G_result, y_rand_c)#.squeeze()
            # print(D_result_fake)
    
            # G_train_loss = adver_loss(D_result, y_real_)
            # G_train_loss = -torch.mean(D_result_fake)
            G_train_loss = gen_loss(D_result_fake, None)
    
            G_train_loss.backward()
            G_optimizer.step()
    
            G_losses.append(G_train_loss.item())

        num_iter += 1

        if (num_iter % 100) == 0:
            with torch.no_grad():
                eval_acc_iter = Gen_eval()
                print(f'epoch:{epoch+1} -- iter:{num_iter} complete! (test_acc={eval_acc_iter:.4f})')
            if eval_acc_iter > 0.52:
                gen_fn = root +model +'G_epo' +str(epoch+1) +'_iter' +str(num_iter) +'_acc' +str(np.round(eval_acc_iter,3)) +'.pkl' 
                dis_fn = root +model +'D_epo' +str(epoch+1) +'_iter' +str(num_iter) +'_acc' +str(np.round(eval_acc_iter,3)) +'.pkl' 
                torch.save(G.state_dict(), gen_fn)
                torch.save(D.state_dict(), dis_fn)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    with torch.no_grad():
        # eval_acc_epo = Gen_eval()
        eval_acc_epo = show_result((epoch+1), save=True, path=fixed_p)
        train_hist['Test_acc'].append(eval_acc_epo)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        if eval_acc_epo > 0.53:
            gen_fn = root +model +'G_epo' +str(epoch+1) +'_acc' +str(np.round(eval_acc_epo,3)) +'.pkl' 
            dis_fn = root +model +'D_epo' +str(epoch+1) +'_acc' +str(np.round(eval_acc_epo,3)) +'.pkl' 
            torch.save(G.state_dict(), gen_fn)
            torch.save(D.state_dict(), dis_fn)


end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_finished.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_finished.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
test_acc_hist(train_hist, save = True, path=root + model + 'TestAcc_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
