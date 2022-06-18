import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from sngan_projD.discriminators.resblocks import Block
from sngan_projD.discriminators.resblocks import OptimizedBlock


class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self, num_features=64, num_classes=24, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        # self.block1 = OptimizedBlock(3, num_features)
        self.block1 = OptimizedBlock(3+3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.cond_emb_in = utils.spectral_norm(
                               nn.Embedding(num_classes, num_features ** 2))
            self.l_y_embed = utils.spectral_norm(
                               nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y_emb=None, y_lin=None):
        y_cond = self.cond_emb_in(y_emb).view(y_emb.size(0), -1, 
                                         self.num_features, self.num_features)
        h = x
        h = torch.cat([x, y_cond],1)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        ##### Global pooling #####
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        if y_emb is not None:
            l_y_sum = self.l_y_embed(y_emb)
            
        ##### Inner products for Embedding projections of 1 to 3 objects #####
        output1 = torch.sum(torch.mul(l_y_sum[:,0,:] ,h), dim=1, keepdim=True)
        output2 = torch.sum(torch.mul(l_y_sum[:,1,:] ,h), dim=1, keepdim=True)
        output3 = torch.sum(torch.mul(l_y_sum[:,2,:] ,h), dim=1, keepdim=True)
        output = output + output1 + output2 +output3
        
        return output


class SNResNetConcatDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes, activation=F.relu,
                 dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim_emb = dim_emb
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, dim_emb))
        self.block4 = Block(num_features * 4 + dim_emb, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        if hasattr(self, 'l_y'):
            init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        h = self.block4(h)
        h = self.block5(h)
        h = torch.sum(self.activation(h), dim=(2, 3))
        return self.l6(h)
