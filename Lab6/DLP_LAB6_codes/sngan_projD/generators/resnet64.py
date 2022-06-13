import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from sngan_projD.generators.resblocks import Block


class ResNetGenerator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=64, dim_z=100, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution
        self.cond_emb = nn.Embedding(num_classes, num_features ** 2)
        
        self.l1 = nn.Linear(dim_z, (16-4) * num_features * bottom_width ** 2)
        self.block2 = Block(num_features * (16-4) * 2, num_features * 8,
                       activation=activation, upsample=True, num_classes=num_classes)
        self.block3 = Block(num_features * 8, num_features * 4,
                       activation=activation, upsample=True, num_classes=num_classes)
        self.block4 = Block(num_features * 4, num_features * 2,
                        activation=activation, upsample=True, num_classes=num_classes)
        self.block5 = Block(num_features * 2, num_features,
                        activation=activation, upsample=True, num_classes=num_classes)
        self.b6 = nn.BatchNorm2d(num_features)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)

    def forward(self, z, y=None, **kwargs):
        y_cond = self.cond_emb(y).view(y.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = torch.cat([h, y_cond], 1)
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b6(h))
        return torch.tanh(self.conv6(h))
    
    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)
    
    # def condition_vec(self, y):
    #     return self.cond_embed(y)#.view(1, 1, -1)