import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)
        
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        weight1, weight2, weight3 = weight[:,0,:], weight[:,1,:], weight[:,2,:]
        bias1, bias2, bias3 = bias[:,0,:], bias[:,1,:], bias[:,2,:]
        size = output.size()
        weight1 = weight1.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias1 = bias1.unsqueeze(-1).unsqueeze(-1).expand(size)
        weight2 = weight2.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias2 = bias2.unsqueeze(-1).unsqueeze(-1).expand(size)
        weight3 = weight3.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias3 = bias3.unsqueeze(-1).unsqueeze(-1).expand(size)
        
        out1 = weight1 * output + bias1
        out2 = weight2 * output + bias2
        out3 = weight3 * output + bias3
        output = (out1 + out2 + out3)/3
        return output


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):
    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.weights_emb = nn.Embedding(num_classes, num_features)
        self.biases_emb = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights_emb.weight.data)
        init.zeros_(self.biases_emb.weight.data)

    def forward(self, input, c_emb, **kwargs):
        weight_emb = self.weights_emb(c_emb)
        bias_emb = self.biases_emb(c_emb)
        
        return super(CategoricalConditionalBatchNorm2d, self).forward(
                                              input, weight_emb, bias_emb)

