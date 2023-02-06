import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        adj = adj.to(torch.float32)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class sparse_interaction(Module):
    def __init__(self, max_len):
        super(sparse_interaction, self).__init__()
        self.theta0 = Parameter(torch.FloatTensor(max_len, 2))
        self.theta1 = Parameter(torch.FloatTensor(max_len, 2))
        #self.LAYERS = 3
        self.alpha = 0.4 #Hyperparameter
        self.beta = (1-self.alpha) / 2
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta0.size(1))
        self.theta0.data.uniform_(-stdv, stdv)
        self.theta1.data.uniform_(-stdv, stdv)
    
    def forward(self, outputs, layer):
        theta0 = F.gumbel_softmax(self.theta0, tau=1.0, hard=True)
        theta1 = F.gumbel_softmax(self.theta1, tau=1.0, hard=True)
        t0 = theta0[:,1:]
        t1 = theta1[:,1:]
        f0 = F.leaky_relu(outputs[0] + torch.mul(outputs[2], t0))
        f1 = F.leaky_relu(outputs[1] + torch.mul(outputs[2], t1))
        #f0 = F.leaky_relu(outputs[0] + outputs[2])
        #f1 = F.leaky_relu(outputs[1] + outputs[2])
        
        theta0 = theta0[:,1:].sum(axis = 0, keepdim = False)
        theta1 = theta1[:,1:].sum(axis = 0, keepdim = False)
        
        delta = (torch.log(torch.mul(theta0, theta1) + 1) * 0.5).view(-1)

        f2 = F.leaky_relu(outputs[2] * self.alpha + f0 * self.beta + f1 * self.beta)

        fs = torch.stack((f0, f1, f2))

        return fs, delta, t0, t1