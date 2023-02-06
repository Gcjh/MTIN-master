import torch
import torch.nn as nn
from torch.nn.functional import softmax, gumbel_softmax, leaky_relu
from transformers import BertPreTrainedModel, BertModel, RobertaModel
from model.layers import GraphConvolution, sparse_interaction
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter
import math

LAYERS = 3

class MTL(nn.Module):
    def __init__(self, args, gpu):
        super(MTL,self).__init__()
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size
        self.max_len = args.maxlen_train
        self.gpu = gpu
        
        self.bert=BertModel.from_pretrained('covid-bert')
        self.dropout = nn.Dropout(0.5)
        self.bert_dropout = nn.Dropout(0.1)
        
        self.gcn0 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn1 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcn2 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn3 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcn4 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn5 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcn6 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn7 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcn8 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn9 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcna = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcnb = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcnc = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcnd = GraphConvolution(self.hidden_dim, self.hidden_dim)


        self.fc0 = nn.Linear(self.hidden_dim, 3)
        self.fc1 = nn.Linear(self.hidden_dim, 3)

    def attention(self, pool, output, feature, masks = None):
        if masks != None:
            for i in range(feature.shape[0]):
                feature[i] = torch.mul(masks[i], feature[i])
        atten = torch.matmul(feature, pool)
        atten = softmax(atten, dim = 1)
        #print(atten.shape)
        #print(atten)
        logits = torch.mul(output, atten).sum(axis = 1, keepdim = False)
        return logits

    def forward(self, input_ids, attention_mask, graph, target_len):
        ### input_ids.size : 3 batch, sentence_len, hidden_dim
        out = self.bert(input_ids[2], attention_mask[2])
        pool = out.pooler_output
        pool = pool.view(pool.shape[0], pool.shape[1], -1)
        output = out.last_hidden_state
        output = self.bert_dropout(output)
        outputs = torch.stack((output, output))
        
        graph0 = graph[0].squeeze(0)
        graph1 = graph[1].squeeze(0)
               
        feature0 = leaky_relu(self.gcn0(outputs[0], graph1))
        feature1 = leaky_relu(self.gcn1(outputs[1], graph0))
        
        outputs = torch.stack((feature0, feature1))
        
        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
       
        feature0 = leaky_relu(self.gcn2(outputs[0], graph1))
        feature1 = leaky_relu(self.gcn3(outputs[1], graph0))
        
        outputs = torch.stack((feature0, feature1))
        '''
        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
       
        feature1 = leaky_relu(self.gcn4(outputs[0], graph0))
        feature0 = leaky_relu(self.gcn5(outputs[1], graph1))
        
        outputs = torch.stack((feature0, feature1))
        
        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
       
        feature1 = leaky_relu(self.gcn6(outputs[0], graph0))
        feature0 = leaky_relu(self.gcn7(outputs[1], graph1))
        
        outputs = torch.stack((feature0, feature1))
        
        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
       
        feature1 = leaky_relu(self.gcn8(outputs[0], graph0))
        feature0 = leaky_relu(self.gcn9(outputs[1], graph1))
        
        outputs = torch.stack((feature0, feature1))
        
        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
       
        feature1 = leaky_relu(self.gcna(outputs[0], graph0))
        feature0 = leaky_relu(self.gcnb(outputs[1], graph1))
        
        outputs = torch.stack((feature0, feature1))
        
        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
       
        feature1 = leaky_relu(self.gcnc(outputs[0], graph0))
        feature0 = leaky_relu(self.gcnd(outputs[1], graph1))
        
        outputs = torch.stack((feature0, feature1))
        '''
        masks = []
        for i in range(output.shape[0]):
            mask = np.ones((output.shape[1] - target_len[i], 1))
            mask = np.pad(mask, ((target_len[i], 0), (0, 0)), 'constant')
            masks.append(mask)
        masks = torch.tensor(masks).cuda(self.gpu)
        
        logits0 = self.attention(pool, output, outputs[0])  
        pa = outputs[0].mean(axis = 2, keepdim = False)
        logits0 = outputs[0].mean(axis = 1, keepdim = False)
        logits0 = self.dropout(logits0)
        logits0 = self.fc0(logits0)
        
        logits1 = self.attention(pool, output, outputs[1])
        logits1 = outputs[1].mean(axis = 1, keepdim = False)
        logits1 = self.dropout(logits1)
        logits1 = self.fc1(logits1)
        l_sp = 0
        l_sh = 0

        return logits0, logits1, l_sp, l_sh, pa