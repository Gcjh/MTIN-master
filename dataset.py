#coding:utf-8

import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import sentiment_hash, stance_hash, normalize_adj
import spacy
import pickle as pk
import numpy as np

nlp = spacy.load('en_core_web_sm')


class myDataset(Dataset):
    def __init__(self, filename, theme,tokenizer, maxlen, target = ''):

        self.df = pd.read_csv(filename + '.csv')

        self.texts = []
        with open(filename + '_clean.txt', 'rb') as f:
            for line in f.readlines():
                self.texts.append(line.strip().decode('latin1'))

        pkl_file1 = open('./data/'+target+'/' +theme+'_set/interaction_graph.pkl', 'rb')
        self.interaction_graph = pk.load(pkl_file1)
        pkl_file2 = open('./data/'+target+'/' +theme+'_set/stance_graph.pkl', 'rb')
        self.stance_graph = pk.load(pkl_file2)
        pkl_file3 = open('./data/'+target+'/' +theme+'_set/sentiment_graph.pkl', 'rb')
        self.sentiment_graph = pk.load(pkl_file3)
        
        '''
        length = int(len(self.texts) * 0.8)
        
        self.texts = self.texts[:length]
        self.interaction_graph = self.interaction_graph[:length]
        self.stance_graph = self.stance_graph[:length]
        self.sentiment_graph = self.sentiment_graph[:length]
        
        print(len(self.sentiment_graph))
        input()
        '''
        self.tokenizer = tokenizer

        self.maxlen = maxlen
    
    def __len__(self):
        return int(len(self.df))

    def __getitem__(self, index):
        sentence = nlp(self.texts[index])
        stance_label = stance_hash(self.df.loc[index, 'Stance'])
        sentiment_label = sentiment_hash(self.df.loc[index, 'Sentiment'])
        target = self.df.loc[index, 'Target']
        stance_graph = normalize_adj(np.array(self.stance_graph[index]))
        sentiment_graph = normalize_adj(np.array(self.sentiment_graph[index]))
        interaction_graph = normalize_adj(np.array(self.interaction_graph[index]))
        #stance_graph = np.array(self.stance_graph[index])
        #sentiment_graph = np.array(self.sentiment_graph[index])
        #interaction_graph = np.array(self.interaction_graph[index])
        
        target_len = len(target.split())
        #print(target_len)
        #input()
        tokens = []
        for t in target.split():
            tokens.append(t)
        for token in sentence:
            tokens.append(token.text)        
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        #tokens = ['<s>'] + tokens + ['</s>']
        stance_graph = np.pad(stance_graph, ((1, 1), (1, 1)), 'constant')
        sentiment_graph = np.pad(sentiment_graph, ((1, 1), (1, 1)), 'constant')
        interaction_graph = np.pad(interaction_graph, ((1, 1), (1, 1)), 'constant')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #print(tokens)
        #print(input_ids)
        #input()
        types = [0] * len(input_ids)
        k = self.maxlen - len(input_ids)
        if(k > 0):
            input_ids = input_ids + [0] * k
            types = types + [1] * k
            stance_graph = np.pad(stance_graph, ((0, k), (0, k)), 'constant')
            sentiment_graph = np.pad(sentiment_graph, ((0, k), (0, k)), 'constant')
            interaction_graph = np.pad(interaction_graph, ((0, k), (0, k)), 'constant')
        else :
            input_ids = input_ids[:self.maxlen]
            types = types[:self.maxlen]
            stance_graph = stance_graph[:self.maxlen,:self.maxlen]
            sentiment_graph = sentiment_graph[:self.maxlen,:self.maxlen]
            interaction_graph = interaction_graph[:self.maxlen,:self.maxlen]

        
        input_ids = torch.tensor(input_ids)
        types = torch.tensor(types)
        labels = torch.tensor([stance_label, sentiment_label])
        sentiment_label = torch.tensor(sentiment_label)
        graphs = torch.tensor([stance_graph, sentiment_graph, interaction_graph])
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask, types, labels, graphs, target_len


