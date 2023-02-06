#coding:utf-8

import pickle as pk
import pandas as pd
import numpy as np
import spacy
from utils import sentiment_hash, stance_hash

nlp = spacy.load('en_core_web_sm')

class Corpus:
    theme = ''
    dataset = ''
    words_number = 0
    word2id = {}
    id2word = {}

    #词频
    word_freq = {}
    count = 0

    #stance中正标签词数
    stance_0_count = 0
    #stance中负标签词数
    stance_1_count = 0
    
    #sentiment中正标签词数
    sentiment_0_count = 0
    #sentiment中负标签词数
    sentiment_1_count = 0

    #单词在stance中正负标签出现频率
    word_stance_0_freq = {}
    word_stance_1_freq = {}

    #单词在sentiment中正负标签出现频率
    word_sentiment_0_freq = {}
    word_sentiment_1_freq = {}


    def __init__(self, dataset, theme):
        self.dataset = dataset
        self.theme = theme
    
    def read_data(self):
        data = pd.read_csv(self.dataset + '.csv')
        texts = []
        with open(self.dataset + '_clean.txt', 'rb') as f:
            for line in f.readlines():
                texts.append(line.strip().decode('latin1'))
        stances = np.array(data['Stance'].astype(str)).tolist()
        sentiments = np.array(data['Sentiment'].astype(str)).tolist()
        
        
        '''sentiment_0_set = set()
        sentiment_1_set = set()
        stance_0_set = set()
        stance_1_set = set()
        words = set()
        for i in range(len(texts)):
            sen = nlp(texts[i])
            sta = stance_hash(stances[i])
            sti = sentiment_hash(sentiments[i])
            for token in sen:
                words.add(token.text)
                if(sta == 0):
                    stance_0_set.add(token.text)
                elif sta == 1:
                    stance_1_set.add(token.text)
                if(sti == 0):
                    sentiment_0_set.add(token.text)
                elif sti == 1:
                    sentiment_1_set.add(token.text)
        
        self.count = len(words)
        self.sentiment_0_count = len(sentiment_0_set)
        self.sentiment_1_count = len(sentiment_1_set)
        self.stance_0_count = len(stance_0_set)
        self.stance_1_count = len(stance_1_set) ''' 
        
        ''' 
        for i in range(len(texts)):
            #word_list = sen.split(' ')
            #for w in word_list:
            #    words.add(w)
            sen = texts[i]
            doc = nlp(sen)
            sta = stance_hash(stances[i])
            sti = sentiment_hash(sentiments[i])
            for token in doc:
                if(sta == 0):
                    if token.text in self.word_stance_0_freq:
                        self.word_stance_0_freq[token.text] += 1
                    else :
                        self.word_stance_0_freq[token.text] = 1
                elif sta == 1:
                    if token.text in self.word_stance_1_freq:
                        self.word_stance_1_freq[token.text] += 1
                    else :
                        self.word_stance_1_freq[token.text] = 1
                if(sti == 0):
                    if token.text in self.word_sentiment_0_freq:
                        self.word_sentiment_0_freq[token.text] += 1
                    else :
                        self.word_sentiment_0_freq[token.text] = 1
                elif sti == 1:
                    if token.text in self.word_sentiment_1_freq:
                        self.word_sentiment_1_freq[token.text] += 1
                    else :
                        self.word_sentiment_1_freq[token.text] = 1''' 
        
        words = set()
        for sen in texts:    
            doc = nlp(sen)
            for token in doc:
                if token.text in self.word_freq:
                    self.word_freq[token.text] += 1
                else :
                    self.word_freq[token.text] = 1
                words.add(token.text)
        self.words_number = len(words)
        for index, word in enumerate(words):
            self.word2id[word] = index
            self.id2word[index] = word
        

    def save_base(self):
        output1 = open(self.theme + '_word2id.pkl', 'wb')
        output2 = open(self.theme + '_id2word.pkl', 'wb')
        output3 = open(self.theme + '_word_freq.pkl', 'wb')
        pk.dump(self.word2id, output1)
        pk.dump(self.id2word, output2)
        pk.dump(self.word_freq, output3)

    def save_label_count(self):
        output = open(self.theme + '_label_count.pkl', 'wb')
        pk.dump({'count':self.count, 'stance_0_count':self.stance_0_count, 'stance_1_count':self.stance_1_count, 'sentiment_0_count':self.sentiment_0_count, 'sentiment_1_count':self.sentiment_1_count}, output)

    def save_label_freq(self):
        output1 = open(self.theme + '_sentiment_0_freq.pkl', 'wb')
        pk.dump(self.word_sentiment_0_freq, output1)
        
        output2 = open(self.theme + '_sentiment_1_freq.pkl', 'wb')
        pk.dump(self.word_sentiment_1_freq, output2)
        
        output3 = open(self.theme + '_stance_0_freq.pkl', 'wb')
        pk.dump(self.word_stance_0_freq, output3)

        output4 = open(self.theme + '_stance_1_freq.pkl', 'wb')
        pk.dump(self.word_stance_1_freq, output4)

train = Corpus('./data/test_set/face_masks_test', './data/test_set/test')
train.read_data()
train.save_base()

