#coding:utf-8

import pickle as pk
from numpy.lib.polynomial import roots
import pandas as pd
import numpy as np
import spacy
import math

nlp = spacy.load('en_core_web_sm')

target_text = 'face_masks'
theme = 'test'

with open('./data/' + theme + '_set/' + theme + '_label_count.pkl', 'rb') as f:
    count_list = pk.load(f)

with open('./data/' + theme + '_set/' + theme + '_word_freq.pkl', 'rb') as f:
    word_freq_list = pk.load(f)

with open('./data/' + theme + '_set/' + theme + '_stance_0_freq.pkl', 'rb') as f:
    stance_0_list = pk.load(f)

with open('./data/' + theme + '_set/' + theme + '_stance_1_freq.pkl', 'rb') as f:
    stance_1_list = pk.load(f)

with open('./data/' + theme + '_set/' + theme + '_sentiment_0_freq.pkl', 'rb') as f:
    sentiment_0_list = pk.load(f)

with open('./data/' + theme + '_set/' + theme + '_sentiment_1_freq.pkl', 'rb') as f:
    sentiment_1_list = pk.load(f)


'''def get_dependenceTree(x):
    doc = nlp(x)
    x = x.split()
    text_len = len(x)
    target_len = len(target_text.split())
    seq_len = text_len + target_len
    tree = np.zeros((seq_len, seq_len))
    for chunk in doc.noun_chunks:
        root_word = x.index(chunk.root.text) + target_len
        tree[root_word][0] = 1
        tree[0][root_word] = 1
    for token in doc:
        for child in token.children:
            if child.i < seq_len:
                tree[token.i + target_len][child.i + target_len] = 1
                tree[child.i + target_len][token.i + target_len] = 1
    return tree'''
    

def get_tf(w):
    return word_freq_list[w] / count_list['count']

def get_cross_cr_stance(w):
    if w in stance_0_list:
        label0 = stance_0_list[w] / count_list['stance_0_count']
    else :
        label0 = 0
    if w in stance_1_list :
        label1 = stance_1_list[w] / count_list['stance_1_count']
    else:
        label1 = 0
    return abs(label0 - label1)

def get_cross_cr_sentiment(w):
    if w in sentiment_0_list:
        label0 = sentiment_0_list[w] / count_list['sentiment_0_count']
    else:
        label0 = 0
    if w in sentiment_1_list:
        label1 = sentiment_1_list[w] / count_list['sentiment_1_count']
    else:
        label1 = 0
    return abs(label0 - label1)

def jaccard(x, y):
    a = np.array(x)
    b = np.array(y)
    c1 = np.dot(a, b)
    c2 = (np.dot(a, a) + np.dot(b, b) - np.dot(a, b))
    if c2 == 0:
        return 0
    else:
        return c1 / c2

def get_TF_CR(w):
    stance = []
    sentiment = []
    if w in stance_0_list:
        label0 = stance_0_list[w] ** 2 / count_list['stance_0_count'] * word_freq_list[w]
        stance.append(label0)
    else :
        label0 = 0
        stance.append(label0)
    if w in stance_1_list:
        label1 = stance_1_list[w] ** 2 / count_list['stance_1_count'] * word_freq_list[w]
        stance.append(label1)
    else:
        label1 = 0
        stance.append(label1)
    if w in sentiment_0_list:
        label2 = sentiment_0_list[w] ** 2 / count_list['sentiment_0_count'] * word_freq_list[w]
        sentiment.append(label2)
    else:
        label2 = 0
        sentiment.append(label2)
    if w in sentiment_1_list:
        label3 = sentiment_1_list[w] ** 2 / count_list['sentiment_1_count'] * word_freq_list[w]
        sentiment.append(label3)
    else:
        label3 = 0
        sentiment.append(label3)
    return jaccard(stance, sentiment)

def normalization(x):
    s = np.array(x)
    return ((s - np.mean(s)) / np.std(s)).tolist()

def get_len(x):
    len = 0
    for token in x:
        len += 1
    return len

def get_index(x, k):
    for token in x:
        if(token.text == k):
            return token.i

def get_stance_adj(x):
    
    doc = nlp(x)
    text_len = get_len(doc)

    target_len = len(target_text.split())
    seq_len = text_len + target_len

    adj = np.zeros((seq_len, seq_len))
    
    for chunk in doc.noun_chunks:
        if(chunk.root.text in x):
            root_index = get_index(doc, chunk.root.text) + target_len
            for i in range(target_len):
                adj[root_index][i] = 1 + word_tf_dict[chunk.root.text] * word_cr_stance_dict[chunk.root.text]
                adj[i][root_index] = 1 + word_tf_dict[chunk.root.text] * word_cr_stance_dict[chunk.root.text]

    for token in doc:
        for child in token.children:
            if child.i + target_len < seq_len:
                adj[token.i + target_len][child.i + target_len] = word_tf_dict[token.text] * word_cr_stance_dict[token.text] + word_tf_dict[child.text] * word_cr_stance_dict[child.text]
                adj[child.i + target_len][token.i + target_len] = word_tf_dict[token.text] * word_cr_stance_dict[token.text] + word_tf_dict[child.text] * word_cr_stance_dict[child.text]
    return adj


def get_sentiment_adj(x):
    
    doc = nlp(x)
    text_len = get_len(doc)

    target_len = len(target_text.split())
    seq_len = text_len + target_len

    adj = np.zeros((seq_len, seq_len))
    
    for chunk in doc.noun_chunks:
        if(chunk.root.text in x):
            root_index = get_index(doc, chunk.root.text) + target_len
            for i in range(target_len):
                adj[root_index][i] = 1 + word_tf_dict[chunk.root.text] * word_cr_sentiment_dict[chunk.root.text]
                adj[i][root_index] = 1 + word_tf_dict[chunk.root.text] * word_cr_sentiment_dict[chunk.root.text]

    for token in doc:
        for child in token.children:
            if child.i + target_len < seq_len:
                adj[token.i + target_len][child.i + target_len] = word_tf_dict[token.text] * word_cr_sentiment_dict[token.text] + word_tf_dict[child.text] * word_cr_sentiment_dict[child.text]
                adj[child.i + target_len][token.i + target_len] = word_tf_dict[token.text] * word_cr_sentiment_dict[token.text] + word_tf_dict[child.text] * word_cr_sentiment_dict[child.text]
    return adj

def get_interaction_adj(x):
    
    doc = nlp(x)
    text_len = get_len(doc)

    target_len = len(target_text.split())
    seq_len = text_len + target_len

    adj = np.zeros((seq_len, seq_len))
    
    for chunk in doc.noun_chunks:
        if(chunk.root.text in x):
            root_index = get_index(doc, chunk.root.text) + target_len
            for i in range(target_len):
                adj[root_index][i] = 1 + word_interaction_dict[chunk.root.text]
                adj[i][root_index] = 1 + word_interaction_dict[chunk.root.text]

    for token in doc:
        for child in token.children:
            if child.i + target_len < seq_len:
                adj[token.i + target_len][child.i + target_len] = word_interaction_dict[token.text] + word_interaction_dict[child.text]
                adj[child.i + target_len][token.i + target_len] = word_interaction_dict[token.text] + word_interaction_dict[child.text]
    return adj



texts = []
with open('./data/'+theme+'_set/face_masks_' +theme+'_clean.txt', 'rb') as f:
    for line in f.readlines():
        texts.append(line.strip().decode('latin1'))

words_set = set()
for text in texts:
    doc = nlp(text)
    for token in doc:
        words_set.add(token.text)
    #dependence_matrix = get_dependenceTree(text)
    #print(dependence_matrix)
    #input()

word_tf_list = []
word_cr_stance_list = []
word_cr_sentiment_list = []
word_interaction_list = []

word_tf_dict = {}
word_cr_stance_dict = {}
word_cr_sentiment_dict = {}
word_interaction_dict = {}

for i, v in enumerate(words_set):
    word_tf_list.append(get_tf(v))
    word_cr_sentiment_list.append(get_cross_cr_sentiment(v))
    word_cr_stance_list.append(get_cross_cr_stance(v))
    word_interaction_list.append(get_TF_CR(v))


word_tf_list = normalization(word_tf_list)

word_cr_sentiment_list = normalization(word_cr_sentiment_list)
word_cr_stance_list = normalization(word_cr_stance_list)
word_interaction_list = normalization(word_interaction_list)

for i, v in enumerate(words_set):
    word_tf_dict[v] = word_tf_list[i]
    word_cr_stance_dict[v] = word_cr_stance_list[i]
    word_cr_sentiment_dict[v] = word_cr_sentiment_list[i]
    word_interaction_dict[v] = word_interaction_list[i]

id2graph_stance = {}
id2graph_sentiment = {}
id2graph_interaction = {}

for i in range(len(texts)):
    doc = texts[i]
    id2graph_stance[i] = get_stance_adj(doc)
    id2graph_sentiment[i] = get_sentiment_adj(doc)
    id2graph_interaction[i] = get_interaction_adj(doc)

with open('./data/'+theme+'_set/sentiment_graph.pkl', 'wb') as f:
    pk.dump(id2graph_sentiment, f)

with open('./data/'+theme+'_set/stance_graph.pkl', 'wb') as f:
    pk.dump(id2graph_stance, f)

with open('./data/'+theme+'_set/interaction_graph.pkl', 'wb') as f:
    pk.dump(id2graph_interaction, f)

print('succeed!')

