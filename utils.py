import numpy as np
import scipy.sparse as sp
import re
from tqdm import tqdm
import torch
from sklearn import metrics
# import sparse

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()).toarray()

def get_acc_from_logits(logits, labels):
    soft_probs = torch.argmax(logits, -1)
    #print(soft_probs)
    #print(labels)
    #input()
    acc = (soft_probs == labels).float().mean()
    return acc, soft_probs, labels

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_loss, count =  0, 0
    mean_acc0, mean_acc1 = 0, 0
    n_pre0, n_labels0 = None, None
    n_pre1, n_labels1 = None, None
    with torch.no_grad():
        for input_ids, attention_mask, types, labels, graphs, target_len in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, types= input_ids.to(device), attention_mask.to(device), types.to(device)
            labels, graphs = labels.to(device), graphs.to(device)
            
            labels = labels.permute(1,0)
            graphs = graphs.permute(1,0,2,3)
            inputs = torch.stack((input_ids, input_ids, input_ids))
            masks = torch.stack((attention_mask, attention_mask, attention_mask))

            logits0, logits1, ls, lh = model(inputs, masks, graphs, target_len)
            
            mean_loss += 0.8 * criterion(logits0, labels[0]) + 0.7 * criterion(logits1, labels[1]) + 0.0005 * (ls+lh)
            
            
            k0, y0, l0 = get_acc_from_logits(logits0, labels[0])
            k1, y1, l1 = get_acc_from_logits(logits1, labels[1])
            mean_acc0 += k0
            mean_acc1 += k1
            if(n_pre0 == None):
                n_pre0 = y0
                n_labels0 = l0
            else :
                n_pre0 = torch.cat((n_pre0, y0), dim=0)
                n_labels0 = torch.cat((n_labels0, l0), dim=0)
            if(n_pre1 == None):
                n_pre1 = y1
                n_labels1 = l1
            else :
                n_pre1 = torch.cat((n_pre1, y1), dim=0)
                n_labels1 = torch.cat((n_labels1, l1), dim=0)
            count += 1
    f10 = metrics.f1_score(n_labels0.cpu(), n_pre0.cpu(), labels=[0, 1],average='macro')
    mf10 = metrics.f1_score(n_labels0.cpu(), n_pre0.cpu(), labels=[0, 1],average='micro')
    f11 = metrics.f1_score(n_labels1.cpu(), n_pre1.cpu(), labels=[0, 1],average='macro')
    mf11 = metrics.f1_score(n_labels1.cpu(), n_pre1.cpu(), labels=[0, 1],average='micro')
    
    favor = metrics.f1_score(n_labels0.cpu(), n_pre0.cpu(), labels=[0],average='macro')						
    against = metrics.f1_score(n_labels0.cpu(), n_pre0.cpu(), labels=[1],average='macro')
    return mean_acc0 / count, mean_acc1 / count ,mean_loss / count, f10, mf10, f11, mf11, favor, against


def sentiment_hash(x):
    if x == 'Positive' or x == 'pos':
        return 0
    elif x == 'Negative' or x == 'neg':
        return 1
    else:
        return 2

def stance_hash(x):
    if x == 'FAVOR':
        return 0
    elif x == 'AGAINST':
        return 1
    else:
        return 2

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

