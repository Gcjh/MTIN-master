import sys
import nltk
from nltk.corpus import stopwords
from utils import clean_str
import pandas as pd
import numpy as np

if len(sys.argv) < 2:
    sys.exit("Use: python remove_words.py <dataset>")

dataset = sys.argv[1]
func = clean_str

try:
    least_freq = sys.argv[2]
except:
    least_freq = 2
    print('using default least word frequency = 3')


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#print(stop_words)


doc_content_list = []
data  = pd.read_csv(dataset + '.csv')
lines = np.array(data['Text'].astype(str)).tolist()

for l in lines :
    doc_content_list.append(l.strip())

'''with open('data/corpus/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))'''


word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = func(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = func(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        #if word not in stop_words and word_freq[word] >= least_freq:
        doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)


clean_corpus_str = '\n'.join(clean_docs)
with open(dataset + '_clean.txt', 'w') as f:
    f.write(clean_corpus_str)


len_list = []
with open(dataset + '_clean.txt', 'r') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        temp = line.strip().split()
        len_list.append(len(temp))

print('min_len : ' + str(min(len_list)))
print('max_len : ' + str(max(len_list)))
print('average_len : ' + str(sum(len_list)/len(len_list)))