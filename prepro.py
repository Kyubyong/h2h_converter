#/usr/bin/python2
# coding: utf-8
'''
Preprocessing.
Note:
You need to create your own parallel Hangul-Hanja corpus first and
put it in `corpus/corpus.tsv`.
Each line must look like `Hangul sentence[Tab]Hanja sentence`.

For example,
나는 오늘 학교에 간다    나는 오늘 學校에 간다

This file should create the following files from the corpus.
`data/X.npy`: vectorized hangul sentences
`data/Y.npy`: vectorized hanja sentences
`data/charmaps.pkl`: 4 python dictionaries of character-index collections.
'''

import numpy as np
import cPickle as pickle
import codecs
import re

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 16
    embed_dim = 200
    maxlen = 100
    hidden_dim = 200
    
def prepro():
    '''Embeds and vectorize words in corpus'''

    hangul_sents = [line.split('\t')[0] for line in codecs.open('corpus/corpus.tsv', 'r', 'utf-8').read().splitlines()]
    hanja_sents = [line.split('\t')[1] for line in codecs.open('corpus/corpus.tsv', 'r', 'utf-8').read().splitlines()]
        
    print "# Create Vocabulary sets"
    hangul_vocab, hanja_vocab = set(), set()
    
    for hangul_sent, hanja_sent in zip(hangul_sents, hanja_sents):
        assert len(hangul_sent) == len(hangul_sent), \
                    "Hangul sentence and hanja sentence must be the same in length."

        for char1, char2 in zip(hangul_sent, hanja_sent):
            if re.search(u'[\u4E00-\u9FFF]', char2) is not None: # if char2 is hanja
                hanja_vocab.add(char2)
                if re.search(u'[\uAC00-\uD7AF]', char1) is not None: # if char1 is not hangul
                    hangul_vocab.add(char1)
    
    print "# Create character maps"   
    hangul_vocab = ["<EMP>", "<OOV>"] + list(hangul_vocab) # <EMP> for zero-padding. <OOV> for non-conversion hangul char.
    hanja_vocab   = ["<EMP>", "<SAME>"] + list(hanja_vocab) # <EMP> for zero-padding. <SAME> for non-conversion.
    
    hangul2idx = {hangul:idx for idx, hangul in enumerate(hangul_vocab)}
    idx2hangul = {idx:hangul for idx, hangul in enumerate(hangul_vocab)}
    
    hanja2idx = {hanja:idx for idx, hanja in enumerate(hanja_vocab)}
    idx2hanja = {idx:hanja for idx, hanja in enumerate(hanja_vocab)}
    
    print "hangul vocabulary size is", len(hangul2idx)
    print "hanja vocabulary size is", len(hanja2idx)
    
    pickle.dump((hangul2idx, idx2hangul, hanja2idx, idx2hanja), open('data/charmaps.pkl', 'wb'))
    
    print "# Vectorize"
    xs, ys = [], [] # vectorized sentences
    for hangul_sent, hanja_sent in zip(hangul_sents, hanja_sents):
        if len(hangul_sent) <= Hyperparams.maxlen:
            x, y = [], []
            for char in hangul_sent:
                if char in hangul2idx:
                    x.append(hangul2idx[char])
                else:
                    x.append(1) #"OOV", i.e., not converted.
            
            for char in hanja_sent:
                if char in hanja2idx:
                    y.append(hanja2idx[char])
                else:
                    y.append(1) #"<SAME>", i.e., not converted.
            
            x.extend([0] * (Hyperparams.maxlen - len(x))) # zero post-padding
            y.extend([0] * (Hyperparams.maxlen - len(y))) # zero post-padding
            
            xs.append(x) 
            ys.append(y) 
 
    print "# Convert to 2d-arrays"    
    X = np.array(xs)
    Y = np.array(ys)
    
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be matrices, got {}, {}".format(X.shape, Y.shape)
 
    np.save('data/X', X)
    np.save('data/Y', Y)
             
def load_charmaps():
    '''Loads character dictionaries'''
    hangul2idx, idx2hangul, hanja2idx, idx2hanja = pickle.load(open('data/charmaps.pkl', 'rb'))
    return hangul2idx, idx2hangul, hanja2idx, idx2hanja

def load_data():
    '''Loads vectorized input training data
    '''
    return np.load('data/X.npy'), np.load('data/Y.npy')

if __name__ == '__main__':
    prepro()
    print "Done"        
