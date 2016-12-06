# -*- coding: utf-8 -*-
'''
Run.

You are to write test sentences line by line in `data/input.txt`.
This file writes the results to `data/output.txt` 
'''

import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_charmaps
from train import ModelGraph
import codecs

def vectorize_input():
    '''Embeds and vectorize words in input corpus'''
    try:
        with codecs.open('data/input.txt', 'r', 'utf-8') as fin:
            sents = [line for line in fin.read().splitlines()]
    except IOError:
        raise IOError("Write the sentences you want to test line by line in `data/input.txt` file.")
    
    hangul2idx = load_charmaps()[0]
    
    xs = []
    for sent in sents:
        x = []
        for char in sent[:Hyperparams.maxlen]:
            if char in hangul2idx: 
                x.append(hangul2idx[char])
            else:
                x.append(1) #"OOV", i.e., not converted.
        
        x.extend([0] * (Hyperparams.maxlen - len(x))) # zero post-padding
        xs.append(x)
    
    X = np.array(xs)
    return sents, X # original sents, 2-D array.
                    
def main():  
    g = ModelGraph(is_train=False)
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
        
        # Or you could use pretrained model which can be downloaded from here
        # https://drive.google.com/open?id=0B5M-ed49qMsDQ1dEYXF3cTVNM1E
#         saver.restore(sess, 'model-019-1239684')
                     
        sents, X = vectorize_input()
        idx2hanja = load_charmaps()[-1]
        
        with codecs.open('data/output.txt', 'w', 'utf-8') as fout:
            for step in range(len(X) // Hyperparams.batch_size + 1):
                inputs = sents[step*Hyperparams.batch_size: (step+1)*Hyperparams.batch_size]  # batch
                x = X[step*Hyperparams.batch_size: (step+1)*Hyperparams.batch_size]  # batch
                
                # predict characters
                logits = sess.run(g.logits, {g.x: x})
                preds = np.squeeze(np.argmax(logits, -1))
                for ii, xx, pp in zip(inputs, x, preds): # sentence-wise
                    got = ''
                    for ii, xxx, ppp in zip(ii, xx, pp): # character-wise
                        if xxx == 0: break
                        elif xxx == 1 or ppp == 1:
                            got += ii
                        else: 
                            got += idx2hanja.get(ppp, "*")
                        
                    fout.write(got + "\n")
                                        
if __name__ == '__main__':
    main()
    print "Done"

