#/usr/bin/python2
# coding: utf-8

import numpy as np
import codecs
from hyperparams import Hyperparams as hp

def load_vocab():
    hanguls, hanjas = set(), set()
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        for hangul, hanja in zip(hangul_sent, hanja_sent):
            hanguls.add(hangul)
            hanjas.add(hanja)
    hanjas = hanjas - hanguls
    hanguls = ["<EMP>", "<OOV>"] + sorted(list(hanguls))
    hanjas = ["<EMP>", "_"] + sorted(list(hanjas))

    hangul2idx = {hangul: idx for idx, hangul in enumerate(hanguls)}
    idx2hangul = {idx: hangul for idx, hangul in enumerate(hanguls)}

    hanja2idx = {hanja: idx for idx, hanja in enumerate(hanjas)}
    idx2hanja = {idx: hanja for idx, hanja in enumerate(hanjas)}

    return hangul2idx, idx2hangul, hanja2idx, idx2hanja

def load_data(mode="train"):
    hangul2idx, idx2hangul, hanja2idx, idx2hanja = load_vocab()

    # Vectorize
    xs, ys = [], []  # vectorized sentences
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        if len(hangul_sent) <= hp.maxlen:
            x = [hangul2idx.get(hangul, 1) for hangul in hangul_sent]
            y = [hanja2idx.get(hanja, 1) for hanja in hanja_sent]

            x.extend([0] * (hp.maxlen - len(x)))  # zero post-padding
            y.extend([0] * (hp.maxlen - len(y)))  # zero post-padding

            xs.append(x)
            ys.append(y)

    # Convert to 2d-arrays
    X = np.array(xs, np.int32)
    Y = np.array(ys, np.int32)

    if mode=="train":
        X, Y = X[:-hp.batch_size], Y[:-hp.batch_size]
    else: # eval
        X, Y = X[-hp.batch_size:], Y[-hp.batch_size:]

    return X, Y