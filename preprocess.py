"""Sentence level attention model for classification. Preprocess data."""
"""
Data to use:
1. blogs data: 5,000
2. Pang's Twitter data (binary): 10,000
3. Mohammad's Twitter data (fine-grained): 25,000
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch.autograd import Variable
from read_data import *

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

class Language:
    def __init__(self, name):
        self.name = name
        self.n_words = 4
        self.word2index = {}
        self.index2word = {0: 'PAD', 1: 'UNK', 2: 'SOS', 3:'EOS'}
        self.word2count = {}
        if self.name == 'blogs':
            self.label2index = {'ne': 0, 'hp': 1, 'sd': 2, 'ag': 3, 'dg': 4,
            'sp': 5, 'fr': 6}
        elif self.name == 'twitter':
            self.label2index = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'sadness': 4, 'surprise': 5}
        elif self.name == 'bopang':
            self.label2index = {'neg': 0, 'pos': 1}

    def addSentence(self, tokenized_sentence):
        for word in tokenized_sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def initialize(self):
        self.n_words = 4
        self.word2index = {}
        self.index2word = {0: 'PAD', 1: 'UNK', 2: 'SOS', 3:'EOS'}
        self.word2count = {}

def splitData(data, train=0.8, valid=0.1, test=0.1):
    train_data = {}
    valid_data = {}
    test_data = {}
    for key in data:
        train_portion = int(len(data[key]) * train)
        valid_portion = int(len(data[key]) * valid)
        train_data[key] = []
        valid_data[key] = []
        test_data[key] = []
        train_data[key] = data[key][:train_portion]
        valid_data[key] = data[key][train_portion+1:train_portion+valid_portion]
        test_data[key] = data[key][train_portion + valid_portion + 1:]
    return train_data, valid_data, test_data

def shuffle(data, label):
    zipped = list(zip(data, label))
    random.shuffle(zipped)
    data, label = zip(*zipped)
    return list(data), list(label)

def splitLabels(train, valid, test):
    tr = []
    tr_label = []
    te = []
    te_label = []
    va = []
    va_label = []
    for key in train:
        for item in train[key]:
            tr_label.append(key)
            tr.append(item)
    for key in valid:
        for item in valid[key]:
            va_label.append(key)
            va.append(item)
    for key in test:
        for item in test[key]:
            te_label.append(key)
            te.append(item)
    tr, tr_label = shuffle(tr, tr_label)
    va, va_label = shuffle(va, va_label)
    te, te_label = shuffle(te, te_label)
    return tr, tr_label, va, va_label, te, te_label

def batchData(data, label, mini_batch_size):
    batched_data = []
    batched_label = []
    bd = []
    bl = []
    for idx, item in enumerate(data):
        bd.append(item)
        bl.append(label[idx])
        if ((idx+1) % mini_batch_size) == 0:
            batched_data.append(bd)
            batched_label.append(bl)
            bd = []
            bl = []
    lengths = []
    for idx, item in enumerate(batched_data):
        zipped = list(zip(item, batched_label[idx]))
        sort_by_len = sorted(zipped, key=lambda tup: len(tup[0]), reverse=True)
        a, b= zip(*sort_by_len)
        batched_data[idx] = list(a)
        batched_label[idx] = list(b)
        length = [len(words) for words in batched_data[idx]]
        lengths.append(length)
    return batched_data, batched_label, lengths

def makeDictionary(lang, tr, VOCAB_SIZE):
    for batch in tr:
        for s in batch:
            lang.addSentence(s)
    if lang.n_words <= VOCAB_SIZE + 4:
        return
    words = []
    for idx in lang.index2word:
        if idx < 4:
            continue
        word = lang.index2word[idx]
        cnt = lang.word2count[word]
        words.append((word,cnt))
    sort_by_cnt = sorted(words, key=lambda tup: tup[1], reverse=True)
    lang.initialize()

    for idx, tup in enumerate(sort_by_cnt):
        if idx >= VOCAB_SIZE:
            break
        lang.addWord(tup[0])

def phraseToIndex(phrase, lang):
    result = []
    for word in phrase:
        if word not in lang.word2index:
            result.append(UNK_token)
        else:
            result.append(lang.word2index[word])
    return result

def labelToIndex(label, lang):
    return [lang.label2index[lbl] for lbl in label]

def trainDataToVariable(data, label, lang, GPU_use):
    for idx, batch in enumerate(data):
        indicies = [phraseToIndex(phrase, lang) for phrase in batch]
        longest_seq_len = len(indicies[0])
        tensor = torch.LongTensor(len(indicies), longest_seq_len)
        for idx2, phrase in enumerate(indicies):
            while len(phrase) < longest_seq_len:
                phrase.append(PAD_token)
            tensor[idx2] = torch.LongTensor(phrase)
        data_var = Variable(tensor)
        label_var = Variable(torch.LongTensor([labelToIndex(label[idx], lang)]))
        if GPU_use:
            data_var = data_var.cuda()
            label_var = label_var.cuda()
        data[idx] = data_var
        label[idx] = label_var
    return data, label

def testDataToVariable(data, label, lang, GPU_use):
    for idx, phrase in enumerate(data):
        data_var = Variable(torch.LongTensor(phraseToIndex(phrase, lang)))
        label_var = Variable(torch.LongTensor([lang.label2index[label[idx]]]))
        if GPU_use:
            data_var = data_var.cuda()
            label_var = label_var.cuda()
        data[idx] = data_var
        label[idx] = label_var
    return data, label

def readData(data_name, isDependency, isPOS, MAX_LENGTH):
    data_dir = './json_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fname = data_dir + '/' + data_name + str(MAX_LENGTH)
    if isDependency == True:
        fname = fname + 'dep'
    if isPOS == True:
        fname = fname + 'pos'
    fname = fname + '.json'

    if os.path.exists(fname):
        with open(fname, 'r') as jsonfile:
            json_data = json.load(jsonfile)
    else:
        json_data = readDataFromFile(fname, data_name, isDependency, isPOS,
        MAX_LENGTH)
    return json_data

def prepareData(data_name, isDependency=False, isPOS=False, MAX_LENGTH=30,
VOCAB_SIZE=30000, mini_batch_size=64, GPU_use=False):
    # read data
    """Data: {label: [[tokens]]}"""
    print("Reading data...")
    data = readData(data_name, isDependency, isPOS, MAX_LENGTH)
    # split into train/valid/test
    tr, va, te = splitData(data, train=0.8, valid=0.1, test=0.1)
    tr, tr_label, va, va_label, te, te_label = splitLabels(tr, va, te)
    # make batches
    tr, tr_label, tr_lengths = batchData(tr, tr_label, mini_batch_size)
    va, va_label, va_lengths = batchData(va, va_label, mini_batch_size)
    lang = Language(data_name)
    print("Make dictionary...")
    makeDictionary(lang, tr, VOCAB_SIZE)
    tr, tr_label = trainDataToVariable(tr, tr_label, lang, GPU_use)
    va, va_label = trainDataToVariable(va, va_label, lang, GPU_use)
    te, te_label = testDataToVariable(te, te_label, lang, GPU_use)
    return lang, tr, tr_label, tr_lengths, va, va_label, va_lengths, \
    te, te_label

if __name__ == "__main__":
    prepareData('blogs')
