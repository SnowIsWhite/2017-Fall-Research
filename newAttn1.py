import sys
import json
import time
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from masked_cross_entropy import *
from preprocess import *
from utils import *

class BaeAttn1Encoder(nn.Module):
    """ word attention """
    def __init__(self, num_embeddings, embedding_size, mini_batch_size,
    hidden_size, label_size, n_layer, dropout, GPU_use):
        super(basicAttn, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.GPU_use = GPU_use

    def forward(self):
        pass
    def initHidden(self):
        pass

class BaeAttn1Decoder(nn.Module):
    """ feature attention """
    def __init__(self, num_embeddings, embedding_size, mini_batch_size,
    hidden_size, label_size, n_layer, dropout, GPU_use):
        super(basicAttn, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.GPU_use = GPU_use

    def forward(self):
        pass
    def score(self):
        pass
    def initHidden(self):
        pass

class feedfoward(nn.Module):
    """ final decision """
    def __init__(self):
        pass
    def foward(self):
        pass

class finalAttention(nn.Module):
    """ Further attention on decoder results """
    def __init__(self):
        pass
    def forward(self):
        pass
    def initHidden
