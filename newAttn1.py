import os
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
    hidden_size, n_layer, dropout, GPU_use):
        super(BaeAttn1Encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.GPU_use = GPU_use

        self.embeding = nn.Embedding(num_embeddings, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer, dropout=dropout,
        bidirectional=True)

    def forward(self, input_variable, input_lengths, hidden):
        mini_batch_size = input_variable.size()[0]
        embedded = self.embedding(input_variable.transpose(0,1))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return rnn_output, hidden

    def initHidden(self, mini_batch_size):
        hidden = Variable(torch.zeros(self.n_layer*2, mini_batch_size,
        self.hidden_size))
        if self.GPU_use:
            hidden = hidden.cuda()
        return hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size, GPU_use):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.GPU_use = GPU_use

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if self.GPU_use:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to B 1 S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class BaeAttn1Decoder(nn.Module):
    """ feature attention """
    def __init__(self, num_embeddings, embedding_size, mini_batch_size,
    hidden_size, a_hidden_size, n_layer, dropout, function, GPU_use):
        super(BaeAttn1Decoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.init = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(function, hidden_size, GPU_use)
        self.gru = nn.GRU(embedding_size+(hidden_size*2), hidden_size, n_layer,
        dropout=dropout)

    def forward(self, word_inputs, prev_hidden, encoder_outputs):
        last_hidden = prev_hidden[-1]

        attention_weights = self.attn(last_hidden, encoder_outputs)
        # weights: B, 1, S
        # encoder_outputs: S, B, H*2
        context_vector = attention_weights.bmm(encoder_outputs.transpose(0,1))
        # context_vector: B, 1, H*2
        batch = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        embedded = self.embedding(word_inputs)
        gru_input = torch.cat((embedded, context_vector),2)
        output, hidden = self.gru(gru_input, prev_hidden)
        # output: S, B, H
        # hidden: layer, B, H
        return output, hidden, attention_weights

    def initHidden(self, backward_state):
        backward_state = torch.unsqueeze(backward_state, 0)
        hidden = self.init(backward_state)
        hidden = F.tanh(hidden)
        hidden = hidden.repeat(self.n_layer, 1, 1)
        return hidden

class feedfoward(nn.Module):
    """ final decision """
    def __init__(self, input_size, hidden_size, num_classes):
        super(feedfoward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size/2)
        self.fc3 = nn.Linear(hidden_size/2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.lrelu(out)
        out = self.fc3(out)
        return out

class finalAttention(nn.Module):
    """ Further attention on decoder results """
    def __init__(self, hidden_size, n_layer, num_classes, GPU_use):
        super(finalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.num_classes
        self.GPU_use = GPU_use

        self.align = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, decoder_output):
        # attention
        feat_length = decoder_output.size()[0]
        mini_batch_size = decoder_output.size()[1]
        e = Variable(torch.zeros(mini_batch_size, feat_length)) # B, S
        if GPU_use:
            e = e.cuda()

        for b in range(mini_batch_size):
            for i in range(annot_length):
                eij = self.tanh(self.align(decoder_output[i, b].unsqueeze(0))) #1,1
                e[b,i] = eij.squeeze(0)
        alpha = F.softmax(e).unsqueeze(1) # B, 1, S
        # c
        context_vector = alpha.bmm(decoder_output.transpose(0,1))
        # B, 1, S bmm B, S, H -> B, 1, H
        output = self.out(context_vector.squeeze(1))
        return output

def train(encoder, decoder, final, encoder_optimizer, decoder_optimizer,
final_optimizer, criterion, encoder_input, target_features, GPU_use):
    mini_batch_size = encoder_input.size()[0]

def test():
    pass

if __name__ == "__main__":
    data_name = 'bopang'
    num_features = 4
    function = 'general'
    final_layer = 'feedforward'
    isDependency = False
    isPOS = False
    MAX_LENGTH = 30
    VOCAB_SIZE = 30000
    mini_batch_size = 64
    GPU_use = False

    n_epoch = 10
    n_layer = 1
    embedding_size = 1000
    hiddden_size = 1000
    a_hidden_size = 1000
    learning_rate = [0.01, 0.001, 0.0001]
    dropout = 0.5
    print_every = mini_batch_size * 10

    lang, train_data, train_label, train_lengths, valid_data, valid_label, \
    test_data, test_label = prepareData(data_name, isDependency, isPOS,
    MAX_LENGTH, VOCAB_SIZE, mini_batch_size, GPU_use)
    print("Data Preparation Done.")

    # define models
    encoder = BaeAttn1Encoder(lang.n_words, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use)
    decoder = BaeAttn1Decoder(num_features, embedding_size, mini_batch_size,
    hidden_size, a_hidden_size, n_layer, dropout, function, GPU_use)
    if final_layer == 'feedforward':
        final = feedforward(num_features, num_features*hidden_size, num_classes)
    elif final_layer == 'attn':
        final = finalAttention(hidden_size, n_layer, num_classes, GPU_use)

    criterion = nn.CrossEntropyLoss()
    for lr in learning_rate:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        final_optimizer = torch.optim.Adam(final.parameters(), lr=lr)

        if GPU_use:
            encoder.cuda()
            decoder.cuda()
            fina.cuda()
        print("Training...")

        start = time.time()
        plot_losses = []
        plot_loss_total = 0
        print_loss_total = 0
        total_iter = len(train_data) * mini_batch_size * n_epoch * 1.
        iter_cnt = 0
        for epoch in range(n_epoch):
            for i in range(len(train_data)):
                iter_cnt += 1
                
