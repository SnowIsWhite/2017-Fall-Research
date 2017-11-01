"""Basic attention model on classification."""
""" batched bidirectional gru """
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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class basicAttn(nn.Module):
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

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer, dropout=dropout,
        bidirectional=True)
        self.align = nn.Linear(hidden_size*2, 1)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size*2, label_size)

    def forward(self, input_variable, input_lengths, hidden):
        mini_batch_size = input_variable.size()[0]
        embedded = self.embedding(input_variable.transpose(0,1))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        # output: seq_len, B, H*2
        # hidden: layer*2, B, H
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # attention
        annot_length = rnn_output.size()[0]
        e = Variable(torch.zeros(mini_batch_size, annot_length)) # B, S
        if GPU_use:
            e = e.cuda()

        for b in range(mini_batch_size):
            for i in range(annot_length):
                eij = self.tanh(self.align(rnn_output[i, b].unsqueeze(0))) #1,1
                e[b,i] = eij.squeeze(0)
        alpha = F.softmax(e).unsqueeze(1) # B, 1, S
        # c
        context_vector = alpha.bmm(rnn_output.transpose(0,1))
        # B, 1, S bmm B, S, H*2 -> B, 1, H*2
        output = self.out(context_vector.squeeze(1))

        return output, hidden, alpha

    def initHidden(self, mini_batch_size):
        hidden = Variable(torch.zeros(self.n_layer*2, mini_batch_size,
        self.hidden_size))
        if self.GPU_use:
            hidden = hidden.cuda()
        return hidden

def train(model, optimizer, criterion, input_var, label, length, GPU_use):
    optimizer.zero_grad()
    mini_batch_size = input_var.size()[0]
    init_hidden = model.initHidden(mini_batch_size)
    output, hidden, alpha = model(input_var, length, init_hidden)
    # output : B, label_size
    loss = criterion(output, label.squeeze())
    loss.backward()
    optimizer.step()
    return loss.data[0]

def test(model, input_sentence):
    model.train(False)
    mini_batch_size = 1
    length = [input_sentence.size()[0]]
    init_hidden = model.initHidden(mini_batch_size)
    output, hidden, alpha = model(input_sentence.unsqueeze(0), length,
    init_hidden)
    softmax = nn.LogSoftmax()
    output = softmax(output).squeeze()
    topv, topi = output.data.topk(1)
    return topi, alpha

if __name__ == "__main__":
    data_name = 'bopang'
    isDependency = False
    isPOS = False
    MAX_LENGTH = 30
    VOCAB_SIZE = 30000
    mini_batch_size = 64
    GPU_use = False

    n_epoch = 10
    n_layer = 1
    embedding_size = 1000
    hidden_size = 7
    learning_rate = 0.0001
    dropout = 0.1
    print_every = mini_batch_size * 10
    plot_every = mini_batch_size * 5

    lang, train_data, train_label, train_lengths, valid_data, valid_label, \
    test_data, test_label = prepareData(data_name, isDependency, isPOS,
    MAX_LENGTH, VOCAB_SIZE, mini_batch_size, GPU_use)
    print("Data Preparation Done.")
    model = basicAttn(lang.n_words, embedding_size, mini_batch_size,
    hidden_size, len(lang.label2index), n_layer, dropout, GPU_use)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if GPU_use:
        model.cuda()

    print("Training...")

    # train
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    total_iter = len(train_data) * mini_batch_size * n_epoch * 1.
    iter_cnt = 0
    for epoch in range(n_epoch):
        for i in range(len(train_data)):
            iter_cnt += mini_batch_size
            input_var = train_data[i]
            label = train_label[i]
            length = train_lengths[i]
            loss = train(model, optimizer, criterion, input_var, label, length,
            GPU_use)
            print_loss_total += loss
            plot_loss_total += loss

            if iter_cnt % print_every == 0:
                print_loss_avg = print_loss_total / print_every*1.
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % ((timeSince(start,iter_cnt/total_iter)),
                iter_cnt, iter_cnt/total_iter * 100, print_loss_avg))

            if iter_cnt % plot_every == 0:
                plot_loss_avg = plot_loss_total / (plot_every*1.)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            break
        break
    showPlot(plot_losses, 'batched_vanilaRNN')
    print("Training done.")

    # save model
    torch.save(model.state_dict(), './basic_attention.pkl')
    print("Model Saved.")

    # test
    predicted = []
    attention_weights = []
    for target_var in test_data:
        label, alpha = test(model, target_var)
        predicted.append(label[0])
        attention_weights.append(alpha.data)

    # print accuracy, confusion_matrix
    cnt = 0
    for idx, p in enumerate(predicted):
        if p == test_label[idx].data[0]:
            cnt += 1
    print("acc: "+ str(cnt*1./len(predicted)))
    # print attention weights
