import os
import sys
import json
import time
import pickle
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from preprocess import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Attn(nn.Module):
    def __init__(self, hidden_size, GPU_use):
        super(Attn, self).__init__()
        self.GPU_use = GPU_use

        self.score_attn = nn.Linear(hidden_size*2, hidden_size)
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(hidden_size, 1))

    def forward(self, num_features, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        attn_energies = Variable(torch.zeros(this_batch_size, max_len,
        num_features))
        if self.GPU_use:
            attn_energies = attn_energies.cuda()

        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b,i] = self.score(num_features,
                encoder_outputs[i,b].unsqueeze(0))
        # attn_energies : B, S, F
        return F.softmax(attn_energies).transpose(1,2)

    def score(self, n_feat, encoder_output):
        # encoder_output : 1, H*2
        encoder_output = encoder_output.repeat(n_feat, 1)
        # encoder_output : n_feat, H*2
        energy = self.score_attn(encoder_output)
        # energy : n_feat, H
        energy = self.tanh(energy)
        energy = torch.mm(energy, self.v)
        # energy: n_feat, 1
        return energy.squeeze()


class Attn5Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use):
        super(Attn5Encoder, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer, dropout=dropout,
        bidirectional=True)
        self.attn = Attn(hidden_size, GPU_use)

    def forward(self, input_variable, input_lengths, hidden, num_features):
        mini_batch_size = input_variable.size()[0]
        embedded = self.embedding(input_variable.transpose(0,1))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # rnn_output: S, B, H*2
        attention_weights = self.attn(num_features, rnn_output)

        # attention_weights: B, F, S
        context_vector = attention_weights.bmm(rnn_output.transpose(0,1))
        # context_vector : B, F, H*2
        return context_vector, attention_weights

    def initHidden(self, mini_batch_size):
        hidden = Variable(torch.zeros(self.n_layer*2, mini_batch_size,
        self.hidden_size))
        if self.GPU_use:
            hidden = hidden.cuda()
        return hidden

class CNN(nn.Module):
    def __init__(self, hidden_size, num_features, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU())
        self.fc = nn.Linear((hidden_size*2/4)*(num_features/4)*32, num_classes)

    def forward(self, feature_map):
        # feature_map : B, F, H*2
        out = torch.unsqueeze(feature_map, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train(encoder, cnn, encoder_optimizer, decoder_optimizer, criterion,
input_var, input_label, input_lengths, num_features, GPU_use):
    clip = 0.5
    mini_batch_size = input_var.size()[0]
    encoder_optimizer.zero_grad()
    cnn_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(mini_batch_size)
    hidden_size = encoder_hidden.size()[2]
    cv, aw = encoder(input_var, input_lengths, encoder_hidden, num_features)
    # cv: B, F, H*2
    # aw: B, F, S
    output = cnn(cv)
    loss = criterion(output, input_label.squeeze())
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(cnn.parameters(), clip)
    encoder_optimizer.step()
    cnn_optimizer.step()
    return loss.data[0]

def test(encoder, cnn, input_var, input_lengths, num_features, GPU_use):
    mini_batch_size = input_var.size()[0]
    encoder_hidden = encoder.initHidden(mini_batch_size)
    hidden_size = encoder_hidden.size()[2]
    cv, aw = encoder(input_var, input_lengths, encoder_hidden, num_features)
    output = cnn(cv)
    output = F.softmax(output)
    return output.data, aw

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    data_name = 'bopang'
    num_features = 8
    isDependency = False
    isPOS = False
    MAX_LENGTH = 30
    VOCAB_SIZE = 30000
    mini_batch_size = 64
    GPU_use = False

    n_epoch = 15
    n_layer = 1
    embedding_size = 300
    hidden_size = 100
    learning_rate = 1e-4
    dropout = 0.1
    print_every = mini_batch_size * 10
    plot_every = mini_batch_size
    plot_dir = './plots/'
    plot_name = 'new5Bopang1'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    lang, train_data, train_label, train_lengths, valid_data, valid_label,\
    valid_lengths, test_data, test_label = prepareData(data_name, isDependency,
    isPOS, MAX_LENGTH, VOCAB_SIZE, mini_batch_size, GPU_use)
    print("Data Preparation Done.")

    num_classes = len(lang.label2index)

    # define models
    encoder = Attn5Encoder(lang.n_words, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use)
    cnn = CNN(hidden_size, num_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    if GPU_use:
        encoder.cuda()
        cnn.cuda()

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    total_iter = len(train_data) * mini_batch_size * n_epoch * 1.
    iter_cnt = 0

    for epoch in range(n_epoch):
        adjust_learning_rate(encoder_optimizer, epoch, learning_rate)
        adjust_learning_rate(cnn_optimizer, epoch, learning_rate)
        for i in range(len(train_data)):
            iter_cnt += mini_batch_size
            input_var = train_data[i]
            input_label = train_label[i]
            input_lengths = train_lengths[i]

            loss = train(encoder, cnn, encoder_optimizer, cnn_optimizer,
            criterion, input_var, input_label, input_lengths, num_features,
            GPU_use)

            if math.isnan(loss):
                print("Loss NaN")
                continue
            print_loss_total += loss
            plot_loss_total += loss
            if iter_cnt % print_every == 0:
                print_loss_avg = print_loss_total / print_every*1.
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' %
                ((timeSince(start,iter_cnt/total_iter)),
                iter_cnt, iter_cnt/total_iter * 100, print_loss_avg))
            if iter_cnt % plot_every == 0:
                plot_loss_avg = plot_loss_total / (plot_every*1.)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            break
        break
    showPlot(plot_losses, plot_dir + plot_name)
    print("Training done.")

    encoder.eval()
    cnn.eval()

    # test
    if not os.path.exists('./attn_results'):
        os.makedirs('./attn_results')
    fname = './attn_results/' + plot_name + '.txt'
    open(fname, 'w').close()
    acc = 0
    for i in range(len(test_data)):
        correct = 0
        input_var = test_data[i]
        input_lengths = [input_var.size()[0]]
        label = test_label[i].data
        predicted, attn = test(encoder, cnn, input_var.unsqueeze(0),
        input_lengths, num_features, GPU_use)
        # attn: B, F, S
        topv, topi = predicted.topk(1)
        if topi[0][0] == label[0]:
            acc += 1
            correct = 1
        attn = attn.transpose(0,1).data
        printAttentions5(fname, correct, attn, lang, input_var.data, label[0])
    acc = acc / (len(test_data)*1.)
    print("Test accuracy : " + str(acc))