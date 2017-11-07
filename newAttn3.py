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

class Attn2Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use):
        super(Attn2Encoder, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
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
        self.GPU_use = GPU_use

        if method == 'general':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size*3, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))
        # B, S
        if self.GPU_use:
            attn_energies = attn_energies.cuda()

        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b,i] = self.score(hidden[:,b],
                encoder_outputs[i,b].unsqueeze(0))
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.mm(hidden, energy.transpose(0,1))
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output),1))
            energy = torch.mm(self.v, energy.transpose(0,1))
            return energy

class Attn2Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, mini_batch_size,
    hidden_size, a_hidden_size, n_layer, dropout, function, GPU_use):
        super(Attn2Decoder, self).__init__()
        self.n_layer = n_layer

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.init = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(function, hidden_size, GPU_use)
        self.gru = nn.GRU(embedding_size+(hidden_size*2), hidden_size, n_layer,
        dropout=dropout)

    def forward(self, word_inputs, prev_hidden, encoder_outputs):
        # weights: B, 1, S
        # encoder_outputs: S, B, H*2
        # last_hidden: 1, B, H
        last_hidden = prev_hidden[-1].unsqueeze(0)
        attention_weights = self.attn(last_hidden, encoder_outputs)
        context_vector = attention_weights.bmm(encoder_outputs.transpose(0,1))
        # context_vector : B, 1, H*2
        batch = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        embedded = self.embedding(word_inputs)
        gru_input = torch.cat((embedded, context_vector), 2)\
        .view(seq_len, batch, -1)
        output, hidden = self.gru(gru_input, last_hidden)
        output = output.squeeze(0)
        # output: B, H
        # hidden: layer, B, H
        return output, hidden, attention_weights, context_vector

    def initHidden(self, backward_state):
        hidden = self.init(backward_state).unsqueeze(0)
        hidden = F.tanh(hidden)
        hidden = hidden.repeat(self.n_layer, 1, 1)
        return hidden

class finalOut(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(finalOut, self).__init__()
        self.fc1 = nn.Linear(num_features*hidden_size*2, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size/2)
        self.fc4 = nn.Linear(hidden_size/2, hidden_size/4)
        self.fc5 = nn.Linear(hidden_size/4, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, context_vectors):
        mini_batch_size = context_vectors.size()[0]
        context_vectors = context_vectors.view(mini_batch_size, -1)
        # context_vectors: B, S, H*2
        out = self.fc1(context_vectors)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = self.fc5(out)
        return out

def train(encoder, decoder, final, encoder_optimizer, decoder_optimizer,
final_optimizer, criterion, encoder_input, input_label, target_features,
input_lengths, GPU_use):
    clip = 0.5
    mini_batch_size = encoder_input.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    final_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(mini_batch_size)
    hidden_size = encoder_hidden.size()[2]
    encoder_outputs, encoder_hidden = encoder(encoder_input, input_lengths,
    encoder_hidden)
    backward_state = encoder_outputs[0][:,:-hidden_size]
    prev_hidden = decoder.initHidden(backward_state)
    # prev_hidden: L, B, H

    decoder_input = Variable(torch.LongTensor([0]*mini_batch_size))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    if GPU_use:
        decoder_input = decoder_input.cuda()
    loss = 0
    num_features = target_features.size()[1]

    # context_vectors: B, S, H*2
    context_vectors = Variable(torch.FloatTensor(mini_batch_size, num_features,
    hidden_size*2))
    if GPU_use:
        context_vectors = context_vectors.cuda()

    context_vectors = context_vectors.transpose(0,1)
    for di in range(num_features-1):
        target = target_features[:, di]
        decoder_output, decoder_hidden, _, context_vector = \
        decoder(decoder_input, prev_hidden, encoder_outputs)
        prev_hidden = decoder_hidden
        # target teaching?
        decoder_input = Variable(torch.LongTensor([di+1]*mini_batch_size))
        decoder_input = torch.unsqueeze(decoder_input, 1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        context_vectors[di] = context_vector.squeeze()
    context_vectors = context_vectors.transpose(0,1)

    output = final(context_vectors)
    loss = criterion(output, input_label.squeeze())
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(final.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    final_optimizer.step()
    return loss.data[0]

def test(encoder, decoder, final, encoder_input, target_features, input_lengths,
GPU_use):
    encoder.train(False)
    decoder.train(False)
    final.train(False)
    mini_batch_size = encoder_input.size()[0]
    encoder_hidden = encoder.initHidden(mini_batch_size)
    hidden_size = encoder_hidden.size()[2]
    encoder_outputs, encoder_hidden = encoder(encoder_input, input_lengths,
    encoder_hidden)
    backward_state = encoder_outputs[0][:,:-hidden_size]
    prev_hidden = decoder.initHidden(backward_state)

    decoder_input = Variable(torch.LongTensor([0] * mini_batch_size))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    if GPU_use:
        decoder_input = decoder_input.cuda()

    # context_vectors: B, S, H*2
    context_vectors = Variable(torch.FloatTensor(mini_batch_size, num_features,
    hidden_size*2))
    if GPU_use:
        context_vectors = context_vectors.cuda()

    context_vectors = context_vectors.transpose(0,1)
    attention_weights = []
    for di in range(num_features-1):
        target = target_features[:, di]
        decoder_output, decoder_hidden, weights, context_vector = \
        decoder(decoder_input, prev_hidden, encoder_outputs)
        # weights: B, 1, S
        attention_weights.append(weights.squeeze().cpu().data)
        prev_hidden = decoder_hidden
        decoder_input = Variable(torch.LongTensor([di+1]*mini_batch_size))
        decoder_input = torch.unsqueeze(decoder_input, 1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        context_vectors[di] = context_vector.squeeze()
    context_vectors = context_vectors.transpose(0,1)
    output = final(context_vectors)
    output = F.softmax(output)
    encoder.train(True)
    decoder.train(True)
    final.train(True)
    return output.data, attention_weights

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    data_name = 'blogs'
    num_features = 4
    #function = 'general'
    function = 'concat'
    isDependency = False
    isPOS = False
    MAX_LENGTH = 30
    VOCAB_SIZE = 30000
    mini_batch_size = 64
    GPU_use = True

    n_epoch = 14
    n_layer = 1
    embedding_size = 500
    hidden_size = 500
    a_hidden_size = 500
    learning_rate = 1e-4
    dropout = 0.5
    print_every = mini_batch_size * 10
    plot_every = mini_batch_size
    plot_dir = './plots/'
    plot_name = 'newAttentionBlogs'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    lang, train_data, train_label, train_lengths, valid_data, valid_label, \
    valid_lengths, test_data, test_label = prepareData(data_name, isDependency,
    isPOS, MAX_LENGTH, VOCAB_SIZE, mini_batch_size, GPU_use)
    print("Data Preparation Done.")

    num_classes = len(lang.label2index)

    # define models
    encoder = Attn2Encoder(lang.n_words, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use)

    decoder = Attn2Decoder(num_features, embedding_size, mini_batch_size,
    hidden_size, a_hidden_size, n_layer, dropout, function, GPU_use)

    final = finalOut(num_features, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    final_optimizer = torch.optim.Adam(final.parameters(), lr=learning_rate)
    if GPU_use:
        encoder.cuda()
        decoder.cuda()
        final.cuda()

    start = time.time()
    plot_losses = []
    plot_loss_total = 0
    print_loss_total = 0
    total_iter = len(train_data) * mini_batch_size * n_epoch * 1.
    iter_cnt = 0

    target_var = []
    for j in range(num_features):
        target_var.append(j+1)
    target_var = Variable(torch.LongTensor(target_var).unsqueeze(0)\
    .repeat(mini_batch_size,1))
    if GPU_use:
        target_var = target_var.cuda()

    for epoch in range(n_epoch):
        adjust_learning_rate(encoder_optimizer, epoch, learning_rate)
        adjust_learning_rate(decoder_optimizer, epoch, learning_rate)
        adjust_learning_rate(final_optimizer, epoch, learning_rate)
        for i in range(len(train_data)):
            iter_cnt += mini_batch_size
            input_var = train_data[i]
            input_label = train_label[i]
            input_lengths = train_lengths[i]

            loss = train(encoder, decoder, final, encoder_optimizer,
            decoder_optimizer, final_optimizer, criterion, input_var,
            input_label, target_var, input_lengths, GPU_use)

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

    # test
    # test data: list of long tensors
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
        predicted, attn = test(encoder, decoder, final, input_var.unsqueeze(0),
        target_var, input_lengths, GPU_use)
        # attn: list of Variable(S,)
        # predicted: num_classes
        topv, topi = predicted.topk(1)
        if topi[0][0] == label[0]:
            acc += 1
            correct = 1
        printAttentions(fname, correct, attn, lang, input_var.data)
    acc = acc / (len(test_data) * 1.)
    print("Test accuracy: " + str(acc))
