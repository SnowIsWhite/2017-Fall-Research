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
from preprocess import *
from utils import *

class BaeAttn1Encoder(nn.Module):
    """ word attention """
    def __init__(self, num_embeddings, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use):
        super(BaeAttn1Encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout
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
        self.hidden_size = hidden_size
        self.GPU_use = GPU_use

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
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
                attn_energies[b, i] = self.score(hidden[:, b],
                encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to B 1 S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.mm(hidden, energy.transpose(0,1))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.mm(self.v, energy.transpose(0,1))
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
        self.n_layer = n_layer
        self.dropout = dropout
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.init = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(function, hidden_size, GPU_use)
        self.gru = nn.GRU(embedding_size+(hidden_size*2), hidden_size, n_layer,
        dropout=dropout)

    def forward(self, word_inputs, prev_hidden, encoder_outputs):
        last_hidden = prev_hidden[-1].unsqueeze(0)
        attention_weights = self.attn(last_hidden, encoder_outputs)
        # weights: B, 1, S
        # encoder_outputs: S, B, H*2
        # last_hidden: B, H
        context_vector = attention_weights.bmm(encoder_outputs.transpose(0,1))
        # context_vector: B, 1, H*2
        batch = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        embedded = self.embedding(word_inputs)
        gru_input = torch.cat((embedded, context_vector),2)\
        .view(seq_len, batch, -1)
        output, hidden = self.gru(gru_input, prev_hidden)
        output = output.squeeze()
        # output: B, H
        # hidden: layer, B, H
        return output, hidden, attention_weights

    def initHidden(self, backward_state):
        backward_state = torch.unsqueeze(backward_state, 0)
        hidden = self.init(backward_state)
        hidden = F.tanh(hidden)
        hidden = hidden.repeat(self.n_layer, 1, 1)
        return hidden

class feedforward(nn.Module):
    """ final decision """
    def __init__(self, input_size, hidden_size, num_classes):
        super(feedforward, self).__init__()
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
        self.num_classes = num_classes
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
            for i in range(feat_length):
                eij = self.tanh(self.align(decoder_output[i, b].unsqueeze(0))) #1,1
                e[b,i] = eij.squeeze(0)
        alpha = F.softmax(e).unsqueeze(1) # B, 1, S
        # c
        context_vector = alpha.bmm(decoder_output.transpose(0,1))
        # B, 1, S bmm B, S, H -> B, 1, H
        output = self.out(context_vector.squeeze(1))
        # B, num_classes
        return output

def train(encoder, decoder, final, encoder_optimizer, decoder_optimizer,
final_optimizer, criterion, encoder_input, input_label, target_features,
input_lengths, final_method, GPU_use):
    mini_batch_size = encoder_input.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    final_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(mini_batch_size) # L*2, B, H
    hidden_size = encoder_hidden.size()[2]
    encoder_outputs, encoder_hidden = encoder(encoder_input, input_lengths,
    encoder_hidden)
    backward_state = encoder_outputs[0][:,:-hidden_size]
    prev_hidden = decoder.initHidden(backward_state)
    # prev_hidden: L, B, H

    decoder_input = Variable(torch.LongTensor([0] * mini_batch_size))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    if GPU_use:
        decoder_input = decoder_input.cuda()
    loss = 0
    num_features = target_features.size()[1]
    # decoder
    decoder_outputs = Variable(torch.FloatTensor(num_features, mini_batch_size,
    hidden_size))
    if GPU_use:
        decoder_outputs = decoder_outputs.cuda()

    for di in range(num_features-1):
        target = target_features[:, di]
        decoder_output, decoder_hidden, _ = decoder(decoder_input, prev_hidden,
        encoder_outputs)
        prev_hidden = decoder_hidden
        # target teaching?
        decoder_input = Variable(torch.LongTensor([di+1]*mini_batch_size))
        decoder_input = torch.unsqueeze(decoder_input, 1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        decoder_outputs[di] = decoder_output

    if final_method == 'feedforward':
        decoder_outputs = decoder_outputs.transpose(0,1).contiguous().\
        view(mini_batch_size,-1)
        output = final(decoder_outputs)
    elif final_method == 'attention':
        output = final(decoder_outputs)
    loss = criterion(output, input_label.squeeze())
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    final_optimizer.step()
    return loss.data[0]

def test(encoder, decoder, final, encoder_input, target_features, input_lengths,
final_method, GPU_use):
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

    decoder_outputs = Variable(torch.FloatTensor(num_features, mini_batch_size,
    hidden_size))
    if GPU_use:
        decoder_outputs = decoder_outputs.cuda()

    attention_weights = []
    for di in range(num_features-1):
        target = target_features[:, di]
        decoder_output, decoder_hidden, weights = \
        decoder(decoder_input, prev_hidden, encoder_outputs)
        # weights: B, 1, S
        attention_weights.append(weights.squeeze().cpu().data)
        prev_hidden = decoder_hidden
        decoder_input = Variable(torch.LongTensor([di+1]*mini_batch_size))
        decoder_input = torch.unsqueeze(decoder_input, 1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        decoder_outputs[di] = decoder_output

    if final_method == 'feedforward':
        decoder_outputs = decoder_outputs.transpose(0,1).contiguous().\
        view(mini_batch_size,-1)
        output = final(decoder_outputs)
    elif final_method == 'attention':
        output = final(decoder_outputs)
    softmax = nn.LogSoftmax()
    output = softmax(output)
    encoder.train(True)
    decoder.train(True)
    final.train(True)
    return output.data, attention_weights

if __name__ == "__main__":
    data_name = 'bopang'
    num_features = 4
    function = 'general'
    #function = 'concat'
    final_method = 'feedforward'
    #final_method = 'attention'
    isDependency = False
    isPOS = False
    MAX_LENGTH = 30
    VOCAB_SIZE = 30000
    mini_batch_size = 64
    GPU_use = False

    n_epoch = 10
    n_layer = 1
    embedding_size = 1000
    hidden_size = 1000
    a_hidden_size = 1000
    learning_rate = [0.01, 0.001, 0.0001]
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
    max_learning_rate = learning_rate[0]
    max_acc = 0
    print("Training...")
    for lr in learning_rate:
        # define models
        encoder = BaeAttn1Encoder(lang.n_words, embedding_size, hidden_size,
        mini_batch_size, n_layer, dropout, GPU_use)
        decoder = BaeAttn1Decoder(num_features, embedding_size, mini_batch_size,
        hidden_size, a_hidden_size, n_layer, dropout, function, GPU_use)
        if final_method == 'feedforward':
            final = feedforward(num_features*hidden_size, hidden_size,
            num_classes)
        elif final_method == 'attention':
            final = finalAttention(hidden_size, n_layer, num_classes, GPU_use)
        criterion = nn.CrossEntropyLoss()
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        final_optimizer = torch.optim.Adam(final.parameters(), lr=lr)

        if GPU_use:
            encoder.cuda()
            decoder.cuda()
            fina.cuda()

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
            for i in range(len(train_data)):
                iter_cnt += 1
                input_var = train_data[i]
                input_label = train_label[i]
                input_lengths = train_lengths[i]

                loss = train(encoder, decoder, final, encoder_optimizer,
                decoder_optimizer, final_optimizer, criterion, input_var,
                input_label, target_var, input_lengths, final_method, GPU_use)

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
        showPlot(plot_losses, plot_dir + plot_name + str(lr))

        # validation on learning rate
        acc = 0
        for i in range(len(valid_data)):
            input_var = valid_data[i]
            input_lenghts = valid_lengths[i]
            label = valid_label[i].squeeze().data
            predicted, _ = test(encoder, decoder, final, input_var, target_var,
            input_lengths, final_method, GPU_use)
            # attention weights: array of B,S tensor
            # predicted: B, num_classes tensor
            topv, topi = predicted.topk(1)
            for bi in range(len(topi)):
                if topi[bi][0] == label[bi]:
                    acc += 1
        acc = acc / (len(valid_data) * mini_batch_size *1.)

        if acc > max_acc:
            max_acc = acc
            max_learning_rate = lr

        print("For learning rate " + str(lr) + ", accuracy: " + str(acc))
        break
    print("Training done.")

    print("Test with maximum learning rate.")
    lr = max_learning_rate
    encoder = BaeAttn1Encoder(lang.n_words, embedding_size, hidden_size,
    mini_batch_size, n_layer, dropout, GPU_use)
    decoder = BaeAttn1Decoder(num_features, embedding_size, mini_batch_size,
    hidden_size, a_hidden_size, n_layer, dropout, function, GPU_use)
    if final_method == 'feedforward':
        final = feedforward(num_features*hidden_size, hidden_size,
        num_classes)
    elif final_method == 'attention':
        final = finalAttention(hidden_size, n_layer, num_classes, GPU_use)
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    final_optimizer = torch.optim.Adam(final.parameters(), lr=lr)

    if GPU_use:
        encoder.cuda()
        decoder.cuda()
        fina.cuda()

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
        for i in range(len(train_data)):
            iter_cnt += 1
            input_var = train_data[i]
            input_label = train_label[i]
            input_lengths = train_lengths[i]

            loss = train(encoder, decoder, final, encoder_optimizer,
            decoder_optimizer, final_optimizer, criterion, input_var,
            input_label, target_var, input_lengths, final_method, GPU_use)

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
        target_var, input_lengths, final_method, GPU_use)
        # attn: list of Variable(S,)
        # predicted: 1, num_classes
        topv, topi = predicted.topk(1)
        if topi[0][0] == label[0]:
            acc += 1
            correct = 1
        printAttentions(fname, correct, attn, lang, input_var.data)
    acc = acc / (len(test_data) * 1.)
    print("Test accuracy: " + str(acc))
