from __future__ import unicode_literals, print_function, division

import copy
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from model import *
from evaluate import evaluate, evaluate_gaussian

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataLoader import TenseSet, readData, TestSet

"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""

SOS_token = 0
EOS_token = 1

# ----------Hyper Parameters----------#
hidden_size = 256
# The number of vocabulary
vocab_size = 28
teacher_forcing_ratio = 1.0
empty_input_ratio = 0.1
# KLD_weight with higher value giving more structured latent space but poorer reconstruction,
# lower value giving better reconstruction with less structured latent space
# (though their focus is specifically on learning disentangled representations)
KLD_weight = 0.0
LR = 0.01
MAX_LENGTH = 10


def train(input_tensor, target_tensor, types, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(types)
    encoder_cell = encoder.initHidden(types)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # ----------sequence to sequence part for encoder----------#
    hidden_dis, cell_dis = encoder(input_tensor, encoder_hidden, encoder_cell)

    decoder_hidden = decoder.initHidden(reparameter(hidden_dis[0], hidden_dis[1]), types)
    decoder_cell = decoder.initHidden(reparameter(cell_dis[0], cell_dis[1]), types)
    decoder_input = torch.tensor([[SOS_token]], device=device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    kld = KLD_lose(*hidden_dis) + KLD_lose(*cell_dis)
    cross_entropy_lose = 0.
    loss = kld * KLD_weight

    # ----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            cross_entropy_lose += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            # TODO: Survey topk
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            cross_entropy_lose += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss += cross_entropy_lose
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return cross_entropy_lose.item() / target_length, kld, loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def KLD_lose(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    global KLD_weight
    global teacher_forcing_ratio

    # Init variable
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # kld_delta = (0.5 - KLD_weight) / ((n_iters - 20000) // print_every)
    kld_delta = 1 / 5000
    teacher_forcing_delta = (teacher_forcing_ratio - 0.5) / ((n_iters // 2) // print_every)

    # Best record and weight
    best_record = 0
    best_encoder = None
    best_decoder = None

    # Handle dataset
    train_set = TenseSet(readData('data', 'train'))
    test_set = TestSet(readData('data', 'test'))
    pairs = train_set.get_pairs()

    # Init optimizer and loss function
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        # TODO: decrease the KLD_weight and teacher forcing ratio through the epochs
        encoder.train(), decoder.train()
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0].to(device)
        target_tensor = training_pair[1].to(device)
        types = training_pair[2].to(device)

        # Forward pass
        ce_loss, kld_loss, loss = train(input_tensor, target_tensor, types, encoder,
                                  decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # Show the current status
        if iter % print_every == 0:
            candidate, bleu_score = evaluate(encoder, decoder, test_set)
            generated_word, gau_score = evaluate_gaussian(decoder)
            # if bleu_score > best_record:
            if gau_score > best_record:
                best_record = bleu_score
                # best_encoder = copy.deepcopy(encoder.state_dict())
                # best_decoder = copy.deepcopy(decoder.state_dict())

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print(generated_word)
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print(f'cross_entropy: {ce_loss}')
            print(f'KL divergence: {kld_loss}')
            # print(candidate)
            print(f'KLD_weight: {KLD_weight}')
            print(f'teacher forcing ratio: {teacher_forcing_ratio}')
            print(f'Average BLEU-4 score : {bleu_score}')
            print(f'Gaussian score: {gau_score}')
            print('-' * 30)

            # Monotonic kld annealing
            # if iter >= 20000:
            #     KLD_weight += kld_delta
            # Teacher forcing decrease
            # if iter >= n_iters // 2:
            #     teacher_forcing_ratio -= teacher_forcing_delta

        # Cyclical kld annealing
        if iter % 1e4 == 0:
            KLD_weight = 0
        elif KLD_weight < 1:
            KLD_weight += kld_delta

    print('Finish')
    # print(f'Best BLEU-4: {best_record}')
    print(f'Best Gaussian score: {best_record}')
    print('save the model..')
    # encoder.load_state_dict(best_encoder)
    # decoder.load_state_dict(best_decoder)
    torch.save(encoder, 'gau_encoder.pth')
    torch.save(decoder, 'gau_decoder.pth')


if __name__ == "__main__":
    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
    trainIters(encoder1, decoder1, 200000, print_every=1000, learning_rate=LR)
