import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, fig_name):
    fig_name = fig_name + '.png'
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig(fig_name)

def printAttentions(fname, correct, attn, lang, input_tensor):
    sentence = ' '.join([lang.index2word[idx] for idx in input_tensor])
    with open(fname, 'a') as f:
        f.write(sentence + '\n')
        if correct == 1:
            f.write('1\n')
        else:
            f.write('0\n')

        for i in range(len(attn)):
            temp = [val for val in attn[i]]
            temp2 = [val for val in input_tensor]
            zipped = list(zip(temp, temp2))
            sorted_zip = sorted(zipped, key=lambda tup: tup[0], reverse=True)
            attn_words = [lang.index2word[tup[1]] for tup in sorted_zip]
            for word in attn_words:
                f.write(word + "\t")
            f.write('\n')
