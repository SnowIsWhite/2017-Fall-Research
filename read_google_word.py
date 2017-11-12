import json
import numpy as np
import gensim
import xlrd
import time
from utils import asMinutes
from read_data import *

google_dir = '/Users/jaeickbae/Documents/projects/data/\
GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(google_dir, binary=True)
vocab = model.vocab.keys()

def getGoogleEmbedding(dataname):
    with open('json_data/'+dataname+'.json', 'r') as f:
        data = json.load(f)
    total_words = 0
    for key in data:
        for line in data[key]:
            total_words += len(line)

    start = time.time()
    cnt = 0
    embedding_list = {}
    for key in data:
        for line in data[key]:
            for word in line:
                cnt += 1
                if cnt % 1000 == 0:
                    now = time.time()
                    print(str(cnt)+' words done. '+str(cnt/(total_words*1.))+'%')
                    print(asMinutes(now-start)+' passed.')
                if word in embedding_list:
                    continue
                if word not in vocab:
                    continue
                embedding_list[word] = model[word].tolist()
    with open('json_data/'+dataname+'_word2vec.json', 'w') as f:
        json.dump(embedding_list, f)

def getFeatureEmbedding(featureset):
    embedding_list = {}
    for key in featureset:
        print(key)
        if key not in embedding_list:
            embedding_list[key] = []
        for word in featureset[key]:
            if word not in vocab:
                continue
            embedding_list[key].append((word, model[word].tolist()))
    with open('json_data/lexicon_word2vec.json', 'w') as f:
        json.dump(embedding_list, f)

def emotionFeatureSet():
    featureset = {'Positive': [],
                'Negative': [],
                'anger': [],
                'anticipation': [],
                'disgust': [],
                'fear': [],
                'joy': [],
                'sadness': [],
                'surprise': [],
                'trust': [],
                'Pleasur': [],
                'Pain': [],
                'Feel': [],
                'Arousal': [],
                'EMOT': [],
                'Virtue': [],
                'Vice': [],
                'Yes': [],
                'No': [],
                'Negate': [],
                'Intrj': [],
                'AffGain': [],
                'AffLoss': [],
                'AffOth': [],
                'AffTot': []}
    with open('./json_data/emolex.json', 'r') as f:
        emolex = json.load(f)
    with open('./json_data/gi.json', 'r') as f:
        gi = json.load(f)

    for key in emolex:
        if key == 'positive':
            featureset['Positive'] = emolex[key]
        elif key == 'negative':
            featureset['Negative'] = emolex[key]
        else:
            featureset[key] = emolex[key]
    for key in gi:
        if key == 'Positiv':
            for word in gi[key]:
                if word not in featureset['Positive']:
                    featureset['Positive'].append(word)
        elif key == 'Negativ':
            for word in gi[key]:
                if word not in featureset['Negative']:
                    featureset['Negative'].append(word)
        else:
            if key in featureset:
                featureset[key] = gi[key]
    with open('json_data/emotion_featureset.json', 'w') as f:
        json.dump(featureset, f)

def readEmolex(emolex):
    emolex_data = {}
    with open(emolex, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            if tokens[1] not in emolex_data:
                emolex_data[tokens[1]] = []
            if tokens[2] == '1':
                emolex_data[tokens[1]].append(tokens[0].lower())
    fname = 'json_data/emolex.json'
    with open(fname, 'w') as jsonfile:
        json.dump(emolex_data, jsonfile)

def readGeneralInquirer(gi):
    gi_data = {}
    book = xlrd.open_workbook(gi)
    first_sheet = book.sheet_by_index(0)
    num_rows = first_sheet.nrows
    first_row = first_sheet.row_values(0)
    gi_tag = [tag for tag in first_row]
    curr_row = 2
    for tag in gi_tag:
        if tag not in gi_data:
            gi_data[tag] = []

    while curr_row < num_rows:
        row = first_sheet.row_values(curr_row)
        word = row[0]
        if len(str(word)) == 0:
            break
        if word == 0:
            word = 'false'
        elif word == 1:
            word = 'true'
        word = ''.join(re.findall('[a-z|A-Z]', word)).lower()
        for idx, tag in enumerate(gi_tag):
            if len(str(row[idx])) != 0:
                if word not in gi_data[tag]:
                    gi_data[tag].append(word)
        curr_row += 1
    fname = 'json_data/gi.json'
    with open(fname, 'w') as jf:
        json.dump(gi_data, jf)

def dictolist(dataname):
    data_dir = 'json_data/'+dataname+'30_word2vec.json'
    data_list = []
    with open(data_dir, 'r') as f:
        data = json.load(f)
    for key in data:
        data_list.append((key, data[key]))
    with open('json_data/'+dataname+'30_veclist.json', 'w') as f:
        json.dump(data_list, f)

if __name__ == "__main__":
    #getGoogleEmbedding('twitter30')
    #emolex_dir = '/Users/jaeickbae/Documents/projects/data/Lexicons/emolex.txt'
    #gi_dir = '/Users/jaeickbae/Documents/projects/data/Lexicons/general_inquirer.xls'
    #readEmolex(emolex_dir)
    #readGeneralInquirer(gi_dir)
    # emotionFeatureSet()
    """
    with open('json_data/emotion_featureset.json', 'r') as f:
        featureset = json.load(f)
    getFeatureEmbedding(featureset)
    """
    dictolist('blogs')
