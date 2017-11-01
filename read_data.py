import os
import sys
import json
import nltk
import re
import unicodedata
from nltk.parse.stanford import StanfordDependencyParser
from stanfordcorenlp import StanfordCoreNLP
reload(sys)
sys.setdefaultencoding('UTF8')

def printStatistics(dictionary):
    print("===Statistics:")
    total = 0
    for key in dictionary:
        print('===== ' + key + ': ' + str(len(dictionary[key])))
        total += len(dictionary[key])
    print('===== total size: ' + str(total))

# Turn a Unicode string to plain ASCII,
# thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def filterText(text):
    text = (text.lower().strip())
    text = text.replace('-', '')
    text = text.replace('_', '')
    text = re.sub(r"([.:;)(<>!?\"\'])", r"\1", text)
    text = text.strip()
    return text

def getDependencyRelations(json_data, temp_data, parser):
    for key in temp_data:
        for s in temp_data[key]:
            sentence = ' '.join(s)
            result = parser.raw_parse(sentence)
            dep = result.next()
            parsed_list = list(dep.triples())
            parsed_sentence = []
            for (a, b, c) in parsed_list:
                parsed_sentence.append(a[0])
                parsed_sentence.append(a[1])
                parsed_sentence.append(b)
                parsed_sentence.append(c[0])
                parsed_sentence.append(c[1])
            json_data[key].append(parsed_sentence)
    return json_data

def getPOStags(json_data, temp_data):
    for key in temp_data:
        for s in temp_data[key]:
            sentence = ' '.join(s)
            parsed = nltk.pos_tag(sentence)
            parsed_sentence = []
            for tup in parsed:
                parsed_sentence.append(tup[0])
                parsed_sentence.append(tup[1])
            json_data[key].append(parsed_sentence)
    return json_data

def readBopangData(data_dir, isDependency, isPOS, MAX_LENGTH, tokenizer, parser):
    json_data = {'neg': [], 'pos': []}
    with open(data_dir + 'rt-polarity.pos', 'r') as f:
        pos_lines = f.readlines()
    with open(data_dir + 'rt-polarity.neg', 'r') as f:
        neg_lines = f.readlines()

    # filter text
    for idx, line in enumerate(pos_lines):
        pos_lines[idx] = filterText(line)
    for idx, line in enumerate(neg_lines):
        neg_lines[idx] = filterText(line)

    temp_data = {'neg': [], 'pos': []}
    for line in pos_lines:
        tokenized_sentence = tokenizer.word_tokenize(line)
        if len(tokenized_sentence) <5 or len(tokenized_sentence) > MAX_LENGTH:
            continue
        temp_data['pos'].append(tokenized_sentence)
    for line in neg_lines:
        tokenized_sentence = tokenizer.word_tokenize(line)
        if len(tokenized_sentence) <5 or len(tokenized_sentence) > MAX_LENGTH:
            continue
        temp_data['neg'].append(tokenized_sentence)
    printStatistics(temp_data)

    if isDependency == True:
        json_data = getDependencyRelations(json_data, temp_data, parser)
    elif isPOS == True:
        json_data = getPOStags(json_data, temp_data)
    else:
        json_data = temp_data
    return json_data

def readTwitterData(data_dir, isDependency, isPOS, MAX_LENGTH, tokenizer, parser):
    json_data = {'anger': [], 'disgust': [], 'fear': [], 'joy': [],
    'sadness': [], 'surprise': []}
    temp_data = {'anger': [], 'disgust': [], 'fear': [], 'joy': [],
    'sadness': [], 'surprise': []}

    with open(data_dir, 'r') as f:
        sentences = f.readlines
    for s in sentences:
        phrase = s.split('\t')
        label = phrase[2][3:].strip()
        if tag not in temp_data:
            continue
        sentence = phrase[1]
        filtered_sentence = filterText(sentence)
        tokenized_sentence = tokenizer.word_tokenize(filtered_sentence)
        if len(tokenized_sentence) < 5 or len(tokenized_sentence) > MAX_LENGTH:
            continue
        temp_data[label].append(tokenized_sentence)
    printStatistics(temp_data)
    if isDependency == True:
        json_data = getDependencyRelations(json_data, temp_data, parser)
    elif isPOS == True:
        json_data = getPOStags(json_data, temp_data)
    else:
        json_data = temp_data
    return json_data

def readBlogsData(data_dir, isDependency, isPOS, MAX_LENGTH, tokenizer, parser):
    json_data = {'ne': [], 'hp': [], 'sd': [], 'ag': [], 'dg': [], 'sp': [],
    'fr': []}
    temp_data = {'ne': [], 'hp': [], 'sd': [], 'ag': [], 'dg': [], 'sp': [],
    'fr': []}

    with open(data_dir, 'r') as f:
        sentences = f.readlines()
    for s in sentences:
        words = s.split()
        sentence = ' '.join(words[2:])
        label = words[0]
        filtered_sentence = filterText(sentence)
        tokenized_sentence = tokenizer.word_tokenize(filtered_sentence)
        if len(tokenized_sentence) < 5 or len(tokenized_sentence) > MAX_LENGTH:
            continue
        temp_data[label].append(tokenized_sentence)
    printStatistics(temp_data)
    if isDependency == True:
        json_data = getDependencyRelations(json_data, temp_data, parser)
    elif isPOS == True:
        json_data = getPOStags(json_data, temp_data)
    else:
        json_data = temp_data
    return json_data

def readDataFromFile(fname, data_name, isDependency, isPOS, MAX_LENGTH):
    while True:
        try:
            stanford_tokenizer = StanfordCoreNLP('/Users/jaeickbae/Documents/\
projects/utils/stanford-corenlp-full-2017-06-09')
            break
        except:
            print("Waiting for Stanford Core NLP Server Response...")
    path_to_jar = '/Users/jaeickbae/Documents/projects/\
utils/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
    path_to_models_jar = '/Users/jaeickbae/Documents/projects/\
utils/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar,
    path_to_models_jar=path_to_models_jar)

    if os.path.exists(fname):
        with open(fname, 'r') as jsonfile:
            json_data = json.load(jsonfile)
    else:
        if data_name == 'bopang':
            data_dir = '/Users/jaeickbae/Documents/projects/data/\
bopang_twitter/rt-polaritydata/'
            json_data = readBopangData(data_dir, isDependency, isPOS,
            MAX_LENGTH, stanford_tokenizer, dependency_parser)
        elif data_name == 'twitter':
            data_dir = '/Users/jaeickbae/Documents/projects/data/\
Mohammad_twitter.txt'
            json_data = readTwitterData(data_dir, isDependency, isPOS,
            MAX_LENGTH, stanford_tokenizer, dependency_parser)
        elif data_name == 'blogs':
            data_dir = '/Users/jaeickbae/Documents/projects/data/blogs/\
Benchmark/category_gold_std.txt'
            json_data = readBlogsData(data_dir, isDependency, isPOS, MAX_LENGTH,
            stanford_tokenizer, dependency_parser)
        with open(fname, 'w') as f:
            json.dump(json_data, f)
    return json_data
