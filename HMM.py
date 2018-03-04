
import sys
import re
from collections import defaultdict, Counter
import numpy
from math import log
from time import clock
import threading


STAR = '*'
STOP = 'STOP'
RARE_MAX_FREQ = 1
UNK = "<UNK>"
LOG_ZERO = -100

def preprocess(train_tagged):
    words = []
    tags = []
    f = open(train_tagged, "r")
    while True:
        sentence = f.readline()
        word = []
        tag = []
        if not sentence:
            break
        sentence_split = sentence.split()
        word_tag = []
        for pair in sentence_split:
            temp = re.split(r'/', pair)
            if len(temp) == 2:
                word_tag.append((temp[0].strip(), temp[1].strip()))
            elif len(temp) >= 3:
                word_tag.append(('/'.join(temp[:-1]).strip(), temp[-1].strip()))
            else:
                print("Split Error")
        word_tag = [[STAR] * 2] * 2 + word_tag + [[STOP] * 2]
        for ele in word_tag:
            if len(ele) == 2:
                word.append(ele[0])
                tag.append(ele[1])
        words.append(word)
        tags.append(tag)
    f.close()
    return words, tags

def nltk_bigrams(sentence_tags):
    n = len(sentence_tags)-1
    return [(sentence_tags[i], sentence_tags[i+1]) for i in range(n)]

def nltk_trigrams(sentence_tags):
    n = len(sentence_tags) - 2
    return [(sentence_tags[i], sentence_tags[i+1], sentence_tags[i+2]) for i in range(n)]


def train(words, tags):
    trigrams = [trigram for sentence in tags \
                for trigram in nltk_trigrams(sentence)]
    bigrams = [bigrams for sentence in tags \
               for bigrams in nltk_bigrams(sentence)]

    trigrams_c = Counter(trigrams)
    bigram_c = Counter(bigrams)
    q_values = {trigram: log(count, 2) - log(bigram_c[trigram[:-1]], 2) \
                for trigram, count in trigrams_c.items()}

    words_count = Counter([word for sentence in words for word in sentence])
    known_words = set([word for word, count in words_count.items() \
                       if count > RARE_MAX_FREQ])
    words_rare = [[word in known_words and word or UNK for word in sentence]
                  for sentence in words]

    tags_flat = [tag for sentence in tags for tag in sentence]
    words_flat = [word for sentence in words_rare for word in sentence]
    tags_c = Counter(tags_flat)
    word_tag = zip(words_flat, tags_flat)
    word_tag_c = Counter(word_tag)

    word_tag_kind = defaultdict(list)
    for wt in word_tag_c:
        if wt[1] not in word_tag_kind[wt[0]]:
            word_tag_kind[wt[0]].append(wt[1])

    e_values = {k: log(float(c), 2) - log(float(tags_c[k[1]]), 2) \
                for k, c in word_tag_c.items()}
    taglist = set(tags_flat)
    return word_tag_kind, taglist, known_words, q_values, e_values

def preprocessDev(dev_raw):
    infile = open(dev_raw, "r")
    data = infile.readlines()
    infile.close()
    dev_words = []
    for sentence in data:
        words = sentence.split()
        temp = []
        for w in words:
            temp.append(w)
        dev_words.append(temp)
    return dev_words


def tag_viterbi(word_tag_kind, token0, tags, known_words, Q, E):
    tokens = [STAR]*2 + token0 + [STOP]

    def get_possible_tags(k):
        if k < 2:
            return [STAR]
        else:
            return word_tag_kind[tokens[k]]

    n = len(tokens)
    for k in range(2, n - 1):
        if tokens[k] not in known_words:
            tokens[k] = UNK

    P, path = {}, {}
    P[0, '*', '*'] = 0
    P[1, '*', '*'] = 0
    path['*', '*'] = []
    for k in range(2, n-1):
        word = tokens[k]
        temp_path = {}

        for u in get_possible_tags(k-1):
            for v in get_possible_tags(k):
                P[k,u,v],prev_w = max([(P[k - 1, w, u] + Q.get((w,u,v),LOG_ZERO) + E.get((word,v),LOG_ZERO), w) \
                                           for w in get_possible_tags(k - 2)])
                temp_path[u, v] = path[prev_w, u] + [v]
        path = temp_path

    if n > 4:
        prob, umax, vmax = max([(P[n-2, u, v] + Q.get((u,v,STOP),LOG_ZERO),u,v) \
                                for u in get_possible_tags(n-3) for v in get_possible_tags(n-2)])
    else:
        prob, umax, vmax = max([(P[n-2, STAR, v] + Q.get((STAR,v,STOP),LOG_ZERO),STAR,v) \
                                for v in get_possible_tags(n-2)])
    res = path[umax, vmax]
    return res


def output_dev_tagged(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence + '\n')
    outfile.close()


def correct(filename1, filename2):
    user_sentences = open(filename1, "r").readlines()
    correct_sentences = open(filename2, "r").readlines()
    num_correct = 0
    total = 0
    for user_sent, correct_sent in zip(user_sentences, correct_sentences):
        user_tok = user_sent.split()
        correct_tok = correct_sent.split()
        for u, c in zip(user_tok, correct_tok):
            if u == c:
                num_correct += 1
            total += 1
    score = float(num_correct) / total * 100
    print("{:d}% of tags are correct".format(int(score)))


if __name__ == '__main__':

    train_file_name = './input/en_train_tagged.txt'
    dev_raw_name = './input/en_dev_raw.txt'
    dev_raw_tagged_name = './input/en_dev_raw_tagged.txt'
    dev_tagged_name = './input/en_dev_tagged.txt'


    clock()
    train_tagged = train_file_name
    words, tags = preprocess(train_tagged)
    word_tag_kind, taglist, known_words, q_values, e_values = train(words, tags)
    print("Trained time: " + str(clock()) + ' sec')

    dev_raw = dev_raw_name
    dev_words_t = preprocessDev(dev_raw)
    viterbi_tagged = []

    for tokens in dev_words_t:
        tag_res = tag_viterbi(word_tag_kind, tokens, taglist, known_words, q_values, e_values)
        tag_sentence = ""
        for i in range(len(tag_res)):
            tag_sentence += tokens[i]+"/"+tag_res[i]+" "
        #print tag_sentence
        viterbi_tagged.append(tag_sentence[:-1])

    output_dev_tagged(viterbi_tagged, dev_raw_tagged_name)
    print("Dev time: " + str(clock()) + ' sec')

    correct(dev_raw_tagged_name, dev_tagged_name)
    print("Done: " + str(clock()) + ' sec')


