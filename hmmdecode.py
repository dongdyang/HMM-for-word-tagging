
import sys
import re
from collections import defaultdict, Counter
import numpy
from math import log
from time import clock

STAR = '*'
STOP = 'STOP'
RARE_MAX_FREQ = 1
UNK = "<UNK>"
LOG_ZERO = -100




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


def model_read(model_save_path):
    f = open(model_save_path, 'r')

    #   Parameter 1: word_tag_kind
    word_tag_kind = defaultdict(list)
    n = int(f.readline())
    for i in range(n):
        line = re.split("\t|\n", f.readline())[:-1]
        word_tag_kind[line[0]] = line[1:]

    #   Parameter 2: taglist
    taglist = set()
    n = int(f.readline())
    for i in range(n):
        line = re.split("\t|\n", f.readline())[:-1]
        taglist.add(line[0])

    #   Parameter 3: known_words
    known_words = defaultdict(int)
    n = int(f.readline())
    for i in range(n):
        line = re.split("\t|\n", f.readline())[:-1]
        known_words[line[0]] = 1

    #   Parameter 4: q_values
    q_values = {}
    n = int(f.readline())
    for i in range(n):
        line = re.split("\t|\n", f.readline())[:-1]
        q_values[(line[0], line[1], line[2])] = float(line[3])


    #   Parameter 5: e_values
    e_values = {}
    n = int(f.readline())
    for i in range(n):
        line = re.split("\t|\n", f.readline())[:-1]
        e_values[(line[0], line[1])] = float(line[2])

    f.close()
    return word_tag_kind, taglist, known_words, q_values, e_values


if __name__ == '__main__':
    clock()
    #'./input/en_dev_raw.txt'
    dev_raw_name = sys.argv[1]
    model_save_path = 'hmmmodel.txt'
    dev_raw_tagged_name = "hmmoutput.txt"

    word_tag_kind, taglist, known_words, q_values, e_values = model_read(model_save_path)

    dev_raw = dev_raw_name
    dev_words_t = preprocessDev(dev_raw)
    viterbi_tagged = []

    for tokens in dev_words_t:
        tag_res = tag_viterbi(word_tag_kind, tokens, taglist, known_words, q_values, e_values)
        tag_sentence = ""
        for i in range(len(tag_res)):
            tag_sentence += tokens[i]+"/"+tag_res[i]+" "
        viterbi_tagged.append(tag_sentence[:-1])

    output_dev_tagged(viterbi_tagged, dev_raw_tagged_name)
    print("Dev time: " + str(clock()) + ' sec')

