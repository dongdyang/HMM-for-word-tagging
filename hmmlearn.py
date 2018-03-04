
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


def model_save(save_path, word_tag_kind, taglist, known_words, q_values, e_values):
    f = open(save_path, 'w')
    '''
    f.write(str(word_tag_kind)+"\n")
    f.write(str(taglist) + "\n")
    f.write(str(known_words) + "\n")
    f.write(str(q_values) + "\n")
    f.write(str(e_values) + "\n")
    '''
    #   Parameter 1
    n = len(word_tag_kind)
    f.write(str(n)+"\n")
    for key in word_tag_kind:
        f.write(str(key)+"\t"+"\t".join(word_tag_kind[key])+"\n")

    #   Parameter 2
    n = len(taglist)
    f.write(str(n) + "\n")
    for key in taglist:
        f.write(str(key)+"\n")

    #   Parameter 3
    n = len(known_words)
    f.write(str(n) + "\n")
    for key in known_words:
        f.write(str(key) + "\n")

    #   Parameter 4
    n = len(q_values)
    f.write(str(n) + "\n")
    for key in q_values:
        f.write("\t".join(key)+"\t"+str(q_values[key])+"\n")

    #   Parameter 5
    n = len(e_values)
    f.write(str(n) + "\n")
    for key in e_values:
        f.write("\t".join(key)+"\t"+str(e_values[key])+"\n")

    f.close()


if __name__ == '__main__':
    #'./input/en_train_tagged.txt'

    train_file_name = sys.argv[1]
    model_save_path = 'hmmmodel.txt'

    clock()
    train_tagged = train_file_name
    words, tags = preprocess(train_tagged)
    word_tag_kind, taglist, known_words, q_values, e_values = train(words, tags)

    model_save(model_save_path, word_tag_kind, taglist, known_words, q_values, e_values)

    print("Trained time: " + str(clock()) + ' sec')




