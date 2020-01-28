"""Data Processing"""
import unicodedata
import re
import random
from config import *

SOS_token = 0   # start of sequence
EOS_token = 1   # end of sequence

MAX_LENGTH = 10     # trim the data set to only short and simple sentences (maximum words including ending punctuation)
Config.max_length = MAX_LENGTH

eng_prefixes = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "they are", "they re"
)


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2ix = {}
        self.word2cnt = {}
        self.ix2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2    # count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2ix:
            self.word2ix[word] = self.n_words
            self.word2cnt[word] = 1
            self.ix2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2cnt[word] += 1


# Turning a unicode string to plain ASCII
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, remove non-letter characters
def normalizeString(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    return s


# Read language
def readLangs(lang1, lang2, reverse=False):
    print('Reading lines...')

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(p):      # for reversed (fra - eng) pairs
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


"""
Preparing data:
1. read text file and split into lines, split lines into pairs
2. normalize text, filter by length and content
3. make word list from sentence in pairs
"""


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def main():
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))


if __name__ == '__main__':
    main()
