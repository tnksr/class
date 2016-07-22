#!/usr/bin/python
# coding: UTF-8
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--topic_size", dest="num_topic", type=int, default=5,
                        help="The size of topic")
    parser.add_argument("--iteration", dest="iteration", type=int, default=1000,
                        help="The maximum number of training iteration")
    return parser.parse_args()

# Data structures
class Corpus:
    def __init__(self):
        self.corpus = []
        self.tags = []
        self.wtoi = {}

    def load(self, input_file):
        with open(input_file, "r") as f:
            for line in f:
                tagtokens = line.split()
                document = []
                tags = []
                for tagtoken in tagtokens:
                    token = tagtoken[:tagtoken.index("_")]
                    document.append(token)
                    if token not in self.wtoi:
                        self.wtoi[token] = len(self.wtoi)
                    tag = tagtoken[tagtoken.index("_")+1:]
                    tags.append(tag)
                self.corpus.append(document)
                self.tags.append(tags)

class LDA:
    def __init__(self, args):
        self.C = Corpus()
        self.C.load(args.input_file)
        self.topic = []
        # counter
        self.n_zw = np.zeros((len(self.C.wtoi), args.num_topic), dtype=np.int64)
        self.n_dz = np.zeros((len(self.C.corpus), args.num_topic), dtype=np.int64)
        self.n_d = np.zeros(len(self.C.corpus), dtype=np.int64)
        self.n_z = np.zeros(args.num_topic, dtype=np.int64)

    def initialization(self):
        for m, document in enumerate(self.C.corpus):
            tmp = []
            for token in document:
                z = self.gaussian(token)
                tmp.append(z)
                self.n_zw[self.C.wtoi[token]][z] += 1
                self.n_dz[m][z] += 1
                self.n_d[m] += 1
                self.n_z[z] += 1

    def gaussian(self, token):
    # return topic ~ the topic distribution follows gaussian
        return np.random.rand(0,1)

    def __call__(self, *args, **kwargs):
        