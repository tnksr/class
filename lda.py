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
        self.itow = {}

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
                        self.itow[self.wtoi[token]] = token
                    tag = tagtoken[tagtoken.index("_")+1:]
                    tags.append(tag)
                self.corpus.append(document)
                self.tags.append(tags)


class LDA:
    # model structures
    def __init__(self, args):
        self.C = Corpus()
        self.C.load(args.input_file)
        self.topic = []
        self.num_topic = args.num_topic
        # counter
        self.n_zw = np.zeros((len(self.C.wtoi), self.num_topic), dtype=np.int64)
        self.n_dz = np.zeros((len(self.C.corpus), self.num_topic), dtype=np.int64)
        self.n_z = np.zeros(self.num_topic, dtype=np.int64)
        self.n_d = np.zeros(len(self.C.corpus), dtype=np.int64)

    # main algorithm of LDA
    def __call__(self):
        # initialization
        self.counter()
        # TODO : while not convergent
        for k in range(100):
            for m, document in enumerate(self.C.corpus):
                for i, token in enumerate(document):
                    # subtraction count of a token and its topic pair
                    z = self.topic[m][i]
                    self.n_zw[self.C.wtoi[token]][z] -= 1
                    self.n_dz[m][z] -= 1
                    self.n_z[z] -= 1
                    self.n_d[m] -= 1
                    # update probability
                    p = self.probability(self.n_zw[self.C.wtoi[token]][:], self.n_dz[m][:], self.n_z[:], self.n_d[m])
                    # update topic
                    p /= np.sum(p)
                    z = self.gibbs_sampling(p)
                    self.topic[m][i] = z
                    # update counter
                    self.n_zw[self.C.wtoi[token]][z] += 1
                    self.n_dz[m][z] += 1
                    self.n_z[z] += 1
                    self.n_d[m] += 1
        # examine result
        self.examine_result()

    # intialization
    def counter(self):
        for m, document in enumerate(self.C.corpus):
            tmp = []
            Z = self.assign(len(document))
            for i, token in enumerate(document):
                z = Z[i]
                tmp.append(z)
                self.n_zw[self.C.wtoi[token]][z] += 1
                self.n_dz[m][z] += 1
                self.n_z[z] += 1
                self.n_d[m] += 1
            self.topic.append(tmp)

    # randomly assign a topic to each token
    def assign(self, document_size):
        return np.random.randint(low=0, high=self.num_topic, size=document_size)

    # calculate probability of each topics
    def probability(self, n_zw, n_dz, n_z, n_d):
        # TODO : alpha, beta
        beta = 1
        alpha = 1
        return (n_zw + beta)/(n_z + beta*n_zw.shape[0]) * (n_dz + alpha)/(n_d + alpha*n_dz.shape[0])

    def gibbs_sampling(self, probability):
        # cumulative sum of probability
        cumsum = np.cumsum(probability)
        # random value
        r = np.random.rand()
        # index(topic) include random value
        z = np.where(cumsum - r > 0)
        return z[0][0]

    def examine_result(self):
        param = (self.n_zw + 1)/(self.n_z + len(self.C.wtoi))
        sort = np.argsort(param, axis=0)
        top_index = [ np.where(sort==i) for i in range(sort.shape[0]-15, sort.shape[0]) ]
        topic_token = [ [] for _ in range(self.num_topic)]
        for index in top_index:
            for topic in range(self.num_topic):
                token_index = index[0][topic]
                topic_token[topic].append(self.C.itow[token_index])
        print(topic_token)

def main():
    args = parse_args()
    l = LDA(args)
    l()

if __name__ == '__main__':
    main()