#!/usr/bin/python
# coding: UTF-8

# naive bayes classification

import sys
import math


class Multinomial:
    def __init__(self):
        self.positive_vocabularies = {"<unk>":0}
        self.negative_vocabularies = {"<unk>":0}
        self.word_counter = [0, 0]
        self.positive_counter = [0]
        self.negative_counter = [0]

    def make_dictionary(self, word, vocabularies, counter):
        if word not in vocabularies:
            vocabularies[word] = len(vocabularies)
            counter += [1]
            if not len(vocabularies) == len(counter):
                print("different id")
            else:
                counter[vocabularies[word]] += 1
        return vocabularies, counter


    def train(self, train_file):
        with open(train_file) as f:
            for line in f:
                category, sentence = line[0], line[2:]
                if category == "N":
                    for word in sentence.split():
                        self.word_counter[0] += 1
                        self.negative_vocabularies, self.negative_counter = \
                            self.make_dictionary(word, self.negative_vocabularies, self.negative_counter)

                elif category == "P":
                    for word in sentence.split():
                        self.word_counter[1] += 1
                        self.positive_vocabularies, self.positive_counter = \
                            self.make_dictionary(word, self.positive_vocabularies, self.positive_counter)
                else:
                    print( str(line) +  "can't define positive or negative")
        return self.word_counter, self.positive_counter, self.negative_counter

    def map_classify(self, test_file):

        correct_classify = 0
        all_classify = 0
        category = []

        with open(test_file, 'r') as f:
            for idx, line in enumerate(test_file):
                sentence = line[2:].split()
                negative_prob = self.word_counter[0] / (self.word_counter[0] + self.word_counter[1])
                positive_prob = self.word_counter[1] / (self.word_counter[0] + self.word_counter[1])
                for word in sentence:
                    if word in self.negative_vocabularies:
                        negative_prob += math.log(
                            self.negative_counter[self.negative_vocabularies[word]] / self.word_counter[0])
                    else:
                        negative_prob += 0
                    if word in self.positive_vocabularies:
                        positive_prob += math.log(
                            self.positive_counter[self.positive_vocabularies[word]] / self.word_counter[1])
                    else:
                        positive_prob += 0

                if positive_prob > negative_prob:
                    category = "P"
                else:
                    category = "N"

                # accuracy
                if category == line[0]:
                    correct_classify += 1
                    all_classify += 1
                else:
                    all_classify += 1
        return correct_classify/all_classify

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    nb = Multinomial()
    print(nb.train(train_file))
    print(nb.map_classify(test_file))
