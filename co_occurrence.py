#!/usr/bin/python
# coding: UTF-8
import sys
import numpy as np


class Cooccurence:
    def __init__(self):
        self.noun_wtoi = {}
        self.verb_wtoi = {}

    def make_dictionary(self, input_file):
        with open(input_file, "r") as f:
            for line in f:
                words = line.split()
                noun, verb = words[0], words[1]
                if noun not in self.noun_wtoi:
                    self.noun_wtoi[noun] = len(self.noun_wtoi)
                if verb not in self.verb_wtoi:
                    self.verb_wtoi[verb] = len(self.verb_wtoi)
        return self.noun_wtoi, self.verb_wtoi

    def noun_to_id(self, noun):
        if noun in self.noun_wtoi:
            return self.noun_wtoi[noun]
        return self.noun_wtoi["<unk>"]

    def verb_to_id(self, verb):
        if verb in self.verb_wtoi:
            return self.verb_wtoi[verb]
        return self.verb_wtoi["<unk>"]

    def count_occurrence(self, input_file):
        N = np.zeros((len(self.noun_wtoi), len(self.verb_wtoi)))
        with open(input_file, "r") as f:
            for line in f:
                noun = line.split()[0]
                verb = line.split()[1]
                # noun -> id , verb -> id
                N[self.noun_to_id(noun), self.verb_to_id(verb)] += 1
        return N


def main(filename):
    v = Cooccurence()
    v.make_dictionary(filename)
    print(v.count_occurrence(filename))

if __name__ == "__main__":
    main(sys.argv[1])
