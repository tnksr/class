#!/usr/bin/python
# coding: UTF-8

import sys


def count_type(text):
    # count the number of word types in the input file.
    words = text.split()
    # set don't include same word
    word_set = set()
    for word in words:
        word_set.add(word)
    return len(word_set)


# count the number of word tokens in the input file.
def count_token(text):
    return len(text.split())

if __name__ == '__main__':
    input_file = sys.argv[1]
    with open(input_file, "r") as f:
        input_text = f.read()
    print(count_type(input_text))
    print(count_token(input_text))
