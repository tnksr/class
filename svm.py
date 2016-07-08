import argparse
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str)
    parser.add_argument("test_file", type=str)
    parser.add_argument("c", type=float)
    return parser.parse_args()


class Vocabulary:
    def __init__(self, input_file):
        self.wordtoid = {"<unk>":0}
        self.input_file = input_file
        self.num_sample = int

    def dictionary(self):
        self.num_sample = 0
        with open(self.input_file, 'r') as f:
            for line in f:
                self.num_sample += 1
                for word in line.split():
                    if word not in self.wordtoid:
                        self.wordtoid[word] = len(self.wordtoid)
        return self.wordtoid

    def vector(self, is_train=True, test_file=None):

        wordtoid = Vocabulary.dictionary(self)
        category_set = np.zeros(self.num_sample, dtype=np.int32)
        feature_set = np.zeros((self.num_sample, len(wordtoid)), dtype=np.int32)
        feature = np.zeros((len(wordtoid)), dtype=np.int32)

        if is_train:
            input_file = self.input_file
        else:
            input_file = test_file

        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                category_set[idx] = 0 if line[0] == 'N' else 1
                sentence = line[2:]
                for word in sentence.split():
                    if word not in wordtoid:
                        word = '<unk>'
                    feature[wordtoid[word]] = 1
                #print(feature)
                feature_set[idx] = feature
        return feature_set, category_set


def classify(train_data, test_file, c):
    X, y = train_data.vector(True)
    Z, answer = train_data.vector(False, test_file)
    clf = svm.SVC(C=c,kernel='linear')
    clf.fit(X, y)
    prediction = clf.predict(Z)
    print(prediction, answer)
    return prediction, answer


def main():
    args = parse_args()
    train_data = Vocabulary(args.train_file)
    prediction, answer = classify(train_data, args.test_file, args.c)
    print(prediction, answer)
    print(classification_report(answer, prediction))
    print(accuracy_score(answer, prediction))
    print(confusion_matrix(answer, prediction))


if __name__ == "__main__":
    main()