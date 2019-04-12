import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import re
from collections import defaultdict, Counter

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def setify(wordf):
    with open(wordf) as f:
        return set([w.strip() for w in f.readlines()])


eng_stops = setify('lexicon/english.stop')


def my_tokenize(text):
    remove = str.maketrans("", "", string.punctuation + string.digits)
    all_token = [tok.translate(remove).lower() for tok in word_tokenize(text)]
    return list(filter(lambda word: len(word) > 0 and word not in eng_stops, all_token))


class NaiveBayesNLP:
    """Naive Bayes classfier for binary sentiment analysis
        Lidstone smoothing(alpha)
    """

    def __init__(self):
        self.params = {"wc": {}, "num_tokens": {}, "class_prob": {}}

    def _vocab(self, text):
        return set(my_tokenize(text))

    # def clean(self, text):  # clean and word_frequency need to be refined when feature engineer
    #     return text.translate(remove).lower()

    # def word_frequency(self, X):
    #     """Laplace +1 smooth"""
    #     # freq = defaultdict(lambda: 1)
    #     freq = defaultdict(int)
    #     for word in re.split("\W+", clean_text):
    #         self.vocab.add(word)
    #         freq[word] += 1
    #     return freq
    def word_frequency(self, text):
        return Counter(my_tokenize(text))

    def train(self, X, y):
        self.vocab = self._vocab(" ".join(X))
        self.vocab_size = len(self.vocab)
        # class probability
        self.params["class_prob"]["pos"] = np.log(np.mean(y == 1))
        self.params["class_prob"]["neg"] = np.log(np.mean(y == -1))

        # Count(W | Class) and gather vocabulary
        self.params["wc"]["pos"] = self.word_frequency(" ".join(X[y == 1]))
        self.params["wc"]["neg"] = self.word_frequency(" ".join(X[y == -1]))

        self.params["num_tokens"]["pos"] = sum(self.params["wc"]["pos"].values())
        self.params["num_tokens"]["neg"] = sum(self.params["wc"]["neg"].values())

    def _prob(self, word, sentiment, alpha=.5):
        "sentiment is either pos or neg, str type, represent the positive or negative class"
        return (self.params["wc"][sentiment][word] + alpha) / (self.params["num_tokens"][sentiment] + alpha * self.vocab_size + alpha)

    def predict(self, X, alpha=.5):
        result = []
        for x in X:
            pos, neg = 0, 0
            freqs = self.word_frequency(x)
            for word, _ in freqs.items():
                # if word not in self.vocab:
                #     continue
                pos += np.log(self._prob(word, "pos", alpha=alpha))
                neg += np.log(self._prob(word, "neg", alpha=alpha))
            pos += self.params["class_prob"]["pos"]
            pos += self.params["class_prob"]["neg"]
            if pos > neg:
                result.append(1)
            else:
                result.append(-1)
        return result
