import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import re
from collections import defaultdict

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer

class NaiveBayesNLP:
    """Naive Bayes classfier for binary sentiment analysis
        Lidstone smoothing(alpha)
    """

    def __init__(self):
        self.vocab = set()
        self.params = {"wc": {}, "num_tokens": {}, "class_prob": {}}
        self.ps = PorterStemmer()

    def clean(self, text):  # clean and word_frequency need to be refined when feature engineer
        remove = str.maketrans("", "", string.punctuation + string.digits)
        return text.translate(remove).lower()

    def word_frequency(self, clean_text):
        """Laplace +1 smooth"""
        # freq = defaultdict(lambda: 1)
        freq = defaultdict(int)
        for word in re.split("\W+", clean_text):
            self.vocab.add(self.ps.stem(word))
            freq[self.ps.stem(word)] += 1
        return freq

    def train(self, X, y):
        # class probability
        self.params["class_prob"]["pos"] = np.log(np.mean(y == 1))
        self.params["class_prob"]["neg"] = np.log(np.mean(y == -1))

        # Count(W | Class) and gather vocabulary
        self.params["wc"]["pos"] = self.word_frequency(self.clean(" ".join(X[y == 1])))
        self.params["wc"]["neg"] = self.word_frequency(self.clean(" ".join(X[y == -1])))

        self.params["num_tokens"]["pos"] = sum(self.params["wc"]["pos"].values())
        self.params["num_tokens"]["neg"] = sum(self.params["wc"]["neg"].values())

    def _prob(self, word, direction, alpha=.5):
        return (self.params["wc"][direction][self.ps.stem(word)] + alpha) / (self.params["num_tokens"][direction] + alpha * len(self.vocab) + alpha)

    def predict(self, X, alpha=.5):
        result = []
        for x in X:
            pos, neg = 0, 0
            freqs = self.word_frequency(self.clean(x))
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
