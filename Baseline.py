import pandas as pd
import numpy as np
import string
import re
from collections import Counter


from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def Baseline(X,Y,A,B):
    scores = Counter()
    for x, y in zip(X, Y):
        remove = str.maketrans("", "", string.punctuation + string.digits)
        x = x.translate(remove).lower()
        # print (x)
        x_word = x.strip().split(" ")
        for w in x_word:
            if y == 1:
                scores[w]+=1
            elif y == -1:
                scores[w]-=1
    # print (scores)    
    good, bad = set(), set()
    for w, score in scores.items():
        if score>=0: 
            good.add(w)
        else: 
            bad.add(w)
    # print (good)
    # print (bad)
    # print (f"good size = {len(good)}")
    # print (f"bad size = {len(bad)}")
    # print (f"vocab size = {len(good)+ len(bad)}")
    count = 0
    correct = 0
    for x, y in zip(A, B):
        remove = str.maketrans("", "", string.punctuation + string.digits)
        x = x.translate(remove).lower()
        x_word = x.strip().split(" ")
        count +=1
        if Baseline_Classifier(good, bad, x_word) == y:
            correct += 1
    return correct/count
    
def Baseline_Classifier(good, bad, x):
    score = 0
    for w in x:
        if w in good:
            score +=1
        elif w in bad:
            score -=1
    # print (x)
    # print (score)
    if score > 0:
        return 1
    return -1

matrix = np.zeros([8,5])
data = pd.read_csv("review-ascii-only.dev", sep = "\t")
bases = np.zeros(5)
for seed in (1,2,3,4,5):

    X_train, X_test, y_train, y_test = train_test_split(data["review"], data["label"], test_size=.2, random_state=seed)
    # test = pd.read_csv("review-ascii-only.test", sep = "\t")
    # X_test, y_test = test['review'], test['label']
    # X_train, y_train = data['review'], data['label']
    tmp = 0
    base = Baseline(X_train, y_train, X_test, y_test)
    bases[seed - 1] = base
ans = np.mean(matrix, axis = 1)
ans2 = np.mean(bases)
tmp = 0
print(f'---------- Average ----------')
print (f"Accuracy of Baseline = {ans2}")
# nb._metrics()
