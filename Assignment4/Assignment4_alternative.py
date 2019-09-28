#%%
import numpy as np
import pandas as pd
import glob

easy_ham_train = "./data/train_easy_ham"
hard_ham_train = "./data/train_hard_ham"
spam_train = "./data/train_spam"

easy_ham_test = "./data/test_easy_ham"
hard_ham_test = "./data/test_hard_ham"
spam_test = "./data/test_spam"

from collections import Counter
# Takes folder path as input, returns dictionary
# of folder's email content
def returnDict(path):
    dicts = {}
    for filePath in glob.glob(path + "/*"):
        with open(filePath, 'r', encoding="latin-1") as email:
            data = email.read()
            wordlist = data.split()
            countDict = Counter(wordlist)
            if "spam" in path:
                value = "spam"
            else:
                value = "ham"
        # Create a dictionary with email words as values
        dict = {value:k for k in countDict}
        dicts.update(dict)

    return dicts

dictOfDict = returnDict(easy_ham_train)
#%%
from collections import Counter
dict = ['hello', 'world', 'great']
count = Counter(dict)
# print(count)
dicts = {k:'spam' for k in count}
print(dicts)
