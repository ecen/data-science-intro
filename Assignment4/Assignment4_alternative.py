#%%
import numpy as np
import pandas as pd
import glob
import math
import time

easy_ham_train = "./data/train_easy_ham"
hard_ham_train = "./data/train_hard_ham"
spam_train = "./data/train_spam"

easy_ham_test = "./data/test_easy_ham"
hard_ham_test = "./data/test_hard_ham"
spam_test = "./data/test_spam"

# Takes folder path as input, returns a list of
# lists of all emails
def list_of_emails(path):
    email_word_lists = []
    keys = []
    unique_words = []
    for filePath in glob.glob(path + "/*"):
        with open(filePath, 'r', encoding="latin-1") as email:
            data = email.read()
            word_list = data.split()
        # Create a dictionary with email words as values
        email_word_lists.append(word_list)
    return email_word_lists

# Returns a dictionary where each key is a word in email_list
# and the value is the number of emails that word occurs in.
# The result will contain one key for each unique word.
def calc_occurrences(email_list):
    all_words = [item for sublist in email_list for item in sublist] # Flatten
    unique_words = list(set(all_words))
    dictionary = {}
    for word in unique_words:
        dictionary[word] = 0
        for email in email_list:
                if word in email:
                    dictionary[word] += 1
    return dictionary

# Counts the number of occurences of key in
# email list
def occurrences_in(email_dict, key):
    if key in email_dict:
        return email_dict[key]
    else:
        return 0

# hamtrain: dictionary of word counts in all ham emails
# spamtrain: dictionary of word counts in all spam emails
def isHam(hamtrain, hamdict, spamtrain, spamdict, test_email):
    # Calculate combined number of words in hamtrain
    # and spamtrain
    likelihood_ham = likelihood_spam = 0
    N_ham = len(hamtrain)
    N_spam = len(spamtrain)

    # List of unique words in test set
    keys = list(set(test_email))
    K = len(keys)
    # Calculate probability of email being ham/spam
    # with Laplace smoothing, alpha := 1
    for key in keys:
        hamtrain_count = spamtrain_count = 0
        hamtrain_count = occurrences_in(hamdict, key)
        spamtrain_count = occurrences_in(spamdict, key)
        likelihood_ham += math.log((hamtrain_count + 1)/(N_ham + 2*K*1))
        likelihood_spam += math.log((spamtrain_count + 1)/(N_spam + 2*K*1))
    return likelihood_ham > likelihood_spam

# Returns how many emails were classified as ham
def test(hamtrain, hamdict, spamtrain, spamdict, test):
    predictedHam = 0
    for email in test:
        if isHam(hamtrain, hamdict, spamtrain, spamdict, email):
            predictedHam += 1
    return predictedHam

def run():
    start = time.time()
    hamtrain = list_of_emails(easy_ham_train)
    spamtrain = list_of_emails(spam_train)
    hamtest = list_of_emails(easy_ham_test)
    spamtest = list_of_emails(spam_test)

    hamdict = calc_occurrences(hamtrain)
    print(str(((time.time()-start))/60) + " minutes")

    spamdict = calc_occurrences(spamtrain)
    print(str(((time.time()-start))/60) + " minutes")

    correctHam = test(hamtrain, hamdict, spamtrain, spamdict, hamtest)
    totalHam = len(hamtrain)
    print("Ham accuracy: " + str(correctHam / totalHam))
    print(str(((time.time()-start))/60) + " minutes")

    correctHam = test(hamtrain, hamdict, spamtrain, spamdict, spamtest)
    totalSpam = len(spamtrain)
    print("Spam accuracy: " + str(1 - (correctHam / totalSpam)))
    print(str(((time.time()-start))/60) + " minutes")

#%%
run()

#%%
stop_words_path = "./data/stopwords.txt"
with open(stop_words_path, 'r', encoding="latin-1") as text:
    data = text.read()
list_of_stop_words = list(data)

#%%
