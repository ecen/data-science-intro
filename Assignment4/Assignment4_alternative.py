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
    for filePath in glob.glob(path + "/*"):
        with open(filePath, 'r', encoding="latin-1") as email:
            data = email.read()
            word_list = data.split()
        # Create a dictionary with email words as values
        email_word_lists.append(word_list)
    return email_word_lists

#%%
# Counts the number of occurences of key in
# email list
def occurrences_in(email_list, key):
    count = 0
    for email in email_list:
        if key in email:
            count += 1
    return count

#%%
# hamtrain: dictionary of word counts in all ham emails
# spamtrain: dictionary of word counts in all spam emails
def isHam(hamtrain, spamtrain, test_email):
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
        hamtrain_count = occurrences_in(hamtrain, key)
        spamtrain_count = occurrences_in(spamtrain, key)
        likelihood_ham += math.log((hamtrain_count + 1)/(N_ham + 2*K*1))
        likelihood_spam += math.log((spamtrain_count + 1)/(N_spam + 2*K*1))
    return likelihood_ham > likelihood_spam

# Returns how many emails were classified as ham
def test(hamtrain, spamtrain, test):
    predictedHam = 0
    for email in test:
        if isHam(hamtrain, spamtrain, email):
            predictedHam += 1
    return predictedHam

def run():
    hamtrain = list_of_emails(easy_ham_train)
    spamtrain = list_of_emails(spam_train)
    hamtest = list_of_emails(easy_ham_test)
    spamtest = list_of_emails(spam_test)

    #correctHam = test(hamtrain, spamtrain, hamtest[:20])
    #totalHam = 20
    #print("Ham accuracy: " + str(correctHam / totalHam))

    correctHam = test(hamtrain, spamtrain, spamtest[:20])
    totalHam = 20
    print("Spam accuracy: " + str(1 - (correctHam / totalHam)))

#%%
start = time.time()
run()
end = time.time()
print(((end-start)/20)*845/60)
