#%% md
__2.__ Write a Python program that:<br />
__a.__ Takes the four datasets (hamtrain, spamtrain, hamtest, and spamtest) as input.<br />
__b.__ Trains on the training sets using Maximum Likelihood. Uses Laplace add-one smoothing
to avoid zero-counts of any tokens. [Tip: to avoid working with too low probabilities, use
log probs and sums instead of probabilities and products.]<br />
__c.__ Runs a Naïve Bayes classifier on the test sets and outputs the classification results to the
screen, including
• The number of emails in each of the four data sets.
• The percentage of the ham and spam test sets that were classified correctly.

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

# ham_data is a tuple (nr of emails, dict)
# Where 'dict' is a dictionary where each key is a word and each value is the number of emails that word occur in.
# nr of emails is the total number of emails the dict was created on
def isHam(ham_data, spam_data, test_email):
    # Calculate combined number of words in hamtrain
    # and spamtrain
    likelihood_ham = likelihood_spam = 0
    N_ham = ham_data[0]
    N_spam = spam_data[0]
    hamdict = ham_data[1]
    spamdict = spam_data[1]

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
def howManyHam(ham_data, spam_data, test):
    predictedHam = 0
    for email in test:
        if isHam(ham_data, spam_data, email):
            predictedHam += 1
    return predictedHam

# Given paths to ham train and test set folders,
# Builds ham and spam data
def build_data(ham_train, ham_test):
    hamtrain = list_of_emails(ham_train)
    spamtrain = list_of_emails(spam_train)
    hamtest = list_of_emails(ham_test)
    spamtest = list_of_emails(spam_test)

    hamdict = calc_occurrences(hamtrain)
    spamdict = calc_occurrences(spamtrain)

    ham_data = (len(hamtrain), hamdict)
    spam_data = (len(spamtrain), spamdict)
    return ham_data, spam_data

# Given paths to ham train and test set folders,
# calculate the accuracy of predicting ham and spam
def calc_accuracy(ham_data, spam_data):
    hamCount = howManyHam(ham_data, spam_data, hamtest)
    totalHam = ham_data[0]
    ham_accuracy = hamCount / totalHam

    hamCount = howManyHam(ham_data, spam_data, spamtest)
    totalSpam = spam_data[0]
    spam_accuracy = 1 - (hamCount / totalSpam)
    return ham_accuracy, spam_accuracy

# Given a data tuple (nr of emails, dict)
# Sets all values in the dict to 0 for words on the input word list
def remove_common(data, wordsToRemove):
    for word in wordsToRemove:
        if word in data[1]:
            data[1][word] = 0
    return data

#%%
easy_ham_data, spam_data = build_data(easy_ham_train, easy_ham_test)
hard_ham_data, spam_data = build_data(hard_ham_train, hard_ham_test)

easy_ham_accuracy, easy_spam_accuracy = calc_accuracy(easy_ham_data, spam_data)
hard_ham_accuracy, hard_spam_accuracy = calc_accuracy(hard_ham_data, spam_data)

#%% md
The program trains on the data using maximum likelihood estimation, the label Y is found, that maximizes the likelihood function:
$$\sum_{i=1}^n \log P(X_i|Y)$$ Where $$P(X_k | Y) = \frac{\# \{\text{label Y,} \text{feature} X_k\} + \alpha}{N + 2K\alpha}$$ and we set $$\alpha := 1$$

#%%
print("Spam vs easy ham, ham accuracy:", round(100*easy_ham_accuracy, 2), "%")
print("Spam vs easy ham, spam accuracy:", round(100*easy_spam_accuracy,2), "%")
print("Spam vs hard ham, ham accuracy:", round(100*hard_ham_accuracy,2), "%")
print("Spam vs hard ham, spam accuracy:", round(100*hard_spam_accuracy,2), "%")

#%%
stop_words_path = "./downloaded_data/stopwords.txt"
with open(stop_words_path, 'r', encoding="latin-1") as text:
    data = text.read()

list_of_stop_words = list(data)

#%%
