#%% md
__1.__ Preprocessing:
__a.__ Note that the email files contain a lot of extra information, besides the actual message. Ignore that for now, and run on the entire text. Further down (in the higher grade part), you will be asked to filter out the headers and footers.
__b.__ We don’t want to train and test on the same data. Split the spam and the ham datasets in a training set and a test set in separate folders.

#%% md
__2.__ Write a Python program that:
__a.__ Takes the four datasets (hamtrain, spamtrain, hamtest, and spamtest) as input.
__b.__ Trains on the training sets using Maximum Likelihood. Uses Laplace add-one smoothing
to avoid zero-counts of any tokens. [Tip: to avoid working with too low probabilities, use
log probs and sums instead of probabilities and products.]
__c.__ Runs a Naïve Bayes classifier on the test sets and outputs the classification results to the
screen, including
• The number of emails in each of the four data sets.
• The percentage of the ham and spam test sets that were classified correctly.

#%%
import numpy as np
import pandas as pd
import glob

def countWords(path):
    wordCounts = {}
    for filePath in glob.glob(path + "/*"):
        with open(filePath, 'r', encoding="latin-1") as file:
            data = file.read()

            wordlist = data.split()
            wordCount = {}
            for w in wordlist:
                wordCount[w] = (wordlist.count(w))

            for key in wordCount:
                if (key in wordCounts):
                    wordCounts[key] += wordCount[key]
                else:
                    wordCounts[key] = wordCount[key]
    wc = pd.DataFrame(list(wordCounts.items()), columns=['Word', 'Count'])
    wc = wc.sort_values(by=["Count"], ascending=False)
    return wc

wc_spam = countWords("./data/train_spam")
wc_easy_ham = countWords("./data/train_easy_ham")
wc_hard_ham = countWords("./data/train_hard_ham")

#%%
wc_spam.head()
wc_easy_ham.head()
wc_hard_ham.head()

# Merge ham dataframes into one and sum counts
wc_ham = wc_hard_ham.merge(wc_easy_ham, on='Word', how='outer')
wc_ham = wc_ham.fillna(0)
wc_ham['Count'] = wc_ham['Count_x'] + wc_ham['Count_y']
wc_ham = wc_ham.drop(columns = ['Count_x', 'Count_y'])

print("Check that outer join gives expected results")
print(len(wc_hard_ham.merge(wc_easy_ham, on='Word', how='inner')) + len(wc_ham) - len(wc_easy_ham) - len(wc_hard_ham) == 0)

# Normalize word counts
wc_ham['Count'] = wc_ham['Count'] / wc_ham['Count'].sum(axis=0)
wc_spam['Count'] = wc_spam['Count'] / wc_spam['Count'].sum(axis=0)

# Merge ham and spam dataframe
wc_common = wc_spam.merge(wc_ham, on='Word', how='outer', suffixes=['_spam', '_ham'])
wc_common = wc_common.fillna(0)

print("Check that outer join gives expected results")
print(len(wc_spam.merge(wc_ham, on='Word', how='inner', suffixes=['_spam', '_ham'])) + len(wc_common) - len(wc_ham) - len(wc_spam) == 0)

# Sort dataframe by the normalized difference in word count
# between spam and ham emails
wc_common['abs_diff'] = abs(wc_common['Count_spam'] - wc_common['Count_ham'])
wc_common = wc_common.sort_values(by=['abs_diff'], ascending=False)
wc_common.head()
print(wc_common[wc_common['Count_spam'] == 0].head())
print(wc_common[wc_common['Count_ham'] == 0].head())

#%% md
As a side note, the most common word for ham is the symbol ">". Studying the training data, this seems to be because in replies, the original e-mail is often indented using the ">" symbol. This raises the question, what if our model relied too much on matching that symbol? Our spam filter would likely perform quite well on test data, but could fail in real world scenarios since it might block many mails sent to you, if they were not replies to mails you had previously sent someone else.
#%%
# Let's use the top 100 words (in terms of the normalized difference in word count between spam and ham emails) as features but skip <
paths = ["./data/train_spam", "./data/train_easy_ham", "./data/train_hard_ham"]
wc_common = wc_common.reset_index(drop=True)
top5Words = wc_common.loc[1:5, 'Word']
top5Words = top5Words.reset_index(drop=True)
top5Words.head()

#%% md
For b and c we want to compute the posterior, given files $X_1, ..., X_k$ with labels Y (spam/ham), with the following formula:
$P(Y | X_1, ..., X_k) = \frac{P(Y) \cdot \prod_{k=1}^K P(X_k | Y)}{\prod_{k=1}^KP(X_k)}$. Let $P(Y) := (1)$, $\prod_{k=1}^K P(X_k | Y) := (2)$, $\prod_{k=1}^KP(X_k) := (3)$

#%%
# Step 0: Create count matrix with Laplace smoothing
counts = pd.DataFrame(1, columns=top5Words[0:,], index=['spam','ham'])
counts.head()
features = [0]*len(top5Words)
featuresVector = []
labelVector = []
#%%
for path in paths:
    if "spam" in path:
        label = "spam"
    else:
        label = "ham"
    for filePath in glob.glob(path + "/*"):
        with open(filePath, 'r', encoding="latin-1") as file:
           file_contents = file.read()
           for i in range(len(top5Words)):
               if top5Words[i] in file_contents:
                   counts.loc[label, counts.columns.values[i]] += 1
                   features[i] = 1
               else:
                   features[i] = 0
           labelVector.append(label)
           featuresVector.append(features)

# Add column and row totals
counts.loc['Total',:]= counts.sum(axis=0)
counts.loc[:,'Total'] = counts.sum(axis=1)
counts.head()

#%%
# Step 1: Compute (1) (The prior probabilities of an email being spam or ham)
def prior(counts, max_likelihood):
    if max_likelihood:
        total = counts.loc['Total', 'Total']
        spamPer = counts.loc['spam', 'Total'] / counts.loc['Total', 'Total']
        return [spamPer, 1-spamPer]
    else:
        return [0.5, 0.5]

#%%
print(counts.head())
print(counts.columns.get_loc("Total"))

#%%
print(counts.index)
print(counts.index.get_loc('Total'))


#%%

# Step 2: Compute (2) (The denominator)
def denom(counts, featuresVector):
    total = counts.loc['Total', 'Total']
    denominator = 1
    for i in range(len(featuresVector)):
        if featuresVector[i] != 0:
            denominator *= counts.iloc[counts.index.get_loc('Total'), i] / total
    return denominator

# Step 3: Compute (3) (The likelihood)
def likelihood(counts, featuresVector, labels):
    frac = 1
    likelihoods = pd.DataFrame(columns=['spam', 'ham'])
    for label in labels:
        for i in range(len(featuresVector)):
            if featuresVector[i] != 0:
                frac *= counts.iloc[counts.index.get_loc(label), i] /   counts.iloc[counts.index == 'Total', i]
    likelihoods[label] = frac
    return likelihoods

def probability(counts, featuresVector, max_likelihood=False):
    priors = prior(counts, max_likelihood)
    denominator = denom(counts, featuresVector)
    likelihoods = likelihood(counts, featuresVector)
    mult = map(lambda x,y : x*y, prios, likelihoods)
    results = mult/denom
    return results
#%%
print(probability(counts, [1,0,0,0,1]))
#%%
# def prediction(counts, featuresVector, max_likelihood=False):


#    labels = ['spam', 'ham']
#    for label in labels:
