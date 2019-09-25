#%% md
__1.__ Preprocessing:
__a.__ Note that the email files contain a lot of extra information, besides the actual message. Ignore that for now, and run on the entire text. Further down (in the higher grade part), you will be asked to filter out the headers and footers.
__b.__ We don’t want to train and test on the same data. Split the spam and the ham datasets in a training set and a test set in separate folders.

#%%

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

#%% md
For b and c we want to compute the posterior, given files $X_1, ..., X_k$ with labels Y (spam/ham), with the following formula:
$P(Y | X_1, ..., X_k) = \frac{P(Y) \cdot \prod_{k=1}^K P(X_k | Y)}{\prod_{k=1}^KP(X_k)}$. Let $P(Y) := (1)$, $\prod_{k=1}^K P(X_k | Y) := (2)$, $\prod_{k=1}^KP(X_k) := (3)$

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

# Normalize word counts
wc_ham['Count'] = wc_ham['Count'] / wc_ham['Count'].sum(axis=0)
wc_spam['Count'] = wc_spam['Count'] / wc_spam['Count'].sum(axis=0)

# Merge ham and spam dataframe
wc_common = wc_spam.merge(wc_ham, on='Word', how='outer', suffixes=['_spam', '_ham'])
wc_common = wc_common.fillna(0)

# Sort dataframe by the normalized difference in word count
# between spam and ham emails
wc_common['abs_diff'] = abs(wc_common['Count_spam'] - wc_common['Count_ham'])
wc_common = wc_common.sort_values(by=['abs_diff'], ascending=False)

print(len(wc_spam) + len(wc_easy_ham) + len(wc_hard_ham) - wc_common.shape[0])
print(len(wc_ham.merge(wc_spam, on="Word", how='inner')))
wc_common.head(40)

#%% md
As side note, the most common word for ham is the symbol ">". Studying the training data, this seems to be because in replies, the original e-mail is often indented using the ">" symbol. This raises the question, what if our model relied too much on matching that symbol? Our spam filter would likely perform quite well on test data, but could fail in real world scenarios since it might block many mails sent to you, if they were not replies to mails you had previously sent someone else.

#%%

ar = [1,2]
nar = pd.Series(ar)
df_spam = pd.DataFrame([nar])
df_spam['labels'] = 'spam'


# df_ham =
# print(df_spam)

#%%
def spamDetector(hamtrain, spamtrain, hamtest, spamtest, max_likelihood=False):
    df = DataFrame()
    # Step 1: Compute (1) (The prior probabilities for spam and ham emails)
    if max_likelihood:
        priors = [0.5, 0.5]
    else:
        # spamPer = df[df.label=="spam"].shape[0]/df.shape[0]
        # priors = [spamPer, 1-spamPer]
    # Step 2: Compute (2) (The denominator)
    # for i in len()
