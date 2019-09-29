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
wc_common = wc_common.reset_index(drop=True)


#%% md
As a side note, the most common word for ham is the symbol ">". Studying the training data, this seems to be because in replies, the original e-mail is often indented using the ">" symbol. This raises the question, what if our model relied too much on matching that symbol? Our spam filter would likely perform quite well on test data, but could fail in real world scenarios since it might block many mails sent to you, if they were not replies to mails you had previously sent someone else.

#%% md
We compute the posterior, given files $X_1, ..., X_k$ with labels Y (spam/ham), with the following formula:
$P(Y | X_1, ..., X_k) = \frac{P(Y) \cdot \prod_{k=1}^K P(X_k | Y)}{\prod_{k=1}^KP(X_k)}$. Let $P(Y) := (1)$, $\prod_{k=1}^K P(X_k | Y) := (2)$, $\prod_{k=1}^KP(X_k) := (3)$

#%%
# Step 1: Compute (1) (The prior probabilities of an email being spam or ham)
def prior(counts, max_likelihood):
    if max_likelihood:
        return [0.5, 0.5]
    else:
        total = counts.loc['Total', 'Total']
        spamPer = counts.loc['spam', 'Total'] / counts.loc['Total', 'Total']
        return [spamPer, 1-spamPer]
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
def likelihood(counts, featuresVector):
    frac = 1
    labels = counts.index.values[:-1]
    likelihoods = pd.DataFrame(0, index=[0], columns=labels)
    for label in labels:
        for i in range(len(featuresVector)):
            if featuresVector[i] != 0:
                frac *= counts.iloc[counts.index.get_loc(label), i] /   counts.iloc[counts.index.get_loc('Total'), i]
        likelihoods[label] = frac
    return likelihoods

# Step 4: predict label, given feature vector
def probability(counts, featuresVector, max_likelihood):
    priors = prior(counts, max_likelihood)
    denominator = denom(counts, featuresVector)
    likelihoods = likelihood(counts, featuresVector)
    # Element-wise multiplication/division of lists
    mult = np.multiply(priors, likelihoods)
    result = np.divide(mult, denominator)
    return result

# Predict a list of labels, given a list of feature vectors
def predictOne(counts, featuresVector, max_likelihood):
    prob = probability(counts, featuresVector, max_likelihood)
    label = prob.idxmax(axis=1)[0]
    return label

def predictAll(counts, featuresVectors, max_likelihood):
    y_pred = []
    for v in featuresVectors:
        y_pred.append(predictOne(counts, v, max_likelihood))
    return y_pred

#%%
def train(hamtrainPath, spamtrainPath, N):
    # Use the top N of words as features (in terms of the normalized difference in word count between spam and ham emails) as features but skip <
    topNWords = wc_common.loc[1:N, 'Word']
    topNWords = topNWords.reset_index(drop=True)
    # Create count matrix with Laplace smoothing
    paths = [hamtrainPath, spamtrainPath]
    trainCounts = pd.DataFrame(1, columns=topNWords[0:,], index=['spam','ham'])
    featuresTemp = [0]*len(topNWords)
    featuresVector = []
    labelVector = []

    for path in paths:
        if "spam" in path:
            label = "spam"
        else:
            label = "ham"
            for filePath in glob.glob(path + "/*"):
                with open(filePath, 'r', encoding="latin-1") as file:
                    file_contents = file.read()
                    for i in range(len(topNWords)):
                        if topNWords[i] in file_contents:
                            trainCounts.loc[label, trainCounts.columns.values[i]] += 1

    # Add column and row totals
    trainCounts.loc['Total',:] = counts.sum(axis=0)
    trainCounts.loc[:,'Total'] = counts.sum(axis=1)

    return trainCounts

def test(trainCounts, hamtestPath, spamTestPath, N, max_likelihood):
    # Use the top N of words as features (in terms of the normalized difference in word count between spam and ham emails) as features but skip <
    topNWords = wc_common.loc[1:N, 'Word']
    topNWords = topNWords.reset_index(drop=True)
    featuresTemp = [0]*len(topNWords)
    featuresVector = []
    y = []
    paths = [hamtestPath, spamTestPath]

    for path in paths:
        if "spam" in path:
            label = "spam"
        else:
            label = "ham"
            for filePath in glob.glob(path + "/*"):
                with open(filePath, 'r', encoding="latin-1") as file:
                    file_contents = file.read()
                    for i in range(len(topNWords)):
                        if topNWords[i] in file_contents:
                            featuresTemp[i] = 1
                        else:
                            featuresTemp[i] = 0
                            y.append(label)
                            featuresVector.append(features)
    y_pred = predictAll(trainCounts, featuresVector, max_likelihood)
    correctPredictions = sum(a == b for a,b in zip(y_pred, y))
    accuracy = (correctPredictions*1.0) / len(y_pred)
    return accuracy

# Test
easy_ham_train_path = "./data/train_easy_ham"
hard_ham_train_path = "./data/train_hard_ham"
spam_train_path = "./data/train_spam"

easy_ham_test_path = "./data/test_easy_ham"
hard_ham_test_path = "./data/test_hard_ham"
spam_test_path = "./data/test_spam"

# Number of top words used as features (in terms of the normalized difference in word count between spam and ham emails) as features but skip <)
N = 20
max_likelihood = True

#%%
trainCounts = train(easy_ham_train_path, spam_train_path, N)

#%%
accuracy = test(trainCounts, easy_ham_test_path, spam_test_path, N, max_likelihood)

#%%
print(accuracy)





#%%
# def prediction(counts, featuresVector, max_likelihood=False):


#    labels = ['spam', 'ham']
#    for label in labels: