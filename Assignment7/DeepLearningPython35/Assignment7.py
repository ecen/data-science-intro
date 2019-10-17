#%% md
# # Assignment 7
# ## Students:
- Davíð Freyr Björnsson
- Eric Guldbrand

# ## 1) Using the matplotlib library, add few lines to the program to: # ### A) also calculate the training accuracy (the number of correctly classified training samples out of 50000) after each epoch.
# ### B) Plot the training and testing accuracies vs the epoch as two curves in the same plot.

Changed mnist_loader.load_data_wrapper to also return training_data_zip, the training data in the same format as the test data. This set is created with

```
training_data_zip = zip(training_inputs, tr_d[1])
```

Changes were also done to `network.py` to return both training and test accuracy.

#%%

# read the input data
import network
import mnist_loader
import matplotlib.pyplot as plt

def train_network(layers, epochs, batch_size, eta):
    training_data, validation_data, test_data, training_data_zip = mnist_loader.load_data_wrapper()
    training_data = list(training_data)

    net = network.Network(layers)

    accuracy_train, accuracy_test = net.SGD(training_data, epochs=epochs, mini_batch_size=batch_size, eta=eta, test_data_train=training_data_zip, test_data_test = test_data)
    return accuracy_train, accuracy_test

import seaborn as sns
import pandas as pd

def plot_accuracy(accuracy_train, accuracy_test, epochs):
    alpha = 1
    df = pd.DataFrame({'epochs': range(0, epochs), 'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test})
    plot = sns.scatterplot(x="epochs", y="accuracy_train", ci="sd", data=df, alpha=alpha)
    plot = sns.scatterplot(x="epochs", y="accuracy_test", ci="sd", data=df, alpha=alpha)
    plt.legend(labels=['Training accuracy', 'Test accuracy']);

def train_and_plot(layers, epochs, batch_size, eta):
    acc_train, acc_test = train_network(layers, epochs, batch_size, eta)
    plot_accuracy(acc_train, acc_test, epochs)
    return acc_train, acc_test

def train_and_repeat(layers, epochs, batch_size, eta, repeat):
    acc_tests = []
    for i in range(0, repeat):
        acc_train, acc_test = train_network(layers, epochs, batch_size, eta)
        acc_tests.append(acc_test[-1])
    return acc_tests

#%%
acc_train, acc_test = train_and_plot([784, 30, 10], 1, 10, 3.0)

#%%
print("Training accuracy each epoch:", acc_train, acc_test)

#%% md
# ## 2) Use your program to perform the following experiments.
# ### a) Plot the training and testing curves for the case of a single hidden layer with 30 units and step size 3 with 30 epochs.

#%%
train_and_plot([784, 30, 10], 30, 10, 3.0)

# ## b) Change the number epochs to 10 and the number of hidden units to 100. Try different step sizes from 3 to 15. Repeat each step size 3 times. Report the testing result at the last epoch of each trial. For the learning rate with the best performance and learning rate 3, make two separate plots of performance with 30 epochs.

#%%
# Returns best learning rate and corresponding accuracy
def find_best_learning_rate(layers, epochs, batch_size, etas, repeat):
    best_learning_rate = etas[0]
    max_acc = -1
    for learning_rate in etas:
        max_acc_new = max(train_and_repeat(layers, epochs, batch_size, learning_rate, 3))
        if max_acc_new > max_acc:
            best_learning_rate = learning_rate
            max_acc = max_acc_new
    return [best_learning_rate, max_acc]

#%%
best_learning_rate = find_best_learning_rate([784, 10, 10], 1, 10, [3, 5, 10, 15], 3)
print("Best learning rate:", best_learning_rate)

#%%
train_and_plot([784, 100, 10], 1, 10, best_learning_rate)
plt.figure()
train_and_plot([784, 100, 10], 1, 10, 3)

#%% md # ### c) Fix the number of epochs to 10. Create a chart of testing performance for different number of hidden units (one hidden layer and repeat 3 times) with the best learning rate by repeating part 2 above. Report the best size and best learning rate with the plot for the performance with 30 epochs.

#%%
keys = [10, 20, 40, 80]
unit_options = {key: None for key in keys}
best_learning_rate = -1
max_acc = -1

for hidden_units in keys:
    unit_options[hidden_units] = find_best_learning_rate([784, hidden_units, 10], 1, 10, [3, 5, 10, 15], 1)

#%%
for key in unit_options:
    if unit_options[key][0] > best_learning_rate:
        best_learning_rate = unit_options[key][0]
    if unit_options[key][1] > max_acc:
        max_acc = unit_options[key][1]

#%%
print(unit_options)



#%% md # ### 3) Experiment with noise: a) Add few lines to the network.Network.SGD (after the line “n = len(training_data)”) to add a centered i.i.d Gaussian noise with standard deviation (std) 1 to each training data point. Use command “np.random.randn()” to create noise and note that the training_data variable is a list of tuples [x,y] of data and labels.

#%%
