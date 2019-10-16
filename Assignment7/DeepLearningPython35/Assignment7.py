#%% md
# # Assignment 7
# ## Students:
- Davíð Freyr Björnsson
- Eric Guldbrand

# ### 1) Using the matplotlib library, add few lines to the program to: A) also calculate the training accuracy (the number of correctly classified training samples out of 50000) after each epoch.
# ### B) Plot the training and testing accuracies vs the epoch as two curves in the same plot.

Changed mnist_loader.load_data_wrapper to also return training_data_zip, the training data in the same format as the test data. This set is created with

```
training_data_zip = zip(training_inputs, tr_d[1])
```

#%%

# read the input data
import network
import mnist_loader
import seaborn as sb
training_data, validation_data, test_data, training_data_zip = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# network.py example:

net = network.Network([784, 30, 10])

accuracy_train, accuracy_test = net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data_train=training_data_zip, test_data_test = test_data)

#%%
sb.
print(accuracy_train, "out of", len(training_data))
