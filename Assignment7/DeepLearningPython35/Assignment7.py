#%% md
# # Assignment 7
# ## Students:
- Davíð Freyr Björnsson
- Eric Guldbrand

#%%
# ### 1) Using the matplotlib library, add few lines to the program to: A) also calculate the training accuracy (the number of correctly classified training samples out of 50000) after each epoch.
# ### B) Plot the training and testing accuracies vs the epoch as two curves in the same plot.

# ----------------------
# - read the input data:
import network
import mnist_loader
import seaborn as sb
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data_zip = training_data
training_data = list(training_data)

# ---------------------
# - network.py example:

net = network.Network([784, 30, 10])

accuracy_training = net.SGD(training_data, epochs=2, mini_batch_size=10, eta=3.0, test_data=training_data_zip)

#accuracy_test = net.SGD(training_data, epochs=2, mini_batch_size=10, eta=3.0, test_data=test_data)

#%%
print(accuracy_training, "out of", len(training_data))
