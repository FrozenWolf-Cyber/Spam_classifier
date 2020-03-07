import pandas as pd
import matplotlib.pyplot as plt
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.utils import shuffle

# ham - 1 spam - 0
index = [0, 1]
unique_dict_values = dict()
total_length = []
text_x = []
n_lstm_layer = 2
validation_size = 50

file_data = pd.read_csv("spam.csv")
text = list(file_data["Message"])
ham_or_spam = list(file_data["Category"])

# Assigining ham as 1 and spam as 0

for i in range(len(ham_or_spam)):
    if ham_or_spam[i] == "ham":
        ham_or_spam[i] = index[1]
    else:
        ham_or_spam[i] = index[0]

# Appending all the unique values so that it can be used to tokenize the text

for i in string.printable:
    unique_dict_values[len(unique_dict_values.keys())] = i

for i in text:
    total_length.append(len(i))
    for j in i:
        if j not in unique_dict_values.values():
            unique_dict_values[len(unique_dict_values.keys())] = j

# Maximum length of string is used to normalize all the text to same length

max_length = max(total_length)

# This function returns the value of the particualr letter


def get_key(x, dictionary):
    keys = dictionary.keys()

    for i in keys:
        if dictionary[i] == x:
            return i


# 94 is 'space' value
# If the text is smaller than the maximum length then the remaining spaces are filled with 94 (space)


for i in text:
    temp = []
    for j in i:
        temp.append(
            get_key(j, unique_dict_values) / 1000
        )  # Dividing by thousand to decrease the range of values
    while True:
        if len(temp) == max_length:
            break
        temp.append(94 / 1000)
    text_x.append(temp)

# Here we are splitting it into test and validation dataset.
# Here validation is used to find final result of the model

text_final = text_x[-validation_size:]
text_x = text_x[:-validation_size]
ham_or_spam_final = ham_or_spam[-validation_size:]
ham_or_spam = ham_or_spam[:-validation_size]

# Defining our model


class classifier(nn.Module):
    def __init__(self, max_length):
        super(classifier, self).__init__()
        self.lstm_layer = nn.LSTM(max_length, max_length, n_lstm_layer)
        self.linear2 = nn.Linear(max_length, 1)

    def forward(self, x):
        out = self.lstm_layer(
            x,
            (
                torch.zeros((n_lstm_layer, x.shape[1], x.shape[2])),
                torch.zeros((n_lstm_layer, x.shape[1], x.shape[2])),
            ),
        )
        out = self.linear2(x)
        out = F.sigmoid(out)
        return out


# For each epoch the training data is divided into test and train data


def test_validation(x, y):
    x, y = shuffle(x, y, random_state=13)
    text_x_tensor = torch.FloatTensor(x[:-validation_size]).reshape(
        1, len(x[:-validation_size]), len(x[0])
    )
    ham_or_spam_tensor = torch.FloatTensor(y[:-validation_size]).reshape(
        1, len(y[:-validation_size]), 1
    )
    text_validation_x_tensor = torch.FloatTensor(x[-validation_size:]).reshape(
        1, len(x[-validation_size:]), len(x[0])
    )

    ham_or_spam_validation_tensor = torch.FloatTensor(y[-validation_size::1]).reshape(
        1, len(y[-validation_size::1]), 1
    )

    return (
        text_x_tensor,
        ham_or_spam_tensor,
        text_validation_x_tensor,
        ham_or_spam_validation_tensor,
    )


model = classifier(max_length)
criterion = nn.BCELoss()
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
loss_v_history = []
epochs = 100

for epoch in range(epochs):
    (
        text_x_tensor,
        ham_or_spam_tensor,
        text_validation_x_tensor,
        ham_or_spam_validation_tensor,
    ) = test_validation(text_x, ham_or_spam)
    epoch += 1
    optimizer.zero_grad()

    outputs = model(text_x_tensor)
    loss = criterion(outputs, ham_or_spam_tensor)
    loss_history.append(loss)
    print("Epoch " + str(epoch) + " train data loss: " + str(loss), end=" ")
    loss.backward()

    optimizer.step()
    loss_validation = criterion(
        model(text_validation_x_tensor), ham_or_spam_validation_tensor
    )
    print("validation loss: " + str(loss_validation))
    loss_v_history.append(loss_validation)

# Plotting test loss and validation loss

plt.plot(loss_history)
plt.plot(loss_v_history)
plt.show()

# Here in the output values greater than 0.5 means 1 (ham) othervise 0 (spam)

for i in range(len(text_final)):
    text_x_tensor = torch.FloatTensor(text_final[i]).reshape(1, 1, len(text_final[i]))
    print(model(text_x_tensor), ham_or_spam_final[i])
