import pandas as pd
import matplotlib.pyplot as plt
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.utils import shuffle

index = [0, 1]
# ham - 1 spam - 0

file_data = pd.read_csv("spam.csv")
text = list(file_data['Message'])
ham_or_spam = list(file_data['Category'])

for i in range(len(ham_or_spam)):
    if ham_or_spam[i] == 'ham':
        ham_or_spam[i] = 1
    else:
        ham_or_spam[i] = 0

unique_dict_values = dict()

for i in string.printable:
    unique_dict_values[len(unique_dict_values.keys())] = i

total_length = []
for i in text:
    total_length.append(len(i))
    for j in i:
        if j not in unique_dict_values.values():
            unique_dict_values[len(unique_dict_values.keys())] = j

max_length = max(total_length)


def get_key(x, dictionary):
    keys = dictionary.keys()

    for i in keys:
        if dictionary[i] == x:
            return i


# 94 is 'space' value

text_x = []

for i in text:
    temp = []
    for j in i:
        temp.append(get_key(j, unique_dict_values) / 1000)
    while True:
        if (len(temp) == max_length):
            break
        temp.append(94 / 1000)
    text_x.append(temp)

text_final = text_x[-50:]
text_x = text_x[:-50]
ham_or_spam_final = ham_or_spam[-50:]
ham_or_spam = ham_or_spam[:-50]


class classifier(nn.Module):
    def __init__(self, max_length):
        super(classifier, self).__init__()
        self.lstm_layer = nn.LSTM(max_length, max_length, 2)
        self.linear2 = nn.Linear(max_length, 1)

    def forward(self, x):
        out = self.lstm_layer(x, (torch.zeros((2, x.shape[1], 910)), torch.zeros((2, x.shape[1], 910))))
        out = self.linear2(x)
        out = F.sigmoid(out)
        return out


def test_validation(x, y):
    x, y = shuffle(x, y, random_state=13)
    text_x_tensor = torch.FloatTensor(x[:-50]).reshape(1, len(x[:-50]), len(x[0]))
    ham_or_spam_tensor = torch.FloatTensor(y[:-50]).reshape(1, len(y[:-50]), 1)
    text_validation_x_tensor = torch.FloatTensor(x[-50:]).reshape(1, len(x[-50:]), len(x[0]))

    ham_or_spam_validation_tensor = torch.FloatTensor(y[-50::1]).reshape(1, len(y[-50::1]), 1)

    return text_x_tensor, ham_or_spam_tensor, text_validation_x_tensor, ham_or_spam_validation_tensor


model = classifier(max_length)
criterion = nn.BCELoss()
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
loss_v_history = []
epochs = 100

for epoch in range(epochs):
    text_x_tensor, ham_or_spam_tensor, text_validation_x_tensor, ham_or_spam_validation_tensor = test_validation(text_x,
                                                                                                                 ham_or_spam)

    epoch += 1
    optimizer.zero_grad()

    outputs = model(text_x_tensor)
    loss = criterion(outputs, ham_or_spam_tensor)
    loss_history.append(loss)
    print('train data : ' + str(loss), end=' ')
    loss.backward()

    optimizer.step()
    loss_validation = criterion(model(text_validation_x_tensor), ham_or_spam_validation_tensor)
    print('validation loss: ' + str(loss_validation))
    loss_v_history.append(loss_validation)

plt.plot(loss_history)
plt.plot(loss_v_history)
plt.show()

for i in range(len(text_final)):
    text_x_tensor = torch.FloatTensor(text_final[i]).reshape(1, 1, len(text_final[i]))
    print(model(text_x_tensor),ham_or_spam_final[i])
