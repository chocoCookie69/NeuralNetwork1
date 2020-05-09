import random as rn
from random import random
from math import exp
import numpy as np
import cv2
import os


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['net_input'] = activation
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(weights.__len__() - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def backward_propagate_error(network, expected): #error portion of hidden and final layer
    for i in reversed(range(network.__len__())):
        layer = network[i]
        errors = list()
        if i != network.__len__() - 1:  # for hidden layer
            for j in range(layer.__len__()):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(layer.__len__()):  # for the final layer
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(layer.__len__()):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['net_input'])


def transfer_derivative(output):
    x = transfer(output)
    return x * (1 - x)


def update_weights(network, row, l_rate):
    for i in range(network.__len__()):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(inputs.__len__()):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs, expected):
    for epoch in range(n_epoch):
        i = 0
        for row in train:
            forward_propagate(network, row)
            backward_propagate_error(network, expected[i])
            update_weights(network, row, l_rate)
            i = i + 1
        train, expected = shuffle(train, expected)


def predict(network, row):  #for testing
    outputs = forward_propagate(network, row)
    return outputs

training_data = list()
exp_out = list()
def create_training_data(size, exp):
    woo = 0
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (size, size))
            for i in range(size):
                for j in range(size):
                    if (img_array[i][j] > 200):
                        img_array[i][j] = 0
                    else:
                        img_array[i][j] = 1

            img_array = img_array.flatten()
            img_array = img_array.tolist()
            training_data.append(img_array)
            exp_out.append(exp[woo])
        woo = woo + 1
    return training_data, exp_out


def l_to_ch(pred):
    for i in range(pred.__len__()):
        if i == 0:
            to_return = '['
            to_return += ''.join(str(pred[i]))
        else:
            to_return += ', '
            to_return += ''.join(str(pred[i]))
        if i == (pred.__len__()) - 1:
            to_return += ']'
    return to_return


# For shuffling
def funk():
    return 0.5


def shuffle(dat, exp):
    rn.shuffle(dat, funk)
    rn.shuffle(exp, funk)
    return dat, exp

def one_hot_encode(x):
    y = np.identity(x, dtype=int)
    expList = y.tolist()
    return expList

# TRAINING
    #for image
DATADIR = "D:/imageDATASET/fruits/fruits-360/yoohoo"
CATEGORIES = ["Apple3", "Banana3", "Eggplant3"]
size = 6
lene = len(CATEGORIES)
expect = one_hot_encode(lene)   # [[1, 0], [0, 1]] ....similarly for more number of classes

data, expected = create_training_data(size, expect)

n_inputs = data[0].__len__()
n_outputs = expect[0].__len__()
alpha = 0.35
n_epochs = 1000

network = initialize_network(n_inputs, 5, n_outputs)
data, expected = shuffle(data, expected)
train_network(network, data, alpha, n_epochs, n_outputs, expected)

# TESTING
test = list()
test_img = cv2.imread("D:/imageDATASET/fruits/fruits-360/yoohoo/banana.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (size, size))
for i in range(size):
    for j in range(size):
        if (test_img[i][j] > 200):
            test_img[i][j] = 0
        else:
            test_img[i][j] = 1

test = test_img.flatten()
test = test.tolist()
prediction = predict(network, test)
print("N/W predicted O/P = " + l_to_ch(prediction))