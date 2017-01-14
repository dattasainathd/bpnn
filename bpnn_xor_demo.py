#!/usr/bin/env python
from datetime import datetime

from bpnn import BPNeuralNetwork

net = BPNeuralNetwork([2, 3, 1])

error = 1.0
i = 0
training_start_time = datetime.now()

while error > 0.001:
    error = 0.0
    error = error + net.train([0, 0], [0])
    error = error + net.train([0, 1], [1])
    error = error + net.train([1, 0], [1])
    error = error + net.train([1, 1], [0])
    error = error / 4
    print 'epoch: %i error: %f' % (i, error)
    i = i + 1
print 'training time: %s' % (datetime.now() - training_start_time)
net.predict([0, 0])
print net.activations[-1]
net.predict([0, 1])
print net.activations[-1]
net.predict([1, 0])
print net.activations[-1]
net.predict([1, 1])
print net.activations[-1]
