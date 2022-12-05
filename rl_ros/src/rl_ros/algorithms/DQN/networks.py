#!/usr/bin/env python3

  
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DeepQNetwork(keras.Model):
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.fc3 = Dense(n_actions, activation=None)

    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x



