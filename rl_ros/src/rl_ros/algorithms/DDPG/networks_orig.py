#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Input, Concatenate

"""
    # Reference Model Tamplate
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
"""
class CriticNetwork(keras.Model):

    
    def __init__(self, n_actions, obs_shape):
        super(CriticNetwork, self).__init__()
        self.action_shape = n_actions
        self.obs_shape = obs_shape

        self.observation_input = Input(shape =(1,)+ self.obs_shape, name='observation_input')
        self.action_input = Input(shape = (self.action_shape,), name='action_input')
        self.c1 = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu')(self.observation_input)
        self.c2 = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu')(self.c1)
        self.c3 = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(self.c2)
        self.f1 = Flatten()(self.c3)
        self.d1 = Dense(253, activation='relu')(self.f1)
        self.cc = Concatenate()([self.d1, self.action_input])
        self.d2 = Dense(128, activation='relu')(self.cc)
        self.d3 = Dense(32, activation='relu')(self.d2)
        self.d4 = Dense(1, activation='linear')(self.d3)


    # have to define inputs as a tuple because the model.save() function
    # trips an error when trying to save a call function with two inputs.
    def call(self, inputs):
        state, action = inputs
        # action_value = self.fc1(tf.concat([state, action], axis=1))
        # action_value = self.fc2(action_value)
        x = self.observation_input(state)
        y= self.action_input(action)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.cc()([x, y])
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, obs_shape):
        super(ActorNetwork, self).__init__()
        self.action_shape = n_actions
        self.obs_shape = obs_shape

        self.observation_input = Input(shape =(1,)+ self.obs_shape, name='observation_input')
        self.c1 = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu')(self.observation_input)
        self.c2 = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu')(self.c1)
        self.c3 = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(self.c2)
        self.f1 = Flatten()(self.c3)
        self.d1 = Dense(253, activation='relu')(self.f1)
        self.d2 = Dense(128, activation='relu')(self.d1)
        self.d3 = Dense(32, activation='relu')(self.d2)
        self.d4 = Dense(self.action_shape, activation='linear')(self.d3)
    
    def call(self, state):
        x = self.observation_input(state)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x
"""
class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return 
"""

