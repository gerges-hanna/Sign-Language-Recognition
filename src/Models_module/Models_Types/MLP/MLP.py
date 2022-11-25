from src.Models_module.Models_Types.AbstractModel import AbstractModel
from keras.layers import Dense
import keras
import numpy as np


class MLP(AbstractModel):

    def __init__(self):
        super().__init__()
        self.n_hidden = 3
        self.n_neurons = 64
        self.activation = "relu"
        self.optimizer = keras.optimizers.Adam()

    def initialize_model(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=self.input_shape))
        for i in range(self.n_hidden):
            model.add(keras.layers.Dense(self.n_neurons, activation=self.activation))
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.output_shape, activation='softmax'))
        model.compile(loss="categorical_crossentropy",
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    def reshape_data(self, data, labels, n_features):
        data = np.array(data).reshape((len(data), -1, n_features))
        data_mean = []
        for i in data:
            flatten = i.mean(0)
            data_mean.append(flatten.reshape((1, -1)))
        data_mean = np.array(data_mean)
        data_mean = data_mean.reshape((-1, n_features))
        return data_mean, labels


    def get_model_parameters_info(self):
        return {
            self.n_neurons,
            self.n_hidden,
            self.activation,
            self.optimizer,
        }
