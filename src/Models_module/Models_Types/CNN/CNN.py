from src.Models_module.Models_Types.AbstractModel import AbstractModel

import numpy as np
import keras

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras import Sequential


class CNN(AbstractModel):

    def __init__(self):
        super().__init__()
        self.n_hidden = 2
        self.n_neurons = 64
        self.activation = "relu"
        self.optimizer = keras.optimizers.Adamax()
        self.filters = 32
        self.kernel_size = 2
        self.n_hidden_cnn = 2

    def initialize_model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.filters, padding="valid", kernel_size=self.kernel_size, strides=1,
                         activation=self.activation,
                         input_shape=self.input_shape))
        for i in range(self.n_hidden_cnn - 1):
            model.add(
                Conv1D(filters=self.filters, padding="valid", kernel_size=self.kernel_size, strides=1,
                       activation=self.activation))
            model.add(Dropout(0.6))
            model.add(MaxPooling1D(pool_size=2, padding="valid", strides=2))
        model.add(Flatten())
        for i in range(self.n_hidden):
            model.add(Dense(self.n_neurons, activation=self.activation))
        model.add(Dense(self.output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def reshape_data(self, data, labels, n_features):
        reshaped_data = np.array(data).reshape((len(data), -1, n_features))
        return reshaped_data, labels

    def get_model_parameters_info(self):
        return {
            "filters": self.filters,
            "kernal_size": self.kernel_size,
            "n_neurons": self.n_neurons,
            "n_hidden": self.n_hidden,
            "n_hidden_cnn":self.n_hidden_cnn,
            "activation": self.activation,
            "optimizer": self.optimizer,
        }
