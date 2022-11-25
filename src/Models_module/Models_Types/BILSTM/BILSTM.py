from src.Models_module.Models_Types.AbstractModel import AbstractModel
from keras.layers import Dense
import keras
import numpy as np


class BILSTM(AbstractModel):

    def __init__(self):
        super().__init__()
        self.n_hidden = 3
        self.n_neurons = 64
        self.activation = "relu"
        self.optimizer = keras.optimizers.Adam()

    def initialize_model(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=self.input_shape))
        for i in range(self.n_hidden - 1):
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(self.n_neurons, return_sequences=True, activation=self.activation)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(self.output_shape, return_sequences=False, activation=self.activation)))
        model.add(keras.layers.Dense(self.output_shape, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    def reshape_data(self, data, labels, n_features):
        reshaped_data = np.array(data).reshape((len(data), -1, n_features))
        return reshaped_data, labels


    def get_model_parameters_info(self):
        return {
            self.n_neurons,
            self.n_hidden,
            self.activation,
            self.optimizer,
        }
