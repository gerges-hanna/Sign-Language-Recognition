"""
@author: Gerges_Hanna
"""

import keras.models
import matplotlib.pyplot as plt
import os
from keras.models import load_model


def plot_history_graph(history, save=False, save_path=None):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.legend(['train', 'val'], loc='upper left')
    if "val_accuracy" in history.history.keys():
        plt.plot(history.history['val_accuracy'])
        plt.legend(['train', 'val'], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    if save:
        plt.savefig(os.path.join(save_path, "accuracy.png"))
    plt.figure()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.legend(['loss'], loc='upper left')
    if "val_loss" in history.history.keys():
        plt.plot(history.history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    if save:
        plt.savefig(os.path.join(save_path, "loss.png"))
    plt.figure()


def load_model(path, print_summery=True):
    """used to load the saved model.

    Requirement
    -----------
    Initialize keras.models library.

    Parameters
    ----------
    model_path : string.
        take the path for the model with extension h5.

    Returns
    -------
    the saved model.
    """
    model = keras.models.load_model(path)
    if print_summery:
        print(model.summary())
    return model
