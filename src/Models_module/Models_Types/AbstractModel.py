"""
@author: Gerges_Hanna

Each Model Should inherit this interface
"""
from abc import ABC, abstractmethod
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
import os
import utils.data_processing as dp
import utils.helper as helper
from keras.utils.vis_utils import plot_model
import pandas as pd
import datetime

class AbstractModel(ABC):

    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def reshape_data(self, data, labels, n_features):
        "Return reshaped_data and labeles"
        pass

    @abstractmethod
    def get_model_parameters_info(self) -> dict:
        pass

    def define_inputOutput_shape(self, reshaped_data, labels):
        self.output_shape = len(np.unique(labels))
        self.input_shape = np.array(reshaped_data).shape[1:]

    def save_model(self, model, path):
        model.save(os.path.join(path, "model.h5"))

    def load_model(self, path, print_summery=True):
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
        model = load_model(path)
        if print_summery:
            print(model.summary())
        return model

    def encode_labels_toCategorical(self, labels):
        """used to convert a class vector signs to binary class matrix (one-hot encoding).

        Requirement
        -----------
        Initialize numpy library.
        Initialize tensorflow.keras.utils library.

        Parameters
        ----------
        labels : string
            take the name for each action.

        Returns
        -------
        binary class matrix.
        """
        actions = np.unique(labels)
        lbl = []
        label_map = {label: num for num, label in enumerate(actions)}
        for action in labels:
            lbl.append(label_map[action])
        y = to_categorical(lbl).astype(int)
        return y

    def update_meta_data(self, meta_data, n_train_samples, n_validation_samples, n_test_samples, Model_Type,
                         loss_and_acc_train, loss_and_acc_val, loss_and_acc_test, model_params):

        meta_data["train_samples"] = str(n_train_samples)
        meta_data["validation_samples"] = str(n_validation_samples)
        meta_data["test_samples"] = str(n_test_samples)

        meta_data["Model_Type"] = str(Model_Type.upper())

        meta_data["loss & accuracy (train)"] = str(loss_and_acc_train)
        meta_data["loss & accuracy (validation)"] = str(loss_and_acc_val)
        meta_data["loss & accuracy (test)"] = str(loss_and_acc_test)

        meta_data["Model_Params"] = str(model_params)

        self.meta_data = meta_data
        return self.meta_data

    def save_model_files_and_info(self, full_path, model, updated_meta_data, unique_labels, history):

        # For save model
        self.save_model(model, full_path)
        # For Plot the history of training
        helper.plot_history_graph(history, save=True, save_path=full_path)
        # For Save history
        pd.DataFrame(history.history, columns=(list(history.history.keys()))).reset_index().to_csv(
            os.path.join(full_path, "history.csv"), header=True, index=False)

        # Save labels
        np.save(os.path.join(full_path, "labels.npy"), unique_labels)
        # For save new metadata
        dp.save_meta_data_as_CSV(updated_meta_data, full_path)

        # For plot your model summary
        try:
            plot_model(model, to_file=os.path.join(full_path, "model_summary.png"), show_shapes=True,
                       show_layer_names=True)
        except:
            print("You must install pydot and graphviz to save your model summery as png")

        print(
            "1-Model\n2-History csv\n3-Train and loss plots\n4-Unique labels\n5-Meta_Data\n6-Model_Summary\nsaved successfully in:\n",
            full_path)

    def generate_model_folder_name(self,model_name,test_acc):
        # To rename the file with important information like Date and accuracy and type of model
        now = datetime.datetime.now()
        current_time = str(now.strftime("%Y-%m-%d_%H-%M"))
        name = model_name + "_" + current_time + "_test_acc_" + str(test_acc)
        return name
