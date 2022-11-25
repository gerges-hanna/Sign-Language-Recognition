"""
@author: Gerges_Hanna
"""

import os.path
import numpy as np
import keras

from src.Models_module.Models import Models

from src.real_time_module.RealTime import RealTime
from src.keypoints_extraction_module.keypoints_extraction import Keypoints_Extraction
import utils.data_processing as dp
import utils.helper as help

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    select_operation = input(
        "Choose the operation:\n1-Extract keypoints\n2-Train the model\n3-Run Real Time\nEnter your selection:"
    )
    ########################## Keypoints ##########################
    if select_operation == "1":  # Start the keypoint extraction process
        # Set the number of threads you need
        n_thread=5
        keypoints_extraction = Keypoints_Extraction()
        keypoints_extraction.start_keypoints_extraction(n_thread=n_thread)

    ########################## MODEl ##########################
    elif select_operation == "2":  # Train the model
        """
           Initialize the next dictionary whatever suits you.
           """
        params = dict(
            # set the path for the keypoints dataset
            keypoints_folder_path=r"G:\python project\Sign Language Recognition\CSV_centered\dsl46-centered",
            model_enum_type=Models.BILSTM,  # Select your fav model from enum selection
            patience=100,  # Set the number of patience in the early stopping
            epochs=2,  # set number of epochs for fit the model
            val_size=0.2,  # set the percentage for choose the size of the data for the validation
            test_size=0.2,  # set the percentage for choose the size of the data for the test
            # Set the path to save the model files in.
            path_to_save_model_files=r"C:\Users\Gerges_Hanna\Desktop\models"

        )

        model_config = params["model_enum_type"].value  # To access the class of model
        # Read the Metadata file
        meta_data = dp.read_meta_data(os.path.join(params["keypoints_folder_path"], "Meta-Data.csv"))
        # Read data and labels from CSV file
        data, labels = dp.read_dataset_fromCSV(os.path.join(params["keypoints_folder_path"], "Dataset.csv"),
                                               int(meta_data['sequnce_length']))
        # Reshape the data and label to suit the model
        reshaped_data, labels = model_config.reshape_data(data, labels, int(meta_data['count_of_features']))
        # Set the shape of input and output to model architecture
        model_config.define_inputOutput_shape(reshaped_data, labels)
        # Encode the labels to suit the model
        encoded_lbls = model_config.encode_labels_toCategorical(labels)
        # Split the data into train,test, and validation samples
        X_train, X_test, X_val, y_train, y_test, y_val = dp.split_train_test_validation(reshaped_data, encoded_lbls,
                                                                                        params["test_size"],
                                                                                        params["val_size"])
        # Define model architecture
        model = model_config.initialize_model()
        # Define early stopping
        early_stopping = keras.callbacks.EarlyStopping(patience=params["patience"], restore_best_weights=True)
        # Define all callbacks for the model Here
        callbacks = [early_stopping]
        # Fit the model and save the history of training
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params["epochs"],
                            callbacks=callbacks)

        # To detect if the training is ended by normal epochs or forced by early stopping (0 = normal , 0 > early stopping)
        # Update history if it's stopped by early stopping
        if early_stopping.stopped_epoch != 0:
            history = dp.update_history(history, patience=params["patience"])

        # Update metadata by the new info about training
        updated_meta = model_config.update_meta_data(meta_data, len(y_train), len(y_val), len(y_test),
                                                     params["model_enum_type"].name,
                                                     model.evaluate(X_train, y_train, verbose=0),
                                                     model.evaluate(X_test, y_test, verbose=0),
                                                     model.evaluate(X_val, y_val, verbose=0),
                                                     model_config.get_model_parameters_info())

        # Generate Name for the folder
        name = model_config.generate_model_folder_name(params["model_enum_type"].name,
                                                       int(model.evaluate(X_test, y_test, verbose=0)[1] * 100))
        # Save all files related to the model and training
        model_config.save_model_files_and_info(os.path.join(params["path_to_save_model_files"], name),
                                               model,
                                               updated_meta, np.unique(labels), history)

    ########################## Real Time ##########################
    elif select_operation == "3":  # Run Real Time
        """
        Enter the path of model folder
        """
        model_folder_path=r"C:\Users\Gerges_Hanna\Desktop\models\MLP_2022-11-25_18-25_test_acc_56"
        real_time = RealTime()
        real_time.execute(model_folder_path)
    else:  # Error selection
        print("Error selection")




