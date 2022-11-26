# Light-Weight Deep Learning Model for Dynamic Sign Language Recognition

<p align="center">
  <img src="docs/images/2103832.png" alt="Deep Learning Icon" height="70"/>
  <img src="https://i.ya-webdesign.com/images/a-plus-png-2.png" alt="plus" height="40"/>
  <img src="docs/images/logo.png" alt="MediaPipe Icon" height="80"/>
</p>

Dynamic Hand Gestures recognition in Video Sequences and in Real-Time recognition using the Deep Learning models and the MediaPipe framework.

### Prerequisites

The following prerequisites are required to make this repository work:
- Download the source code or clone the repository [here](https://github.com/gerges-hanna/Sign-Language-Recognition)
- Install the following libraries : 
  
  Install opencv:
  ```shell
  pip install opencv-python
  ```
  Install tensorflow:
  ```shell
  pip install tensorflow
  ```
  
- Prepare a Dynamic Sign Language Dataset
- The Dataset folder should contain sub-folders equal to the number of signs labeled with the name of the sign as shown 
<p align="center">
  <img src="docs/images/DSL.PNG" alt="Dataset folder example" height="300"/>
</p>

If you don’t have a Dataset for Dynamic Sign Language, use our dataset which is available for free for educational purposes. Try the [DSL-46 Dataset](https://www.google.com).

# Getting started
The model contains 3 main parts
- Extract keypoints
- Train the model
- Real-Time run


### 1- Extract keypoints
This phase is used to extract the keypoints from the given dataset through MediaPipe framework.

_Before extracting initialize the n_thread parameter with the number of threads_

#### Run the main.py script 
Enter "1" for the selection as shown 
<p align="center">
  <img src="docs/images/extract keypoints.png" alt="Extract keypoints selection" height="150"/>
</p>
After the selection follow the on-screen system questionnaire that will ask for these inputs

- Enter the name of the dataset
- Enter the directory for the dataset
- Enter the number of frames to be extracted from each video
- Choose the extraction type
- in addition to the hands, Do you need to extract the pose or face points 
- Do you need to process the scale (Depth)
- Enter the directory to save the folder
- Enter the folder name

#### Answer the following questions depending on your situation following the on-screen commands 

A snapshot example for the run: 
<p align="center">
  <img src="docs/images/Extract keypoints script1.jpeg" alt="Extract keypoints questionnaire example" height="300"/>
</p>

note: the snapshot was captured before adding the advanced technique for processing the scale

After a successful run, all the occupied threads should complete 100% as shown
<p align="center">
  <img src="docs/images/Threads complete.png" height="150"/>
</p>

Then two CSV files should be saved in the input directory as shown

<p align="center">
  <img src="docs/images/DSL_CSV.PNG" height="100"/>
</p>

- Dataset CSV file: contain the first column to show the labels of the dataset, while the rest columns show the extracted keypoints 
- Meta-Data CSV file: it contains the seven meta-data for the extraction process as shown in the next table

#### Extraction process Meta-Data

| Data Label                     | Description                                |
| ----------------------------- | ------------------------------------------ |
| `dataset`                        | Shows the dataset name as pre-input. |
| `keypoint_extractor_type`                  | Shows the extraction type as pre-input. |
| `sequence_length`         | Shows the number of frames as pre-input. |
| `n_words` | Shows the number of the signs in the used dataset. |
| `n_samples`                 | Shows the number of samples from the used dataset. |
| `count_of_features`        | Shows the number of the extracted features. |
| `extracted body parts` | Shows the extracted parts of the body. |

### 2- Train the model
This phase is used to train the model, the process starts by reading the meta-data and the dataset keypoints from the pre-extracted CSV files.

Then, the following actions will be made respectively to suit the model:
- Reshape the data and labels
- Set the shape of the input and the output
- Encode the labels
- Split the data into train, test, and validation samples

_Before the training process intialize the following params depending on your situation_
<p align="center">
  <img src="docs/images/params.png" alt="A snapshot example of pre intialized params" height="200"/>
</p>

#### Run the main.py script 
Enter "2" for the selection as shown 
<p align="center">
  <img src="docs/images/train the model.png" alt="Train the model selection" height="150"/>
</p>
After the selection, the training process will start immediately using the pre-initialized parameters

After successful training, the output will be six different files that will be saved automatically in the directory that appears on the console.

The output six files:

<p align="center">
  <img src="docs/images/Model_folder.jpeg" height="300"/>
</p>

##### 1. accuracy.png: A plot image for the accuracy throughout the training
##### 2. history.csv: The history of the accuracy and the loss numbers throughout training
##### 3. labels.npy: The labels or the unique actions that the model trained on
##### 4. loss.png: A plot image for the loss throughout the training
##### 5. Meta-Data.csv: A CSV file contains all the details about the model and the dataset process as shown in the next table

#### Training Process Meta-Data

| Data Label                     | Description                                |
| ----------------------------- | ------------------------------------------ |
| `dataset`                        | Shows the dataset name as pre-input. |
| `keypoint_extractor_type`                  | Shows the extraction type as pre-input. |
| `sequence_length`         | Shows the number of frames as pre-input. |
| `n_words` | Shows the number of the signs in the used dataset. |
| `count_of_features`        | Shows the number of extracted features. |
| `extracted body parts` | Shows the extracted parts of the body. |
| `n_samples`               | Shows the total number of samples from the used dataset. |
| `train_samples`              | Shows the number of samples that are used for training. |
| `validation_samples`         | Shows the number of samples used for validation. |
| `test_samples`               | Shows the number of samples used for testing. |
| `Model_Type`                 | Shows the used model name. |
| `loss & accuracy (train)`                | Shows the loss and accuracy of training. |
| `loss & accuracy (validation)`           | Shows the loss and accuracy of validation. |
| `loss & accuracy (test)`                 | Shows the loss and accuracy of testing. |
| `Model_Params`               | Shows the used parameters for initializing the model. |

##### 6. model.h5: The trained model to be loaded for testing and deployment

### 3- Real-Time run
This phase is used for the Real-Time Recognition


_Before the run:_

_1- attach any normal or mobile camera to your device and setup it._

_2- initialize the model_folder_path with the path of your model file as shown_
<p align="center">
  <img src="docs/images/Real-time intialize.png" height="100"/>
</p>

#### Run the main.py script 
Enter "3" for the selection as shown 
<p align="center">
  <img src="docs/images/Real-Time test.png" alt="Real-Time run selection" height="150"/>
</p>
After the selection, the attached camera will run and the Real-Time Recognition process will start immediately.


