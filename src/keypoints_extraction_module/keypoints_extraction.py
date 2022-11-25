"""
@author: Gerges_Hanna
"""

import threading
from builtins import str
import src.keypoints_extraction_module.constant as constant
import src.keypoints_extraction_module.helper as helper
import utils.mediapipe_lib as mp
from utils.ReturnValueThread import ReturnValueThread

import pandas as pd
import numpy as np
import os
import utils.data_processing as dp

class Keypoints_Extraction:
    def __init__(self):
        pass

    def start_keypoints_extraction(self,n_thread=None):
        self.inputs()
        dataset=helper.read_datasete_videos(constant.DIRECTORY)
        n_samples = len(dataset)

        if n_thread == None or n_thread == 0:
            keypoints=self.convert_videos_to_keypoints(dataset)
        else:
            keypoints = self.convert_videos_to_keypoints_using_threads(dataset, n_threads=n_thread)

        labels,keypoints=self.reshape_keypoints_to_one_row(keypoints)

        n_unique_words=len(np.unique(labels))

        self.save_keypoints_as_csv(keypoints,labels,constant.FULL_PATH)
        meta=self.extract_meta_data(n_samples,n_unique_words)
        dp.save_meta_data_as_CSV(meta,constant.FULL_PATH)


    def extract_meta_data(self,n_samples,n_unique_words):
        constant.META_DATA["dataset"] = constant.DATASET_NAME
        constant.META_DATA["count_of_features"] = constant.N_FEATURES
        constant.META_DATA["sequnce_length"] = constant.NO_FRAMES_NEEDED
        constant.META_DATA["n_samples"] = n_samples
        constant.META_DATA["keypoint_extractor_type"] = constant.TYPE_OF_EXTRACTION.__name__
        constant.META_DATA["n_words"] = n_unique_words

        constant.META_DATA['extracted body parts'] = list()
        if constant.EXRTRACTION_PARAMETERS["extract_pose"]:
            constant.META_DATA['extracted body parts'].append("Pose")
        if constant.EXRTRACTION_PARAMETERS["extract_face"]:
            constant.META_DATA['extracted body parts'].append("face")
        constant.META_DATA['extracted body parts'].append("left_hand")
        constant.META_DATA['extracted body parts'].append("right_hand")
        constant.META_DATA['extracted body parts'] = "-".join(constant.META_DATA['extracted body parts'])
        return constant.META_DATA


    def inputs(self):
        constant.DATASET_NAME = str(input("Enter the name of the dataset:"))
        constant.DIRECTORY = os.path.join(input("Enter the directory for the dataset:"))
        constant.NO_FRAMES_NEEDED = int(input("Enter the number of frame you need to extract from each video: "))
        constant.TYPE_OF_EXTRACTION, constant.EXRTRACTION_PARAMETERS, constant.IS_SCALE = mp.ask_and_get_type_of_extract_keypoints()
        constant.SAVED_PATH = os.path.join(input("Enter the directory to save the folder:"))
        constant.CSV_FILE_NAME = os.path.join(input("Enter the folder name you need to save:"))
        constant.FULL_PATH = os.path.join(constant.SAVED_PATH, constant.CSV_FILE_NAME)




    def convert_videos_to_keypoints(self,video_paths_and_actions: list):
        all_videos_keypoints = []
        print(threading.current_thread().name, " Complete 0 of ",
              len(video_paths_and_actions), " 0%")
        for video_number, video_data in enumerate(video_paths_and_actions):  # video_data = [video_path, action_name]
            frames = helper.extract_frames_from_video(video_data[0], constant.NO_FRAMES_NEEDED)
            video_keypoints = []
            for frame in frames:

                with mp.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    _, detected_frame_from_mediapipe = mp.mediapipe_detection(frame,holistic)  # extract keypoints using the old method
                    if constant.IS_SCALE:
                        detected_frame_from_mediapipe = mp.scale_processing(detected_frame_from_mediapipe)

                    if constant.TYPE_OF_EXTRACTION != None:  # extract keypoints using the passed method
                        frame_keypoints = constant.TYPE_OF_EXTRACTION(detected_frame_from_mediapipe,
                                                                     **constant.EXRTRACTION_PARAMETERS)
                    else:
                        frame_keypoints = mp.extract_keypoints(detected_frame_from_mediapipe,
                                                                 **constant.EXRTRACTION_PARAMETERS)

                video_keypoints.append(frame_keypoints)  # append keypoints to a list for this video

            all_videos_keypoints.append(
                {
                    "action_name": video_data[1],
                    "keypoints": np.array(video_keypoints)
                }
            )
            print(threading.current_thread().name, " Complete ", video_number+1, " of ",
                  len(video_paths_and_actions)," {:.2f}".format(((video_number+1)/len(video_paths_and_actions))*100),"%")
            constant.N_FEATURES=np.array(video_keypoints).shape[1]

        return all_videos_keypoints


    def reshape_keypoints_to_one_row(self,all_videos_keypoints):
        labels = []
        keypoints = []
        for i in all_videos_keypoints:
            video_kp = i["keypoints"]
            video_kp = np.reshape(video_kp, (1, -1))

            labels.append(i['action_name'])
            keypoints.append(video_kp)
        keypoints = np.array(keypoints)
        keypoints = np.reshape(keypoints, (keypoints.shape[0], keypoints.shape[2]))
        return labels, keypoints

    def save_keypoints_as_csv(self,kepoints,labels,path):
        """used to save the data as csv file.

            Requirement
            -----------
            Initialize pandas library.
            Initialize numpy library.
            Reshape video keypoints to one row

            Parameters
            ----------
            kepoints : tuple
                take the kepoints as one rows.
            labels : string
                take the name for each action.
            path : string
                take the path to where to save the file.

            Returns
            -------
            save the output to the specific path.
            """
        os.makedirs(path)
        df = pd.DataFrame(kepoints)
        df['label'] = labels
        df.set_index('label', inplace=True)
        df.to_csv(os.path.join(path, "Dataset.csv"))

    def convert_videos_to_keypoints_using_threads(self,data,n_threads):
        spltited_data=helper.split_data_per_thread(data,n_threads)
        maximum_available_threads = len(spltited_data)
        threads = []
        keypoints=[]
        for i in range(maximum_available_threads):
            threads.append(ReturnValueThread(
                target=self.convert_videos_to_keypoints,
                args=(spltited_data[i],),
                name="Thread T"+str(i)
            ))
            threads[-1].start()

        for i in range(len(threads)):
            keypoints+=threads[i].join()

        return keypoints
