import cv2
import utils.mediapipe_lib as mp
import utils.data_processing as dp
import utils.helper as helper
import numpy as np
import os


class RealTime:

    def execute(self,model_folder_path,videoCapture=0,threshold=0.7,n_predection_trust=5):
        """Initialize the constant varaibles"""
        # root_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        model_path = os.path.join(model_folder_path)
        # model_path=os.path.join(root_path,"save_models","rnn_2022-06-20_18-09_test_acc_94")

        META_DATA_PATH = os.path.join(model_path, "Meta-Data.csv")
        # Read Meta Data
        meta_data = dp.read_meta_data(META_DATA_PATH)

        unique_actions = np.load(os.path.join(model_path, "labels.npy"), allow_pickle=True)
        model = helper.load_model(os.path.join(model_path, "model.h5"), print_summery=True)

        """Initialize real-time parameter"""

        real_time_parameters = dict(
            model=model,
            feature_extraction=mp.get_function_by_string_name(meta_data["keypoint_extractor_type"]),
            actions=unique_actions,
            threshold=threshold,  # You can change the threshold to fit the real time
            sequence_length=int(meta_data["sequnce_length"]),
            # videoCapture is equal to zero if you want to turn on the internal camera and one for the external camera
            videoCapture=videoCapture,
            calculate_keypoints_mean=True if "mlp" in meta_data["Model_Type"].lower() else False,
            draw_face=True if "face" in meta_data["extracted body parts"].lower() else False,
            draw_pose=True if "pose" in meta_data["extracted body parts"].lower() else False,
            n_predection_trust=n_predection_trust
        )

        self.__architecture(**real_time_parameters)

    def __architecture(self, model, feature_extraction, actions, threshold=0.5, sequence_length=30, videoCapture=0,
                       calculate_keypoints_mean=False, draw_face=False, draw_pose=False, n_predection_trust=3):
        are_actions = []
        sequence = []
        sentence = []
        predictions = []

        word_background_color = (6, 17, 60)  # RGB # init the color for pretected sign
        bar_color = (21, 19, 60)
        word_color = (236, 153, 75)
        word_background_color = word_background_color[::-1]  # Convert from RGB to BGR
        bar_color = bar_color[::-1]  # Convert from RGB to BGR
        word_color = word_color[::-1]  # Convert from RGB to BGR
        cap = cv2.VideoCapture(videoCapture)  # init which camera will use
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # The width of the frame
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # The height of the frame
        center = (int(width / 2) - 100, int(height / 2))  # get center of frame to show NO ACTION word
        with mp.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                ret, frame = cap.read()  # Read The frame
                image, results = mp.mediapipe_detection(frame, holistic)  # Make detections
                are_actions.append(mp.is_action(results))  # check if this frame contain action or not

                # Draw landmarks on the frame (just for show not predict)
                mp.draw_styled_landmarks(image, results, draw_face=draw_face, draw_pose=draw_pose)

                keypoints = feature_extraction(results, extract_face=draw_face,
                                               extract_pose=draw_pose)  # Extract the keypoints by your choosen function
                sequence.append(keypoints)  # Collect the keypoint
                sequence = sequence[-sequence_length:]  # get last sequence_length==>(30) to predict.
                are_actions = are_actions[
                              -sequence_length:]  # get last sequence_length==>(30) to detect is that action or not.
                if len(sequence) == sequence_length:  # check sequence if has the number of keypoints the model need.
                    if are_actions.count(
                            False) > sequence_length / 1.5:  # check if false more than true then it is not action.
                        # Show NO ACTION in the center of the frame
                        cv2.putText(image, "NO ACTION", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    else:  # if  it's action
                        # In case ANN model
                        if calculate_keypoints_mean:
                            sequence_mean = np.array(sequence).mean(0)  # Get the mean of all keypoints sequence
                            res = model.predict(np.expand_dims(sequence_mean, axis=0))[0]  # Get the result of predict
                        # In case RNN model
                        else:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]  # Get the result of predict
                        res_max = np.argmax(res)  # get index for the highest result
                        print(actions[res_max])

                        predictions.append(np.argmax(res))
                        if np.unique(predictions[-n_predection_trust:])[0] == np.argmax(res):
                            if res[res_max] > threshold:  # if result percentage is greater than the threshold

                                # Here check if previous sentence is equal new sentence then don't show it
                                if len(sentence) > 0:
                                    if actions[res_max] != sentence[-1]:
                                        sentence.append(actions[res_max])
                                else:
                                    sentence.append(actions[res_max])

                        # Show last 5 sentences in the screen
                        if len(sentence) > 5:
                            sentence = sentence[-3:]

                        # Viz probabilities of the result
                        cv2.rectangle(image, (0, 60), (int(np.max(res) * 100), 90), word_background_color, -1)
                        cv2.putText(image, actions[res_max], (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, word_color, 2,
                                    cv2.LINE_AA)

                cv2.rectangle(image, (0, 0), (640, 40), bar_color, -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, word_color, 2, cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
