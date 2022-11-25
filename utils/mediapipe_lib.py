# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 02:10:38 2022

@author: Gerges_Hanna
"""

import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results, draw_face=True, draw_pose=True):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results, draw_face=True, draw_pose=True):
    if draw_face:
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
    if draw_pose:
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results, extract_face=False, extract_pose=False):
    keypoints = []
    if extract_pose:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        keypoints.append(pose)
    if extract_face:
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        keypoints.append(face)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    keypoints.append(lh)
    keypoints.append(rh)
    return np.concatenate(keypoints)


def extract_keypoints_without_Zaxis(results, extract_face=False, extract_pose=False):
    keypoints = []
    if extract_pose:
        pose = np.array([[res.x, res.y, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
        keypoints.append(pose)
    if extract_face:
        face = np.array([[res.x, res.y] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)
        keypoints.append(face)
    lh = np.array([[res.x, res.y] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
    rh = np.array([[res.x, res.y] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 2)
    keypoints.append(lh)
    keypoints.append(rh)
    return np.concatenate(keypoints)


def extract_keypoints_centered_by_nose(results, extract_face=False, extract_pose=False):
    try:
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
    except:
        nose_x = 0
        nose_y = 0
    keypoints = []
    if extract_pose:
        pose = np.array([[res.x - nose_x, res.y - nose_y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        keypoints.append(pose)
    if extract_face:
        face = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        keypoints.append(face)
    lh = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    keypoints.append(lh)
    keypoints.append(rh)
    return np.concatenate(keypoints)


def extract_keypoints_centered_by_nose_Without_Zaxis(results, extract_face=False, extract_pose=False):
    try:
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
    except:
        nose_x = 0
        nose_y = 0

    keypoints = []
    if extract_pose:
        pose = np.array([[res.x - nose_x, res.y - nose_y, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
        keypoints.append(pose)
    if extract_face:
        face = np.array([[res.x - nose_x, res.y - nose_y] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)
        keypoints.append(face)
    lh = np.array([[res.x - nose_x, res.y - nose_y] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
    rh = np.array([[res.x - nose_x, res.y - nose_y] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 2)
    keypoints.append(lh)
    keypoints.append(rh)
    return np.concatenate(keypoints)


"""
Should call before shift processing (Extract by nose)
"""


def scale_processing(results):
    min_x = 1000000000000000000
    max_x = -10000000000000000000

    min_y = 1000000000000000000
    max_y = -10000000000000000000

    if results.pose_landmarks:
        for res in results.pose_landmarks.landmark:
            if res.x < min_x:
                min_x = res.x
            if res.x > max_x:
                max_x = res.x

            if res.y < min_y:
                min_y = res.y
            if res.y > max_y:
                max_y = res.y

    if results.left_hand_landmarks:
        for res in results.left_hand_landmarks.landmark:
            if res.x < min_x:
                min_x = res.x
            if res.x > max_x:
                max_x = res.x

            if res.y < min_y:
                min_y = res.y
            if res.y > max_y:
                max_y = res.y

    if results.right_hand_landmarks:
        for res in results.right_hand_landmarks.landmark:
            if res.x < min_x:
                min_x = res.x
            if res.x > max_x:
                max_x = res.x

            if res.y < min_y:
                min_y = res.y
            if res.y > max_y:
                max_y = res.y

    div_x = (max_x - min_x) / 0.5
    div_y = (max_y - min_y) / 0.7

    if results.pose_landmarks:
        for res in results.pose_landmarks.landmark:
            res.x = res.x / div_x
            res.y = res.y / div_y

    if results.left_hand_landmarks:
        for res in results.left_hand_landmarks.landmark:
            res.x = res.x / div_x
            res.y = res.y / div_y

    if results.right_hand_landmarks:
        for res in results.right_hand_landmarks.landmark:
            res.x = res.x / div_x
            res.y = res.y / div_y

    return results


def is_action(results):
    """
    This function detects if this action or not
    The idea is that: the function checks first if there are hands or not.
    if not then there is no action if true
    then check if there is any hand above the hip in the y axis
    if there is not any hand above the hip then there is no action.

    input
    -----
    The result from mediapipe detection funtion

    output
    -----
    boolean
    """

    # chech if there are any hands
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return False

    # get the y axis for each keypoint in the hand
    # if no hand then give the keypoint large value (10)
    lh = np.array([[res.y] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.full((21),
                                                                                                                10)
    rh = np.array([[res.y] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.full((21),
                                                                                                                  10)

    # Get the max value in each hand
    # Hint: A large value means it goes down the image.
    maxval_lh = np.max(lh)
    maxval_rh = np.max(rh)

    # get the y axis values for left_hip and right_hip
    left_hip = results.pose_landmarks.landmark[23].y
    right_hip = results.pose_landmarks.landmark[24].y

    # get the minimum on
    # Hint: A minimum value means it goes up the image.
    min_hip = min(left_hip, right_hip)

    # check if the minimmum value from the hands is greater than the hip
    # if true it means all hands are below the hip
    if min(maxval_lh, maxval_rh) > min_hip:
        return False

    return True


def ask_and_get_type_of_extract_keypoints():


    type_of_extract = int(input(EXTRACTION_TYPE_MESSAGE))
    param_selection = int(input(EXTRACTION_PARAMETERS_MESSAGE))
    scale_selection = int(input(SCALE_OPERATION_MESSAGE))
    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return EXTRACTION_TYPE_SWITCHER.get(type_of_extract, ValueError('invalid chosen')), EXTRACTION_PARAMETER_SWITCHER.get(param_selection, ValueError(
        'invalid chosen')), SCALE_SWITCHER.get(scale_selection, ValueError('invalid chosen'))


def get_function_by_string_name(function_name: str):
    return globals()[function_name]


# Constant variables
SCALE_SWITCHER = {
        1: True,
        2: False
    }

EXTRACTION_TYPE_SWITCHER = {
        1: extract_keypoints,
        2: extract_keypoints_without_Zaxis,
        3: extract_keypoints_centered_by_nose,
        4: extract_keypoints_centered_by_nose_Without_Zaxis,
    }

EXTRACTION_PARAMETER_SWITCHER = {
        1: dict(extract_pose=True, extract_face=False),
        2: dict(extract_pose=False, extract_face=True),
        3: dict(extract_pose=True, extract_face=True),
        4: dict(extract_pose=False, extract_face=False),
    }

EXTRACTION_TYPE_MESSAGE = "Choose type of extract you need:\n"
EXTRACTION_TYPE_MESSAGE += "1-normal extract keypoints\n"
EXTRACTION_TYPE_MESSAGE += "2-extract_keypoints_without_Zaxis\n"
EXTRACTION_TYPE_MESSAGE += "3-extract_keypoints_centered_by_nose\n"
EXTRACTION_TYPE_MESSAGE += "4-extract_keypoints_centered_by_nose_Without_Zaxis\nchoose: "

EXTRACTION_PARAMETERS_MESSAGE = "In addition to the hands, Do you need to extract pose or face keypoints?:\n"
EXTRACTION_PARAMETERS_MESSAGE += "1-Just pose\n"
EXTRACTION_PARAMETERS_MESSAGE += "2-Just face\n"
EXTRACTION_PARAMETERS_MESSAGE += "3-Pose and face\n"
EXTRACTION_PARAMETERS_MESSAGE += "4-No\nchoose: "

SCALE_OPERATION_MESSAGE = "Do you need to processing the scale (Depth)?\n"
SCALE_OPERATION_MESSAGE += "1-Yes\n"
SCALE_OPERATION_MESSAGE += "2-No\nchoose:"