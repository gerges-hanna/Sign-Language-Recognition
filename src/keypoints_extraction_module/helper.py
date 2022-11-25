import cv2 as cv
import os
from math import ceil
import threading

def extract_frames_from_video(video_file_path, n_frames_needed):
    cap = cv.VideoCapture(video_file_path)
    frames_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, n_frames_needed):
        idx = int((i / n_frames_needed) * frames_count)
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)  # To get spacific frame
        success, frame = cap.read()
        if success:
            frames.append(frame)
        else:
            print("Error in ", video_file_path)
    cap.release()
    return frames


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def read_datasete_videos(directory):
    labels=os.listdir(path=directory)
    actions=[]
    videos=[]
    for label in labels:
        videos_temp=list_full_paths(os.path.join(directory,label))
        actions+=[label]*len(videos_temp)
        videos+=videos_temp
    return [[videos[i],actions[i]] for i in range(len(videos))]


def split_data_per_thread(data, number_of_threads):
    videos_per_thread = len(data) / (number_of_threads)
    videos_per_thread = ceil(videos_per_thread)
    start = 0
    end = videos_per_thread
    spltited_data = []
    while any(data):
        spltited_data.append(data[start:end])
        del data[start:end]

    return spltited_data


