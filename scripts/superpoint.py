from pathlib import Path
import argparse
from queue import Empty
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import time

from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.utilities import *
from utilities.superpoint_utils import SuperInterface
DIMENSIONS = [240, 480]

def get_bbox_center(bbox):
    return np.asarray([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]])

def get_distance(c1, c2):
    return np.sqrt(np.sum(np.square(np.asarray(c1)-np.asarray(c2))))

def get_numb_keypoints(pred):
    return pred['keypoints'][0].cpu().numpy().shape[0]

base_path = '/workspace/data/MOT17/train/'
SEQUENCES = ['MOT17-02-FRCNN', 
             'MOT17-04-FRCNN',
             'MOT17-05-FRCNN', 
             'MOT17-13-FRCNN']

supper = SuperInterface(DIMENSIONS)

def extract_super(SEQUENCE):
    """ Extracts similarity between GT """
    seq = os.path.join(base_path, SEQUENCE)
    dataloader = MOTDataloader(seq)
    cv_tools = CVTools(DIMENSIONS)

    nonmatch_keypoints = []
    nonmatch_keypoints_max = []
    match_keypoints = []
    skipper = 30
    sequence_length = dataloader.get_seuqence_length()
    ran = int(sequence_length/skipper)
    for i in range(ran):
        # Try skipping every fifth?
        # print('i ', i*skipper, ' of ', sequence_length)
        dataloader.set_current_frame_id(i*skipper)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)
        # print(id_list)
        bboxes = cv_tools.extract_bbox_from_list(bbox_list, full=True)
        #description_array = supper.get_description(bboxes, bbox_list)
        # print(len(features))
        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                temp_unmatch = []
                for prev_index, prev_id in enumerate(prev_id_list):
                    if id[0]==prev_id[0]:
                        # Pair
                        c1, c2 = get_bbox_center(bbox_list[index]), get_bbox_center(prev_bbox_list[prev_index])
                        #p1, p2 = description_array[index](), prev_description_array[prev_index]()
                        # Matching
                        dist = get_distance(c1,c2)
                        # numb_matches, _ = supper(p1,p2)
                        # sum_numb = np.sum(numb_matches)
                        match_keypoints.append(dist)
                    else:
                        #Non-pair
                        c1, c2 = get_bbox_center(prev_bbox_list[prev_index]), get_bbox_center(bbox_list[index])
                        #p1, p2 = description_array[index](), prev_description_array[prev_index]()
                        # Matching
                        dist = get_distance(c1,c2)
                        # numb_matches, _ = supper(p1,p2)
                        # sum_numb = np.sum(numb_matches)
                        temp_unmatch.append(dist)
                if len(temp_unmatch) != 0:
                    nonmatch_keypoints_max.append(min(temp_unmatch))
                nonmatch_keypoints += temp_unmatch
        #prev_description_array = description_array   
        prev_bbox_list = bbox_list
        prev_id_list = id_list
        dataloader.next_frame()
    # print(min(match_keypoints))

    save_similarity_vector(match_keypoints, SEQUENCE + "match.pkl")
    save_similarity_vector(nonmatch_keypoints, SEQUENCE + "non_match.pkl")
    save_similarity_vector(nonmatch_keypoints_max, SEQUENCE + "non_match_max.pkl") 

decimals = 8

if __name__ == '__main__':
    for seq in SEQUENCES:
        print(seq)
        extract_super(seq)
        match = np.asarray(load_similarity_vector(seq + "match.pkl"))
        print('\mu ', np.around(np.mean(match), decimals), ' \pm ', np.around(np.std(match), decimals))

        non_match = np.asarray(load_similarity_vector(seq + "non_match.pkl"))
        print('\mu ', np.around(np.mean(non_match), decimals), ' \pm ', np.around(np.std(non_match), decimals))

        non_match_max = np.asarray(load_similarity_vector(seq + "non_match_max.pkl"))
        print('\mu ', np.around(np.mean(non_match_max), decimals), ' \pm ', np.around(np.std(non_match_max), decimals))
