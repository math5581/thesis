from email import header
from utilities.tracker.tracks import *
from utilities.feature_extraction import cosine_similarity, eucledian_distance
import itertools
import numpy as np
from utilities.cv_utilities import *
import cv2 as cv
from munkres import Munkres
import pandas as pd
import copy

# needs to take in the 3 different things, blurred descriptor, bbox and bbox descriptor seperately...

class Tracker:
    def __init__(self, supper, frame_width, frame_height) -> None:
        # You need to fix your bug with not drawing unless redeetcted or something...
        self.active_track_list = []
        self.latest_id = 0
        self.keypoint_threshold = 1 # Maybe make it a percentage threshold??
        self.frame_numb_threshold = 10
        # Maybe put to another class "base_tracker_class"
        self.df = pd.DataFrame(
            data=None, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
        self.w = frame_width
        self.h = frame_height
        # Maybe make such that you can set it here?
        self.supper = supper

    def save_track_file(self, name='output.txt'):
        """ Save the track as MOT txt format"""
        with open(name, 'w') as f:
            f.write(self.df.to_csv(header=None, index=False))

    def add_existing_tracks_to_df(self, frame, conf=1):
        for tra in self.active_track_list:
            if tra.active():
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                # As long as we don't update the confidence this is just conf of bbox.
                temp_df = {"frame": int(frame), "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                        "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                self.df = self.df.append(temp_df, ignore_index=True)

    def draw_active_tracks_on_frame(self, frame):
        for tra in self.active_track_list:
            if tra.active():
                c = tra.get_track_color()
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                frame = cv.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), c, 1)
                track_id = tra.get_track_id()
                frame = cv.putText(frame, str(track_id), (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX,
                                1, c, 1, cv.LINE_AA)
        return frame

    def get_available_id(self):
        # Assumes that the last track always has the highest track ID
        self.latest_id += 1
        return self.latest_id - 1

    def istantiate_track(self, detection_array):
        """ Istantiates a track """
        assigned_id = self.get_available_id()
        self.active_track_list.append(
            track(assigned_id, detection_array))

    def update_tracks(self, detection_array):
        """ Updates tracks based on feature vectors and bbox coordinates """
        self.increment_track_time()
        if len(self.active_track_list):
            track_list = copy.deepcopy(self.active_track_list)
            indexes, m, reg = self.associate_hungarian_algorithm(detection_array, track_list)
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                N_keypoints = m[det_id][track_id]
                # Update if there is anough keypoints detected
                if N_keypoints > self.keypoint_threshold:
                    if det_id < len(detection_array) and track_id < len(self.active_track_list):
                        self.active_track_list[track_id].update_track(
                            detection_array[det_id])#, reg[det_id][track_id])    Not used anymore
                elif det_id < len(detection_array):  # Istantiate new track
                    self.istantiate_track(detection_array[det_id])
        else:
            for i in range(len(detection_array)):
                self.istantiate_track(detection_array[i])
        self.remove_tracks()

    def remove_tracks(self):
        for i, tra in reversed(list(enumerate(self.active_track_list))):
            if tra.get_time() > self.frame_numb_threshold:
                # print('deleted ', self.active_track_list[i].get_track_id())
                del self.active_track_list[i]

    def increment_track_time(self):
        for i in range(len(self.active_track_list)):
            self.active_track_list[i].update_time()

    def associate_hungarian_algorithm(self, det, track):
        m, reg = self.contruct_cost_matrix(det, track)
        ma = 1024
        opt = Munkres()
        indexes = opt.compute(ma-m)
        return indexes, m, reg

    def contruct_cost_matrix(self, det, track, fill=False):
        """ Constructs cost matrix SuperGlue"""
        l1 = len(det)
        l2 = len(track)
        s = max(l1, l2)  # Handles non-square
        m = np.zeros((s, s))
        register = [[0 for x in range(s)] for y in range(s)] 
        if fill:
            m = np.full_like(m, fill)
        for i in range(l1):
            for j in range(l2):
                # Distance screening.
                time = track[j].get_time()
                distance_threshold = self.get_threshold(time)
                dist = self.get_distance(self.get_bbox_center(det[i].tlwh()), self.get_bbox_center(track[j].tlwh())) 
                if dist > distance_threshold:
                    m[i][j] = 0
                else:
                    mconf, pred = self.supper(det[i](), track[j]())
                    #Alternative with sum here maybe????
                    m[i][j], register[i][j]= np.sum(mconf), pred
        return m, register


    def get_threshold(self,time):
        # 0.05*1/np.exp(-time/5) #13 gives 0.5 at 30
        # This is difficult without so fduck it
        if time <= 5:
            return 0.1
        else:
            return 0.15

    def get_description_from_tracks(self):
        """ Returns feature list of track_ids and feature vectors for active tracks """
        id_list = []
        keypoint_list = []
        bbox_list = []
        for i in range(len(self.active_track_list)):
            id_list.append(self.active_track_list[i].get_track_id())
            keypoint_list.append(
                self.active_track_list[i]())
            bbox_list.append(self.active_track_list[i].tlwh())
        return id_list, keypoint_list, bbox_list

    @staticmethod
    def get_bbox_size(bb1, bb2):
        return bb1[2] * bb1[3] + bb2[2] * bb2[3]

    @staticmethod
    def get_bbox_center(bbox):
        return np.asarray([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]])

    @staticmethod
    def get_distance(c1, c2):
        return np.sqrt(np.sum(np.square(np.asarray(c1)-np.asarray(c2))))
