from email import header
from utilities.tracker.tracks import *
from utilities.feature_extraction import cosine_similarity, eucledian_distance
import itertools
import numpy as np
from utilities.cv_utilities import *
import cv2 as cv
from munkres import Munkres
import pandas as pd


class Tracker:
    def __init__(self, frame_width, frame_height) -> None:
        self.active_track_list = []
        self.latest_id = 0
        self.similarity_threshold =  0.085 # now it is dist threshold, 0.95
        self.frame_numb_threshold = 10
        # Maybe put to another class "base_tracker_class"
        self.df = pd.DataFrame(
            data=None, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
        self.w = frame_width
        self.h = frame_height

    def save_track_file(self, name='output.txt'):
        """ Save the track as MOT txt format"""
        with open(name, 'w') as f:
            f.write(self.df.to_csv(header=None, index=False))

    def add_existing_tracks_to_df(self, frame, conf=1):
        for tra in self.active_track_list:
            bbox = CVTools.get_int_bbox(
                tra.get_active_bbox(), ((self.w, self.h)))
            # As long as we don't update the confidence this is just conf of bbox.
            temp_df = {"frame": int(frame), "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                       "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
            self.df = self.df.append(temp_df, ignore_index=True)

    def draw_active_tracks_on_frame(self, frame):
        for tra in self.active_track_list:
            c = tra.get_track_color()
            bbox = CVTools.get_int_bbox(
                tra.get_active_bbox(), ((self.w, self.h)))
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

    def istantiate_track(self, feature_vec, det):
        """ Istantiates a track """
        assigned_id = self.get_available_id()
        self.active_track_list.append(track(assigned_id, feature_vec, det))

    def update_tracks(self, det_vec_list, det_list):
        self.increment_track_time()
        if len(self.active_track_list):
            id_list, track_vec_list = self.get_feature_vectors_from_tracks()
            indexes, m = self.associate_hungarian_algorithm(
                det_vec_list, track_vec_list)
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                sim = m[det_id][track_id]
                # there is a problem here or something...
                # Update if similarity is good enough
                if sim < self.similarity_threshold:
                    if det_id < len(det_list) and track_id < len(self.active_track_list):
                        self.active_track_list[track_id].update_track(
                            det_vec_list[det_id], det_list[det_id])
                elif det_id < len(det_list):  # Istantiate new track
                    self.istantiate_track(
                        det_vec_list[det_id], det_list[det_id])
            # Update time here and delete tracks
        else:
            for i in range(len(det_vec_list)):
                self.istantiate_track(det_vec_list[i], det_list[i])
        self.remove_tracks()

    def remove_tracks(self):
        for i, tra in reversed(list(enumerate(self.active_track_list))):
            if tra.get_time() > self.frame_numb_threshold:
                print('deleted ', self.active_track_list[i].get_track_id())
                del self.active_track_list[i]

    def increment_track_time(self):
        for i in range(len(self.active_track_list)):
            self.active_track_list[i].update_time()

    def associate_hungarian_algorithm(self, vec1, vec2):
        m = self.contruct_cost_matrix(vec1, vec2)
        opt = Munkres()
        indexes = opt.compute(m.copy())  # minus since similarity 0-1
        return indexes, m

    def contruct_cost_matrix(self, vec1, vec2):
        """ Constructs cost matrix"""
        l1 = len(vec1)
        l2 = len(vec2)
        s = max(l1, l2)  # Handles non-square
        m = np.zeros((s, s))
        for i in range(l1):
            for j in range(l2):
                m[i][j] = eucledian_distance(  # cosine_similarity(
                    vec1[i], vec2[j])
        return m

    def get_feature_vectors_from_tracks(self):
        """ Returns feature list of track_ids and feature vectors for active tracks """
        id_list = []
        feature_list = []
        for i in range(len(self.active_track_list)):
            id_list.append(self.active_track_list[i].get_track_id())
            feature_list.append(
                self.active_track_list[i].get_active_feature_vector())
        return id_list, feature_list
