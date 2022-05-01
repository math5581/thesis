from email import header
from utilities.tracker.tracks import *
from utilities.feature_extraction import cosine_similarity, eucledian_distance
import itertools
import numpy as np
from utilities.cv_utilities import *
import cv2 as cv
from munkres import Munkres
import pandas as pd

# needs to take in the 3 different things, blurred descriptor, bbox and bbox descriptor seperately...

class Tracker:
    def __init__(self, frame_width, frame_height) -> None:
        # You need to fix your bug with not drawing unless redeetcted or something...
        self.active_track_list = []
        self.latest_id = 0
        self.cosine_sim = True
        self.max_euc = 30
        self.similarity_threshold = 0.85   # 0.0095
        self.frame_numb_threshold = 5
        # Maybe put to another class "base_tracker_class"
        self.df = pd.DataFrame(
            data=None, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
        self.w = frame_width
        self.h = frame_height
        # Maybe make such that you can set it here?

    def save_track_file(self, name='output.txt'):
        """ Save the track as MOT txt format"""
        with open(name, 'w') as f:
            f.write(self.df.to_csv(header=None, index=False))

    def add_existing_tracks_to_df(self, frame, conf=1):
        for tra in self.active_track_list:
            # if tra.active():
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                # As long as we don't update the confidence this is just conf of bbox.
                temp_df = {"frame": int(frame), "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                        "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                self.df = self.df.append(temp_df, ignore_index=True)

    def draw_active_tracks_on_frame(self, frame):
        for tra in self.active_track_list:
            # if tra.active():
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
            track(assigned_id, detection_array, supper = False))

    def return_min_idx(self, matrix):
        # Remove all the crappy error handling...
        val = np.amin(matrix)
        if self.cosine_sim:
            val = np.amax(matrix)
        
        # Cannot associate if similarity = 0. This is a small(but not important) bug for eucledian distnacv 
        if val == 0:
            if self.cosine_sim:
                return val, 0
            else:
                return self.max_euc, 0
        # If the euclidean distance is 10, we are out of values.
        elif self.cosine_sim is not True and val >= self.max_euc:
            return val, 0
        idx = np.where(matrix == val)
        if np.asarray(idx).shape[1] > 1:
            idx = np.asarray(idx)[:, 0]
        return val, np.asarray(idx, dtype=np.int).reshape((2,))

    def set_matrix_by_idx(self, matrix, idx):
        if self.cosine_sim:
            matrix[idx[0], :] = 0
            matrix[:, idx[1]] = 0
        else:
            matrix[idx[0], :] = self.max_euc
            matrix[:, idx[1]] = self.max_euc
        return matrix

    # Update to fit self.cosine_sim
    def update_tracks_greedy(self, detection_array):
        self.increment_track_time()
        if len(self.active_track_list):
            id_list, track_vec_list = self.get_feature_vectors_from_tracks()
            det_vec_list = [description() for description in detection_array]
            if self.cosine_sim:
                cost_matrix = self.contruct_cost_matrix(
                    det_vec_list, track_vec_list, fill=0)
            else:
                cost_matrix = self.contruct_cost_matrix(
                    det_vec_list, track_vec_list, fill=self.max_euc)
            for i in range(cost_matrix.shape[0]):
                sim, idx = self.return_min_idx(cost_matrix)
                if (self.cosine_sim is not True and sim >= self.max_euc) or (self.cosine_sim and sim == 0):  # No more associations
                    break
                cost_matrix = self.set_matrix_by_idx(cost_matrix, idx)
                det_id, track_id = idx
                # Update if similarity is good enough.
                if (self.cosine_sim and sim > self.similarity_threshold) or (self.cosine_sim is not True and sim < self.similarity_threshold):
                    if det_id < len(detection_array) and track_id < len(self.active_track_list):
                        self.active_track_list[track_id].update_track(
                            detection_array[det_id])
                elif det_id < len(detection_array):  # Istantiate new track
                    self.istantiate_track(detection_array[det_id])
            # Update time here and delete tracks
        else:
            for i in range(len(detection_array)):
                self.istantiate_track(detection_array[i])
        self.remove_tracks()

    def update_tracks(self, detection_array):
        """ Updates tracks based on feature vectors and bbox coordinates """
        self.increment_track_time()
        if len(self.active_track_list):
            id_list, track_descriptions = self.get_feature_vectors_from_tracks()
            det_descriptions = [description()
                                for description in detection_array]
            indexes, m = self.associate_hungarian_algorithm(
                det_descriptions, track_descriptions)
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                sim = m[det_id][track_id]
                # Update if similarity is good enough using cosine similarity or eucledian distance:
                if (self.cosine_sim and sim > self.similarity_threshold) or (self.cosine_sim is not True and sim < self.similarity_threshold):
                    if det_id < len(detection_array) and track_id < len(self.active_track_list):
                        self.active_track_list[track_id].update_track(
                            detection_array[det_id])
                elif det_id < len(detection_array):  # Istantiate new track
                    self.istantiate_track(detection_array[det_id])
        else:
            for i in range(len(detection_array)):
                self.istantiate_track(detection_array[i])
        self.remove_tracks()

    def update_tracks_variation(self, detection_array):
        """ Updates tracks based on feature vectors and bbox coordinates. 
            Variation with bbox sizes """
        self.increment_track_time()
        if len(self.active_track_list):
            id_list, track_descriptions = self.get_feature_vectors_from_tracks()
            det_descriptions = [description()
                                for description in detection_array]
            indexes, m = self.associate_hungarian_algorithm(
                det_descriptions, track_descriptions)
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                sim = m[det_id][track_id]

                # Update if similarity is good enough using cosine similarity or eucledian distance:
                if det_id < len(detection_array) and track_id < len(self.active_track_list):
                    bbox_size = self.get_bbox_size(self.active_track_list[track_id].tlwh(), detection_array[det_id].tlwh())
                    sim = sim / (1 - bbox_size / 4)
                    if (self.cosine_sim and sim > self.similarity_threshold) or (self.cosine_sim is not True and sim < self.similarity_threshold):
                        self.active_track_list[track_id].update_track(
                            detection_array[det_id])
                elif det_id < len(detection_array):  # Istantiate new track
                    self.istantiate_track(detection_array[det_id])
        else:
            for i in range(len(detection_array)):
                self.istantiate_track(detection_array[i])
        self.remove_tracks()

    # Deprecated
    def perform_association(self, det_feat_list, track_vec_list, threshold):
        indexes, m = self.associate_hungarian_algorithm(
            det_feat_list, track_vec_list)

        pair_list = []
        unassigned_det_ids = []
        unassigned_track_ids = []
        for i in range(len(indexes)):
            det_id, track_id = indexes[i]
            sim = m[det_id][track_id]
            # there is a problem here or something...
            # Update if similarity is good enough
            if sim < threshold:
                pair_list.append((det_id, track_id))
                continue
            # Append to unassigned lists assigned afterwards.
            if det_id < len(det_feat_list):
                unassigned_det_ids.append(det_id)
            if track_id < len(self.active_track_list):
                unassigned_track_ids.append(track_id)
        return pair_list, unassigned_det_ids, unassigned_track_ids

    # Deprecated
    def update_tracks_combined(self, det_vec_list, bbox_feat_list, bbox_list):
        self.increment_track_time()
        if len(self.active_track_list):
            id_list, track_global_list = self.get_feature_vectors_from_tracks()
            id_list, track_local_list = self.get_local_feature_vectors_from_tracks()

            # Global update
            # Need some prediction on this in order to perform reasonably
            pair_list, unassigned_det_ids, unassigned_track_ids = self.perform_association(
                det_vec_list, track_global_list, self.similarity_threshold)
            for det_id, track_id in pair_list:
                if det_id < len(det_vec_list) and track_id < len(self.active_track_list):
                    self.active_track_list[track_id].update_track(
                        det_vec_list[det_id], bbox_list[det_id], bbox_feat_list[det_id])

            # local update only if unassigned_det_ids is not empty "RE-IDS"
            if len(unassigned_det_ids) != 0:
                track_global_list = [track_global_list[id]
                                     for id in unassigned_track_ids]
                track_local_list = [track_local_list[id]
                                    for id in unassigned_track_ids]

                det_global_list = [det_vec_list[id]
                                   for id in unassigned_det_ids]
                det_local_list = [bbox_feat_list[id]
                                  for id in unassigned_det_ids]

                pair_list, unassigned_local_det_ids, unassigned_track_ids = self.perform_association(
                    det_local_list, track_local_list, 0.02)  # this threshold is not certain at all!

                for det_id, track_id in pair_list:
                    if det_id < len(det_vec_list) and track_id < len(self.active_track_list):
                        self.active_track_list[track_id].update_track(
                            det_vec_list[det_id], bbox_list[det_id], bbox_feat_list[det_id])

                ids_istantiate = [unassigned_det_ids[id]
                                  for id in unassigned_local_det_ids]
                print('istantiate: ', len(ids_istantiate))
                for id in ids_istantiate:
                    self.istantiate_track(
                        det_vec_list[id], bbox_list[id], bbox_feat_list[id])
                # Update time here and delete tracks
        else:
            for i in range(len(det_vec_list)):
                self.istantiate_track(
                    det_vec_list[i], bbox_list[i], bbox_feat_list[i])
        self.remove_tracks()

    def remove_tracks(self):
        for i, tra in reversed(list(enumerate(self.active_track_list))):
            if tra.get_time() > self.frame_numb_threshold:
                # print('deleted ', self.active_track_list[i].get_track_id())
                del self.active_track_list[i]

    def increment_track_time(self):
        for i in range(len(self.active_track_list)):
            self.active_track_list[i].update_time()

    def associate_hungarian_algorithm(self, vec1, vec2):
        m = self.contruct_cost_matrix(vec1, vec2)
        opt = Munkres()
        if self.cosine_sim:
            indexes = opt.compute(1 - m.copy())  # minus since similarity 0-1
        else:
            indexes = opt.compute(m.copy())
        return indexes, m

    def contruct_cost_matrix(self, vec1, vec2, fill=False):
        """ Constructs cost matrix"""
        l1 = len(vec1)
        l2 = len(vec2)
        s = max(l1, l2)  # Handles non-square
        m = np.zeros((s, s))
        if fill:
            m = np.full_like(m, fill)
        for i in range(l1):
            for j in range(l2):
                if self.cosine_sim:
                    m[i][j] = cosine_similarity(
                        vec1[i], vec2[j])
                else:
                    m[i][j] = eucledian_distance(
                        vec1[i], vec2[j])
        return m

    def get_feature_vectors_from_tracks(self):
        """ Returns feature list of track_ids and feature vectors for active tracks """
        id_list = []
        feature_list = []
        for i in range(len(self.active_track_list)):
            id_list.append(self.active_track_list[i].get_track_id())
            feature_list.append(
                self.active_track_list[i]())
        return id_list, feature_list

    def get_local_feature_vectors_from_tracks(self):
        """ Returns feature list of track_ids and feature vectors for active tracks """
        id_list = []
        feature_list = []
        for i in range(len(self.active_track_list)):
            id_list.append(self.active_track_list[i].get_track_id())
            feature_list.append(
                self.active_track_list[i].get_active_local_feature_vector())
        return id_list, feature_list

    @staticmethod
    def get_bbox_size(bb1, bb2):
        return bb1[2] * bb1[3] + bb2[2] * bb2[3]