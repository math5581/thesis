from email import header
from utilities.tracker.tracks import *
from utilities.feature_extraction import cosine_similarity, eucledian_distance
import numpy as np
from utilities.cv_utilities import *
import cv2 as cv
from munkres import Munkres
import pandas as pd

# needs to take in the 3 different things, blurred descriptor, bbox and bbox descriptor seperately...

def delete_elements(array, elements_to_delete):
    for index in sorted(elements_to_delete, reverse=True):
        del array[index]
    return array

class Tracker:
    def __init__(self, frame_width, frame_height, supper) -> None:
        # You need to fix your bug with not drawing unless redeetcted or something...
        self.supper = supper
        self.track_list = []
        self.latest_id = 0
        self.min_sim_threshold = 0.85 
        self.frame_numb_threshold = 30  # change to be hybrid
        self.kpts_score_threshold = 1  # What to do here???

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
        for tra in self.track_list:
            if tra.active():
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                # As long as we don't update the confidence this is just conf of bbox.
                temp_df = {"frame": int(frame), "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                        "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                self.df = self.df.append(temp_df, ignore_index=True)

    def draw_active_tracks_on_frame(self, frame):
        for tra in self.track_list:
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

    def istantiate_track(self, description):
        """ Istantiates a track """
        assigned_id = self.get_available_id()
        self.track_list.append(
            track(assigned_id, description, supper = False))

    def return_min_idx(self, matrix):
        # Remove all the crappy error handling...
        val = np.amax(matrix)
        
        # Cannot associate if similarity = 0. This is a small(but not important) bug for eucledian distnacv 
        if val == 0:
            return val, 0

        idx = np.where(matrix == val)
        if np.asarray(idx).shape[1] > 1:
            idx = np.asarray(idx)[:, 0]
        return val, np.asarray(idx, dtype=np.int).reshape((2,))

    def set_matrix_by_idx(self, matrix, idx):
        matrix[idx[0], :] = 0
        matrix[:, idx[1]] = 0
        return matrix

    def update_tracks(self, detection_array):
        """ Updates tracks based on feature vectors and bbox coordinates """
        self.increment_track_time()
        if len(self.track_list):
            active_tracks, inactive_tracks = self.get_tracks()
            indexes, m = self.associate_hungarian(detection_array, active_tracks)# active_tracks)
            ids_to_delete = []
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                sim = m[det_id][track_id]
                if det_id < len(detection_array) and track_id < len(active_tracks):
                    # Update if similarity is good enough using cosine similarity or eucledian distance:
                    sim_thres = max(active_tracks[track_id].get_sim_conf(), self.min_sim_threshold)
                    # print(sim_thres)
                    if sim > sim_thres:
                        # Perform association:
                            active_tracks[track_id].update_track(
                                detection_array[det_id], sim)
                            ids_to_delete.append(det_id)

            delete_elements(detection_array, ids_to_delete)
            # Perform RE-ID here with superpoint on inactive tracks.
            if len(inactive_tracks) and len(detection_array):
                ids_to_delete = []
                indexes, m = self.associate_hungarian(detection_array, inactive_tracks, supper=True)
                for i in range(len(indexes)):
                    det_id, track_id = indexes[i]
                    score = m[det_id][track_id]
                    if det_id < len(detection_array) and track_id < len(inactive_tracks):
                        # You can maybe to interpolation here?
                        sim = cosine_similarity(detection_array[det_id](), inactive_tracks[track_id]())
                        if score > self.kpts_score_threshold:
                            inactive_tracks[track_id].update_track(
                                    detection_array[det_id], sim)
                            ids_to_delete.append(det_id)
                delete_elements(detection_array, ids_to_delete)
            # istantiate the rest
            for det in detection_array:
                self.istantiate_track(det)
        
        else:
            for det in detection_array:
                self.istantiate_track(det)
        self.remove_tracks()

    def remove_tracks(self):
        for i, tra in reversed(list(enumerate(self.track_list))):
            if tra.get_time() > self.frame_numb_threshold:
                # print('deleted ', self.track_list[i].get_track_id())
                del self.track_list[i]

    def increment_track_time(self):
        for i in range(len(self.track_list)):
            self.track_list[i].update_time()

    def associate_hungarian(self, det, track, supper=False):
        m = self.contruct_cost_matrix(det, track, supper)
        opt = Munkres()
        if supper:
            ma = 1024
            indexes = opt.compute(ma - m.copy())
        else:
            indexes = opt.compute(1 - m.copy())
        return indexes, m

    def contruct_cost_matrix(self, det, track, supper=False):
        """ Constructs cost matrix"""
        l1 = len(det)
        l2 = len(track)
        s = max(l1, l2)  # Handles non-square
        m = np.zeros((s, s))
        if supper:
            for i in range(l1):
                for j in range(l2):
                    time = track[j].get_time()
                    distance_threshold = self.get_threshold(time)
                    dist = self.get_distance(self.get_bbox_center(det[i].tlwh()), self.get_bbox_center(track[j].tlwh()))
                    if dist > distance_threshold:
                        m[i][j] = 0
                    else:
                        m[i][j] = np.sum(self.supper(
                            det[i].get_local_features(), track[j].get_local_features())[0])
        else:
            for i in range(l1):
                for j in range(l2):
                    m[i][j] = cosine_similarity(
                            det[i](), track[j]())
        return m

    def get_tracks(self):
        """ Returns feature list of track_ids and feature vectors for active tracks """
        track_list = []
        intrack_list = []
        for i in range(len(self.track_list)):
            if self.track_list[i].active():
                track_list.append(
                    self.track_list[i])
            else:
                intrack_list.append(self.track_list[i])
        return track_list, intrack_list

    def get_local_feature_vectors_from_tracks(self):
        """ Returns feature list of track_ids and feature vectors for active tracks """
        id_list = []
        feature_list = []
        for i in range(len(self.track_list)):
            id_list.append(self.track_list[i].get_track_id())
            feature_list.append(
                self.track_list[i].get_active_local_feature_vector())
        return id_list, feature_list

    def get_threshold(self,time):
        # 0.05*1/np.exp(-time/5) #13 gives 0.5 at 30
        # This is difficult without so fduck it
        if time <= 1:
            return 0.05
        if time <= 5:
            return 0.15
        else:
            return 0.3

    @staticmethod
    def get_bbox_size(bb1, bb2):
        return bb1[2] * bb1[3] + bb2[2] * bb2[3]

    @staticmethod
    def get_bbox_center(bbox):
        return np.asarray([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]])

    @staticmethod
    def get_distance(c1, c2):
        return np.sqrt(np.sum(np.square(np.asarray(c1)-np.asarray(c2))))
