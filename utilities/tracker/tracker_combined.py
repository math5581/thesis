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
        self.min_sim_threshold = 0.80 #0.85 Adjusted to also fit with bad detections...
        self.max_frames = 60
        self.kpts_score_threshold = 5  # Percentage threshold?? Can you make this hybrid o??

        # Maybe put to another class "base_tracker_class"
        self.df = pd.DataFrame(
            data=None, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
        self.w = frame_width
        self.h = frame_height
        # Maybe make such that you can set it here?

        # Only to write videos:
        self.N_colors = 30
        self.colors = matplotlib.colors.ListedColormap(
            plt.get_cmap('nipy_spectral')(np.linspace(0, 1, self.N_colors))).colors

    def save_track_file(self, name='output.txt'):
        """ Save the track as MOT txt format"""
        with open(name, 'w') as f:
            f.write(self.df.to_csv(header=None, index=False))

    def get_color(self, id):
        temp = self.colors[id % self.N_colors]
        return tuple(i * 255 for i in temp[:3])

    def save_video_file(self, cap, data_loader):
        for frame_number in range(data_loader.get_seuqence_length()):
            frame_number += 1
            data_loader.set_current_frame_id(frame_number)
            frame = data_loader.get_current_frame()

            temp_df = self.df[self.df['frame']==frame_number]
            for index, line in temp_df.iterrows(): # Iterate through the dfs here.
                track_id = int(line['id'])
                c = self.get_color(track_id)
                bbox = int(line["bb_left"]), int(line["bb_top"]), int(line["bb_width"]), int(line["bb_height"])
                frame = cv.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), c, 1)
                frame = cv.putText(frame, str(track_id), (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX,
                                1, c, 1, cv.LINE_AA)
            frame = cv.putText(frame, str(frame_number), (25, 25), cv.FONT_HERSHEY_SIMPLEX,
                        1, (255,128,0), 1, cv.LINE_AA)    
            cap.write(frame)    
            # writing the detections for this frame...
        # for frameid in frame
        # save=true

    def add_existing_tracks_to_df(self, frame, conf=1):
        for tra in self.track_list:
            # You have to do something here! Perhaps still project forward with 5 ? and just interpolate thereafter?
            if tra.get_time() <= 2:
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                # As long as we don't update the confidence this is just conf of bbox.
                temp_df = {"frame": int(frame), "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                        "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                self.df = self.df.append(temp_df, ignore_index=True)

            # Writing interpolating bb if re-identified a bbox
            bb_array = tra.get_inter_bb()
            if bb_array is not None and True:
                temp = len(bb_array)
                for i, bb in enumerate(bb_array):
                    bbox = CVTools.get_int_bbox(bb, ((self.w, self.h)))
                    # As long as we don't update the confidence this is just conf of bbox.
                    temp_df = {"frame": int(frame) - temp + i, "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                            "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                    self.df = self.df.append(temp_df, ignore_index=True)
        

    def draw_active_tracks_on_frame(self, frame, frame_number):
        for tra in self.track_list:
            if tra.get_time() <= 5:
                c = tra.get_track_color()
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                frame = cv.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), c, 1)
                track_id = tra.get_track_id()
                frame = cv.putText(frame, str(track_id), (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX,
                                1, c, 1, cv.LINE_AA)
    
        frame = cv.putText(frame, str(frame_number), (25, 25), cv.FONT_HERSHEY_SIMPLEX,
                                1, (255,128,0), 1, cv.LINE_AA)             
        return frame

    def get_available_id(self):
        # Assumes that the last track always has the highest track ID
        self.latest_id += 1
        return self.latest_id - 1

    def istantiate_track(self, description):
        """ Istantiates a track """
        assigned_id = self.get_available_id()
        self.track_list.append(
            track(assigned_id, description))

    def update_tracks_combined(self, detection_array):
        """ Updates tracks based on feature vectors and bbox coordinates """
        self.increment_track_time()
        if len(self.track_list):
            indexes, m = self.associate_hungarian_combined(detection_array, self.track_list)
            ids_to_delete = []
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                sim = m[det_id][track_id]
                if det_id < len(detection_array) and track_id < len(self.track_list):
                    # Update if similarity is good enough using cosine similarity or eucledian distance:
                    sim_thres = max(self.track_list[track_id].get_sim_conf(), self.min_sim_threshold)
                    # print('id ', self.track_list[track_id].get_track_id(), ' sim ', sim)
                    if sim > sim_thres:
                        # Perform association:
                            self.track_list[track_id].update_track(
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
                    if sim > sim_thres:
                        # Perform association:
                            active_tracks[track_id].update_track(
                                detection_array[det_id], sim)
                            ids_to_delete.append(det_id)

            delete_elements(detection_array, ids_to_delete)
            # Removed RE_ID
            # Perform RE-ID here with superpoint on inactive tracks. Can also do greedy but...
            if len(inactive_tracks) and len(detection_array):
                ids_to_delete = []
                indexes, m = self.associate_hungarian(detection_array, inactive_tracks, supper=True)
                for i in range(len(indexes)):
                    det_id, track_id = indexes[i]
                    score = m[det_id][track_id]
                    if det_id < len(detection_array) and track_id < len(inactive_tracks):
                        # You can maybe to interpolation here?
                        if score > self.kpts_score_threshold:
                            sim = cosine_similarity(detection_array[det_id].get_glob_feat(), inactive_tracks[track_id].get_glob_feat())
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
            thresh = min(self.max_frames, tra.get_N_associations()/2)
            # Deleting tracks. Aims to only use good tracks.
            if tra.get_time() > thresh:
                del self.track_list[i]

    def increment_track_time(self):
        for i in range(len(self.track_list)):
            self.track_list[i].update_time()

    def associate_hungarian_combined(self, det, track):
        m = self.contruct_cost_matrix_combined(det, track)
        opt = Munkres()
        indexes = opt.compute(-m.copy())
        return indexes, m

    def associate_hungarian(self, det, track, supper=False):
        m = self.contruct_cost_matrix(det, track, supper)
        opt = Munkres()
        if m.shape[0]:
            indexes = opt.compute(-m.copy())
            return indexes, m
        else:
            return [], []


    # are the zero values influencing it?
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
                        m[i][j] = -1
                    else:
                        # print('re-ideing')
                        m[i][j], _ = self.supper(
                            det[i].get_keypoints(), track[j].get_keypoints())
        else:
            for i in range(l1):
                for j in range(l2):
                    m[i][j] = cosine_similarity(
                            det[i].get_glob_feat(), track[j].get_glob_feat())
        return m

    def contruct_cost_matrix_combined(self, det, track):
        """ Constructs cost matrix"""
        l1 = len(det)
        l2 = len(track)
        s = max(l1, l2)  # Handles non-square
        m = np.zeros((s, s)) # Try with ones as well here.
        n = 0
        t = 0
        for i in range(l1):
            for j in range(l2):
                time = track[j].get_time()
                distance_threshold = self.get_threshold(time)
                dist = self.get_distance(self.get_bbox_center(det[i].tlwh()), self.get_bbox_center(track[j].tlwh()))
                if dist > distance_threshold:
                    m[i][j] = -1 # Try the alternative  
                else:
                    m[i][j], _ = self.supper(
                        det[i].get_keypoints(), track[j].get_keypoints(), percentage=True)
                    m[i][j] += cosine_similarity(det[i].get_glob_feat(), track[j].get_glob_feat())
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

    def get_threshold(self, time):
        # Maybe try to increase this??
        if time <= 5:
            return 0.05
        elif time <= 10:
            return 0.15
        elif time <= 30:
            return 0.20
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
