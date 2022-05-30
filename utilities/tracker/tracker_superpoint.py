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


def delete_elements(array, elements_to_delete):
    for index in sorted(elements_to_delete, reverse=True):
        del array[index]
    return array

class Tracker:
    def __init__(self, supper, frame_width, frame_height, bbox_shape) -> None:
        # You need to fix your bug with not drawing unless redeetcted or something...
        self.active_track_list = []
        self.latest_id = 0
        self.keypoint_threshold = 0 # Maybe make it a percentage threshold??
        # self.min_percentage_threshold = 0 # For now just at 0...
        self.max_frames = 60
        # Maybe put to another class "base_tracker_class"
        self.df = pd.DataFrame(
            data=None, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
        self.w = frame_width
        self.h = frame_height
        # Maybe make such that you can set it here?
        self.supper = supper
        self._bbox_shape = bbox_shape

        # To write files.
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

    def add_existing_tracks_to_df(self, frame, conf=1):
        for tra in self.active_track_list:
            if tra.active():
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                # As long as we don't update the confidence this is just conf of bbox.
                temp_df = {"frame": int(frame), "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                        "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                self.df = self.df.append(temp_df, ignore_index=True)

            bb_array = tra.get_inter_bb()
            if bb_array is not None:
                temp = len(bb_array)
                for i, bb in enumerate(bb_array):
                    bbox = CVTools.get_int_bbox(bb, ((self.w, self.h)))
                    # As long as we don't update the confidence this is just conf of bbox.
                    temp_df = {"frame": int(frame) - temp + i, "id": int(tra.get_track_id()), "bb_left": bbox[0], "bb_top": bbox[1],
                            "bb_width": bbox[2], "bb_height": bbox[3], "conf": bbox[4], "x": -1, "y": -1, "z": -1}
                    self.df = self.df.append(temp_df, ignore_index=True)

    def draw_active_tracks_on_frame(self, frame):
        # Deprecated...
        for tra in self.active_track_list:
            #if tra.active():
                c = tra.get_track_color()
                bbox = CVTools.get_int_bbox(
                    tra.tlwh(), ((self.w, self.h)))
                frame = cv.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), c, 1)
                track_id = tra.get_track_id()
                frame = cv.putText(frame, str(track_id), (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX,
                                1, c, 1, cv.LINE_AA)
        return frame

    def draw_keypoints_frame(self, frame):
        for tra in self.active_track_list:
            c = tra.get_track_color()
            kpts = tra()
            kpt = kpts['keypoints'][0].cpu().numpy()
            for k in kpt:
                frame = cv2.circle(frame, (int(k[0]), int(k[1])), 1, c, 2)
        cv2.imwrite('test.png', frame)


    def get_available_id(self):
        # Assumes that the last track always has the highest track ID
        self.latest_id += 1
        return self.latest_id - 1

    def istantiate_track(self, detection_array):
        """ Istantiates a track """
        assigned_id = self.get_available_id()
        self.active_track_list.append(
            track(assigned_id, detection_array, (self.h, self.w), self._bbox_shape))

    def update_tracks(self, detection_array):
        """ Updates tracks based on feature vectors and bbox coordinates """
        self.increment_track_time()
        if len(self.active_track_list):
            track_list = copy.deepcopy(self.active_track_list)
            indexes, m = self.associate_hungarian_algorithm(detection_array, track_list)
            ids_to_delete = []
            for i in range(len(indexes)):
                det_id, track_id = indexes[i]
                score = m[det_id][track_id]
                # Update if there is anough keypoints detected
                if det_id < len(detection_array) and track_id < len(self.active_track_list):
                    thresh = max(self.keypoint_threshold, self.active_track_list[track_id].get_sim_conf())
                    # print(score, thresh)
                    if score > thresh:
                            self.active_track_list[track_id].update_track(
                                detection_array[det_id], score)#,  Not used anymore
                            ids_to_delete.append(det_id)
            delete_elements(detection_array, ids_to_delete)
            for det in detection_array:
                self.istantiate_track(det)
        else:
            for i in range(len(detection_array)):
                self.istantiate_track(detection_array[i])
        self.remove_tracks()

    def remove_tracks(self):
        for i, tra in reversed(list(enumerate(self.active_track_list))):
            thresh = min(self.max_frames, tra.get_N_associations())
            # thresh = max(thresh, 15)
            # however set the activeness lower??? Did'nt you have good results, why keep??
            # Deleting tracks. Aims to only use good tracks.
            if tra.get_time() > thresh:
                del self.active_track_list[i]

    def increment_track_time(self):
        for i in range(len(self.active_track_list)):
            self.active_track_list[i].update_time()

    def associate_hungarian_algorithm(self, det, track):
        m = self.contruct_cost_matrix(det, track)
        ma = 1024
        opt = Munkres()
        indexes = opt.compute(-m)
        return indexes, m

    def contruct_cost_matrix(self, det, track, fill=False):
        """ Constructs cost matrix SuperGlue"""
        l2 = len(track)
        l1 = len(det)
        s = max(l1, l2)  # Handles non-square
        m = np.zeros((s, s))
        for i in range(l1):
            for j in range(l2):
                # Distance screening.
                time = track[j].get_time()
                distance_threshold = self.get_threshold(time)
                dist = self.get_distance(self.get_bbox_center(det[i].tlwh()), self.get_bbox_center(track[j].tlwh())) 
                if dist > distance_threshold:
                    m[i][j] = -1
                else:
                    # mconf, pred = self.supper(det[i](), track[j](), percentage=True)
                    #Alternative with sum here maybe????
                    m[i][j], _ = self.supper(det[i](), track[j]())#, percentage=True)
        return m


    def get_threshold(self, time):
        # Maybe try to increase this??
        if time <= 5:
            return 0.1
        elif time <= 10:
            return 0.15
        elif time <= 30:
            return 0.20
        else:
            return 0.3

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
