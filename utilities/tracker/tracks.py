import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# This track describes all the properties of a single track.
# For now, a track is only described by the deep feature representation of its last assigned detection:
# - Forward prediction in terms of a Kalman Filter
# - what if you make a crazy kalman prediction based on obtained features. Only represent a track by the deep features???
# - Average the feature vector over last N frames


class track:
    def __init__(self, assigned_id: int, global_feat_vec, bbox, local_feat_vec=None, N_average=20):
        # Colors
        self.N_COLORS = 30
        self.colors = matplotlib.colors.ListedColormap(
            plt.get_cmap('nipy_spectral')(np.linspace(0, 1, self.N_COLORS))).colors
        # print(self.colors)

        self.N_average = N_average
        self.set_track_id(assigned_id)
        self.set_track_color(assigned_id)
        self.set_track_status(True)
        # self._active_global_feature_vector
        # self._active_local_feature_vector
        self.list_local_feat_vec = []
        self.set_active_global_feature_vector(
            global_feat_vec)
        self._active_local_feature_vector = None
        self.set_active_local_feature_vector(local_feat_vec)
        self.set_active_bbox(bbox)
        self._frames_since_last_update = 0

    def set_track_color(self, assigned_id: int):
        self._color = self.colors[assigned_id % self.N_COLORS]

    def get_track_color(self):
        return tuple(i * 255 for i in self._color[:3])

    def update_track(self, glob_feat_vec: list, bbox, loc_feat_vec: list = None):
        self._frames_since_last_update = 0
        self.set_active_global_feature_vector(glob_feat_vec)
        self.set_active_local_feature_vector(loc_feat_vec)
        self.set_active_bbox(bbox)

    def update_time(self):
        # motion prediction needs to go here.
        self._frames_since_last_update += 1

    def get_time(self):
        return self._frames_since_last_update

    def set_active_bbox(self, bbox):
        self._active_bbox = bbox

    def get_active_bbox(self):
        return self._active_bbox

    def set_active_global_feature_vector(self, feat_vec):
        """ sets the active global feature vector """
        self._active_global_feature_vector = feat_vec

    def get_active_global_feature_vector(self):
        """ Returns the active global feature vector """
        # Deprecated
        return self._active_global_feature_vector

    def set_active_local_feature_vector(self, feat_vec):
        """ sets the active feature vector """
        if feat_vec is not None:
            self._active_local_feature_vector = self.average_features(feat_vec)

    def get_active_local_feature_vector(self):
        """ Returns the active feature vector """
        # Deprecated
        return self._active_local_feature_vector

    def average_features(self, feats):
        self.list_local_feat_vec.append(feats)
        if len(self.list_local_feat_vec) > self.N_average:
            del self.list_local_feat_vec[0]
        #    N = self.N_average
        # else:
        #    N = len(self.list_local_feat_vec)
        return np.mean(self.list_local_feat_vec, axis=0)

    def get_active_feature_vector(self):
        return self._active_global_feature_vector

    def set_track_status(self, status: bool):
        """ set active status of the track """
        self.status = status

    def get_track_status(self):
        """ Returns the active status of the track """
        return self.status

    def set_track_id(self, id: int):
        """ sets the active track_id"""
        self._track_id = id

    def get_track_id(self):
        """ Returns the active track id"""
        return self._track_id
