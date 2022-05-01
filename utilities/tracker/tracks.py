import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utilities.description import Description
from utilities.tracker.kalman_filter import KalmanFilter
import torch
import copy
# This track describes all the properties of a single track.
# For now, a track is only described by the deep feature representation of its last assigned detection:
# - Forward prediction in terms of a Kalman Filter
# - what if you make a crazy kalman prediction based on obtained features. Only represent a track by the deep features???
# - Average the feature vector over last N frames
# - Also incorporate probability maybe?
# - The features are in the desription class...


class track(Description):
    def __init__(self, assigned_id: int, descrip: Description, N_average=False, kalman=False, supper=True):
        # Can i do the description smarter??
        # Superpoint description??
        self._N_Activeness = 5
        # Colors
        self.N_COLORS = 30
        self.colors = matplotlib.colors.ListedColormap(
            plt.get_cmap('nipy_spectral')(np.linspace(0, 1, self.N_COLORS))).colors
        # print(self.colors)
        self.kalman = kalman
        self.N_average = N_average
        self.set_track_id(assigned_id)
        self.set_track_color(assigned_id)
        self.set_track_status(True)

        # Not really used for now. Can be used for RE-ID, currently encoded in the full description...
        self._frames_since_last_update = 0

        # Init descriptions
        self._bbox, self._global_features, self._local_features = descrip.get_all_description()
        self.sim_conf = []

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(self.xyah())

    def __call__(self):
        return self._global_features
        
        if self._frames_since_last_update > self._N_Activeness:
            return self._local_features
        else:
            return self._global_features

    def set_track_color(self, assigned_id: int):
        self._color = self.colors[assigned_id % self.N_COLORS]

    def get_track_color(self):
        return tuple(i * 255 for i in self._color[:3])

    def get_sim_conf(self):
        if len(self.sim_conf):
            percentage = 0.95
            avg_sim = np.sum(self.sim_conf)/len(self.sim_conf)
            return avg_sim * percentage
        return 0

    def set_sim_conf(self, sim_conf):
        self.sim_conf.append(sim_conf)

    def update_track(self, descrip: Description, sim):
        self._frames_since_last_update = 0
        # descriptors update
        self._bbox, self._global_features, self._local_features = descrip.get_all_description()
        self.set_sim_conf(sim)

        # kf update
        if self.kalman:
            self.mean, self.covariance = self.kf.update(
                self.mean, self.covariance, self.xyah())
            # bbox update
            self.set_bbox_no_c(self.mean_to_bbox())

    def active(self):
        # Track is inactive if not associated
        # Maybe add two different thresholds. One for drawing and one for association. Already have...
        # If not associated don't write it.
        if self._frames_since_last_update > self._N_Activeness:
            return False
        else:
            return True

    def update_time(self):
        # Applying kalman filter?
        if self.kalman:
            self.mean, self.covariance = self.kf.predict(
                self.mean, self.covariance)
            # Update bbox
            self.set_bbox_no_c(self.mean_to_bbox())
        self._frames_since_last_update += 1
        # Delete an element from sim_conf
        if len(self.sim_conf):
            del self.sim_conf[0]

    def mean_to_bbox(self):
        """ Without confidence though... """
        bbox = self.mean[:4].copy()
        bbox[2] *= bbox[3]
        bbox[:2] -= bbox[2:] / 2
        return bbox

    def get_time(self):
        return self._frames_since_last_update

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

    def average_features(self, feats):
        # DEPRECATED
        self.list_local_feat_vec.append(feats)
        if len(self.list_local_feat_vec) > self.N_average:
            del self.list_local_feat_vec[0]
        #    N = self.N_average
        # else:
        #    N = len(self.list_local_feat_vec)
        return np.mean(self.list_local_feat_vec, axis=0)


    def combine(self, lat, best):
        # Maybe change to matches and scores.
        # This is not good, maybe take the matching keypoints instead
        pred0 = {k: v[0].cpu().numpy() for k, v in lat.items()}
        pred1 = {k: v[0].cpu().numpy() for k, v in best.items()}
        kpts0, kpts1 = pred0['keypoints'], pred1['keypoints']
        scores0, scores1 = pred0['scores'], pred1['scores']
        descriptors0, descriptors1 = pred0['descriptors'], pred1['descriptors']
        # N_FREE = N_MAX - len(indices1)
        keypoints = np.append(kpts0, kpts1, axis=0)
        scores = np.append(scores0, scores1, axis=0)
        descriptors = np.append(descriptors0, descriptors1, axis=-1)
        # print(keypoints.shape)
        # Converting to tensors
        keypoints = torch.from_numpy(keypoints).float().to('cuda:0')
        descriptors = torch.from_numpy(descriptors).float().to('cuda:0')
        scores = torch.from_numpy(scores).float().to('cuda:0')
        img = pred0['image']
        img = np.reshape(pred0['image'], (img.shape[1], img.shape[2]))
        image = torch.from_numpy(img).float()[None, None].to('cuda:0')
        return {'keypoints': [keypoints], 'scores': [scores], 'descriptors': [descriptors], 'image': image} #this might be hella scuffed


    def get_new_feature_description(self, current_keypoints, pred):
        # Maybe change to matches and scores.
        # This is not good, maybe take the matching keypoints instead
        N_MAX = 1024
        pred0 = {k: v[0].cpu().numpy() for k, v in current_keypoints.items()}
        pred1 = {k: v[0].cpu().numpy() for k, v in pred.items()}
        # Retrieving current and last keypoints
        kpts0, kpts1 = pred0['keypoints'], pred1['keypoints1']
        scores0, scores1 = pred0['scores'], pred1['scores1']
        descriptors0, descriptors1 = pred0['descriptors'], pred1['descriptors1']

        matches, conf = pred1['matches1'], pred1['matching_scores1']
        valid = matches > -1
        mconf = conf[valid]
        mkpts1 = kpts1[valid]   
        mscores1 = scores1[valid]
        mdescriptors1 = descriptors1[...,valid]

        indices1 = (-mconf).argsort() # [:N_COLLECT]
        # N_FREE = N_MAX - len(indices1)
        keypoints = np.append(kpts0, mkpts1[indices1], axis=0)
        scores = np.append(scores0, mscores1[indices1], axis=0)
        descriptors = np.append(descriptors0, mdescriptors1[...,indices1], axis=-1)
        # print(keypoints.shape)
        # Converting to tensors
        keypoints = torch.from_numpy(keypoints).float().to('cuda:0')
        descriptors = torch.from_numpy(descriptors).float().to('cuda:0')
        scores = torch.from_numpy(scores).float().to('cuda:0')
        img = pred0['image']
        img = np.reshape(pred0['image'], (img.shape[1], img.shape[2]))
        image = torch.from_numpy(img).float()[None, None].to('cuda:0')
        return {'keypoints': [keypoints], 'scores': [scores], 'descriptors': [descriptors], 'image': image} #this might be hella scuffed

    def check_best_keypoints(self,  keypoints):
        # DEPRECATED FOR NOW. MIGHT BECOME RELEVANT LATER ON.
        # pred1 = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        #match conf
        # matches, mconf = pred1['matches1'], pred1['matching_scores1']
        # valid = matches > -1
        # mconf = mconf[valid]
        conf = keypoints['scores'][0].cpu().numpy()
        # conf = pred1['scores1']
        # maybe do scores here instead???
        if np.sum(conf) > self._best_conf:
            # Update the best keypoints
            self._best_conf = np.sum(conf)
            self._best_features = copy.deepcopy(keypoints)
