import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utilities.description_sup import Description
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
    def __init__(self, assigned_id: int, descrip: Description, frame_shape=None, bbox_shape=None, kalman=False):
        # frame_shape and bbox_shape is only used for the new superpoint thing...
        # Can i do the description smarter??
        # Superpoint description??
        self._N_Activeness = 5 # Change back in tracker_combined
        self._N_associations = 0
        self.percentage =  0.88 # 0.1
        self.sim_average =  10
        self.anno_kalman = 0 # minimum number of annnotations before kalman is applied...
        # Colors
        self.N_COLORS = 30
        self.colors = matplotlib.colors.ListedColormap(
            plt.get_cmap('nipy_spectral')(np.linspace(0, 1, self.N_COLORS))).colors
        # print(self.colors)
        self.kalman = kalman
        self.set_track_id(assigned_id)
        self.set_track_color(assigned_id)
        self.set_track_status(True)
        self._local_feature_list = []
        # if frame_shape is not None:
        bbox, kpts, glob_feats = descrip.get_all_description()
        super().__init__(bbox, kpts, frame_shape, bbox_shape, glob_feats)
        # Not really used for now. Can be used for RE-ID, currently encoded in the full description...
        self._frames_since_last_update = 0

        # Init descriptions
        # self._bbox, self._kpts, _ = descrip.get_all_description()
        self.sim_conf = []

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(self.xyah())
        self._bb_array = None

    def __call__(self):
        return self.get_global_features()

    def set_track_color(self, assigned_id: int):
        self._color = self.colors[assigned_id % self.N_COLORS]

    def set_local_features(self, local_features):
        self._local_feature_list.append(local_features)

    #def get_local_features(self):
    #    return self._local_feature_list[0]

    def get_N_associations(self):
        return self._N_associations

    def get_track_color(self):
        return tuple(i * 255 for i in self._color[:3])

    def get_sim_conf(self):
        if len(self.sim_conf):
            avg_sim = np.sum(self.sim_conf)/len(self.sim_conf)
            return avg_sim * self.percentage
        return 0

    def set_sim_conf(self, sim_conf):
        self.sim_conf.append(sim_conf)

    def get_inter_bb(self):
        temp = copy.copy(self._bb_array)
        self._bb_array = None
        return temp

    def interpolate_track(self, new_bbox):
        old_bbox = self._bbox
        new_bbox
        time  = self.get_time() - self._N_Activeness # You automatically have the track active for 5 frames after...
        if time>0:
            # for values:
            dif = np.asarray(new_bbox) - np.asarray(old_bbox)
            step = dif/time
            bb_array = []
            for i in range(time-1):
                bb_array.append(old_bbox+(i+1)*step)
            self._bb_array = bb_array

    def update_track(self, descrip: Description, sim):
        # Add something with RE-ID here interpolation.
        # descriptors update
        bbox, self._kpts, self._glob_feats = descrip.get_all_description()
        # self.set_local_features(local_features)
        self.set_sim_conf(sim)
        self._N_associations += 1

        if self.get_time() > self._N_Activeness: 
            self.interpolate_track(bbox)

        self._bbox = bbox


        self._frames_since_last_update = 0
        # kf update
        if self.kalman:# and self._N_associations>self.anno_kalman:
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
        if self.kalman:# and self._N_associations>self.anno_kalman:
            self.mean, self.covariance = self.kf.predict(
                self.mean, self.covariance)
            # Update bbox
            self.set_bbox_no_c(self.mean_to_bbox())
        self._frames_since_last_update += 1
        # Delete an element from sim_conf
        if len(self.sim_conf) > self.sim_average:
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

    def combine(self, lat, best): # maybe you can do something like this as well.
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
