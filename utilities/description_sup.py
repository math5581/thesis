import numpy as np
import copy
import torch
# Can possibly hold the features as well?
# Remove confidence from bbox!!

def int_or_0(val):
    if int(val) < 0:
        return 0
    else:
        return int(val)

# Can you implement a kalman filter + Camera movement compensation?????
class Description:
    def __init__(self, bbox, keypoints=None, frame_shape = None, bbox_shape = None, global_feat=None) -> None:
        """ 5-Dim bounding box, with t,l,w,h,c 
            features 1280 description
            feature_bbox 1280 description """
        self._bbox = np.asarray(bbox)
        self._frame_shape = frame_shape
        self._bbox_shape = bbox_shape
        self.set_keypoints(keypoints)
        self.set_glob_feat(global_feat)

    def __call__(self):
        """ Returns the description """
        # Change here to change the feature representation ;)
        return self.get_global_features()

    def get_all_description(self):
        return self.tlwh(), self._kpts, self._glob_feats

    def set_glob_feat(self, feat):
        self._glob_feats=feat
    
    def get_glob_feat(self):
        return self._glob_feats

    def set_bbox(self, bbox):
        """ 5-Dim bounding box, with t,l,w,h,c """
        self._bbox = np.asarray(bbox)

    def set_bbox_no_c(self, bbox):
        """ 4-Dim bounding box, with t,l,w,h"""
        # Little bad practice.
        self._bbox[:4] = bbox

    def get_keypoints(self):
        return self._kpts

    def set_keypoints(self, kpts): # No np.asarray for superpoint
        self._kpts = kpts
        # print(self._kpts)
        # self._kpts = kpts # np.asarray(feat)

    def get_global_features(self):
        return self.trans_keypoints_global(self._kpts)
    
    def trans_keypoints_global(self, kpts):
        kpts = copy.copy(kpts)
        int_bb = self.get_int_bbox(self.tlwh(), (self._frame_shape[1], self._frame_shape[0]))
        # tl = np.asarray([self.get_int_bbox(self.tlwh(), self._frame_shape)[1], self.get_int_bbox(self.tlwh(), self._frame_shape)[0]])
        relative_0 = int_bb[2]/self._bbox_shape[0] 
        relative_1 = int_bb[3]/self._bbox_shape[1]
        arr = np.asarray([relative_0, relative_1])
        kpts_cpu = kpts['keypoints'][0].cpu().numpy()
        #print(relative_0, relative_1)
        #print(kpts_cpu*arr)
        kpts_cpu = np.asarray([int_bb[0], int_bb[1]]) + kpts_cpu*arr
        # print(kpts_cpu)
        kpts['keypoints'] = [torch.from_numpy(kpts_cpu).float().to('cuda:0')]
        return kpts

    def tlwh(self):
        """ Returns top left width height and confidence """
        return self._bbox.copy()

    def xyah(self):
        """ Returns center x,y, alpha scaling, height and confidence bbox, """
        bbox = self._bbox.copy()
        bbox[:2] += bbox[2:4] / 2
        bbox[2] /= bbox[3]
        return bbox[:4]

    def tlbr(self):
        """ Returns top left, bottom right, confidence bbox, """
        bbox = self._bbox.copy()[:4]
        bbox[2:4] += bbox[:2]
        return bbox

    @staticmethod
    def get_bbox_center(bbox):
        return np.asarray([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]])


    @staticmethod
    def get_int_bbox(bbox, dim):
        """ Returns int bbox from dimensions """
        return [int_or_0(bbox[0] * dim[0]), int_or_0(bbox[1] * dim[1]),
                int_or_0(bbox[2] * dim[0]), int_or_0(bbox[3] * dim[1]), bbox[4]]