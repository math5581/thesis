import numpy as np
import copy
# Can possibly hold the features as well?
# Remove confidence from bbox!!

class Description:
    def __init__(self, bbox, global_features=None, local_features=None) -> None:
        """ 5-Dim bounding box, with t,l,w,h,c 
            features 1280 description
            feature_bbox 1280 description """
        self._bbox = np.asarray(bbox)
        self.set_global_features(global_features)
        self.set_local_features(local_features)

    def __call__(self):
        """ Returns the description """
        # Change here to change the feature representation ;)
        return self._global_features

    def get_all_description(self):
        return self.tlwh(), self.get_global_features(), self.get_local_features()

    def set_bbox(self, bbox):
        """ 5-Dim bounding box, with t,l,w,h,c """
        self._bbox = np.asarray(bbox)

    def set_bbox_no_c(self, bbox):
        """ 4-Dim bounding box, with t,l,w,h"""
        # Little bad practice.
        self._bbox[:4] = bbox

    def set_global_features(self, feat):
        self._global_features = np.asarray(feat)

    def set_local_features(self, feat):
        self._local_features = feat

    def get_global_features(self):
        return copy.deepcopy(self._global_features)

    def get_local_features(self):
        return copy.deepcopy(self._local_features)

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
