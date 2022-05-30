import torch

from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.utilities import *
from utilities.description_sup import Description

from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import (process_resize, frame2tensor)

torch.set_grad_enabled(False)

# Seperate this...
class SuperInterface(Matching):
    # Load the SuperPoint and SuperGlue models.
    def __init__(self, dimensions):
        config = {
            'superpoint': {
                'nms_radius': 2, # default 4
                'keypoint_threshold': 0.005, #default 0.005
                'max_keypoints': 512 # default 1024
            },
            'superglue': {
                'weights': 'outdoor',#or indoor
                'sinkhorn_iterations': 20, #default 20
                'match_threshold': 0.2, # default Maybe try to adjust, default is 0.2
            }
        }
        super().__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        self.eval().to(self.device)
        self.dimensions = dimensions

    def __call__(self, pred0, pred1, percentage=False):
        pred = {}
        pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        pred = {**pred, **super().__call__(pred)}
        if percentage:
            return self.post_process_percentage(pred), pred # Try with average here as well.
        else:
            return self.post_process(pred), pred # Try with average here as well.

    def prepare_img(self, image, device):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is None:
            print('Problem with image')
            exit(1)
        # Convert to grayscale.
        w, h = image.shape[1], image.shape[0]
        w_new, h_new = process_resize(w, h, self.dimensions)
        scales = (float(w) / float(w_new), float(h) / float(h_new))
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
        # cv2.imwrite('test.png', image)
        inp = frame2tensor(image, device)
        return image, inp, scales

    def run_superpoint(self, img):
        img, inp, scale = self.prepare_img(img, self.device)
        #avg_scale = (scale[0]+scale[1]) / 4
        #nms = int(np.round(self.nms * avg_scale))
        pred0 = self.superpoint({'image': inp})
        pred = {}
        #pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        pred = {**pred, **{k: v for k, v in pred0.items()}}
        pred = {**pred, **{'image': inp}}
        return pred
        
    def get_description(self, bboxes=None, bbox_list=None, frame_shape= None, bbox_shape = None):
        """ Returns the feature description for the current detection """
        if len(bboxes) == 0:
            return []
        keypoints = []
        for bbox in bboxes:
            keypoints.append(self.run_superpoint(bbox))
            # modify the keypoints here:
        description_arr = []
        for keypoint, bbox in zip(keypoints, bbox_list):
            description_arr.append(Description(bbox, keypoint, frame_shape, bbox_shape))
        return description_arr

    def post_process_percentage(self, pred):
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mconf = conf[valid]
        N_kpts = (kpts0.shape[0] + kpts1.shape[0])/2
        return np.sum(mconf) / N_kpts # * 100 #, mkpts1

    def post_process(self, pred):
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        return np.sum(mconf) #, mkpts1

    @staticmethod
    def get_numb_kpts(pred):
        return pred['keypoints'][0].cpu().numpy().shape[0]

    @staticmethod
    def get_match_number(pred):
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        matches, conf = pred['matches0'], pred['matching_scores0']
        # Write the matches to disk.
        out_matches = {'matches': matches, 'match_confidence': conf}
        # Keep the matching keypoints.
        valid = matches > -1
        mconf = conf[valid]
        return len(mconf)
    
    @staticmethod
    def get_match_score(pred):
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        matches, conf = pred['matches0'], pred['matching_scores0']
        # Write the matches to disk.
        out_matches = {'matches': matches, 'match_confidence': conf}
        # Keep the matching keypoints.
        # print(len(matches))
        valid = matches > -1
        mconf = conf[valid]
        return np.sum(mconf)

