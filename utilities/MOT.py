import os
import numpy as np
import cv2
from utilities.utilities import list_file_paths_in_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import configparser

# Helper Functions


class MOTDataloader():
    def __init__(self, base_path, DET_THRESHOLD=0.5, zebrafish=False, mot_synth=False, gt_path=None, seq=None):
        self.base_path = base_path
        self.zebrafish = zebrafish
        self.mot_synth = mot_synth
        # Use imgDir
        self.frames_base_path = os.path.join(base_path, 'img1')
        self.det_file_path = os.path.join(base_path, 'det', 'det.txt')
        if not self.zebrafish:
            self.gt_file_path = os.path.join(base_path, 'gt', 'gt.txt')
        else:
            self.gt_file_path = os.path.join(base_path, 'gt1', 'gt.txt')

        if gt_path is not None:
            self.gt_file_path = os.path.join(gt_path, 'gt', 'gt.txt')
        # Read configs
        self.config = configparser.ConfigParser()
        if gt_path is not None:
            self.config.read(os.path.join(gt_path, 'seqinfo.ini'))
        else:
            self.config.read(os.path.join(base_path, 'seqinfo.ini'))

        self.gt_df = self.load_gt_file_as_pd()

        self.occlusion_level = 0.5
        # Read detection and gt files
        if not self.mot_synth:
            self.files = self.get_files_in_section()
        else:
            self.video_path = os.path.join(base_path, seq + '.mp4')
            self.open_video_file()
        if not self.zebrafish and not self.mot_synth:
            self.det_df = self.load_detection_file_as_pd()
        self._current_frame_id = 1

        self.DET_THRESHOLD = DET_THRESHOLD


    def next_frame(self):
        if self._current_frame_id < self.get_seuqence_length():
            self._current_frame_id += 1
            return True
        else:
            print('max frames reached')
            return False

    def get_dimensions(self):
        return int(self.config['Sequence']['imWidth']), int(self.config['Sequence']['imHeight'])

    def prev_frame(self):
        if self._current_frame_id <= 1:
            print('at starting frame ')
            return False
        else:
            self._current_frame_id -= 1
            return True

    def get_seuqence_length(self):
        return int(self.config['Sequence']['seqLength'])

    def get_current_frame_id(self):
        # Change motsynth
        return self._current_frame_id

    def set_current_frame_id(self, id):
        if id <= self.get_seuqence_length():
            self._current_frame_id = id


    def get_current_frame(self):
        # corresponds to -1 in files list
        if not self.mot_synth:
            return cv2.imread(self.files[self._current_frame_id - 1])
        else:
            self._cap.set(1, self._current_frame_id - 1)
            ret, frame = self._cap.read()
            if ret:
                return frame
            return 'Error'

    def get_current_gt_bbox_int(self):
        return self.get_int_bbox(self.gt_df)

    def get_current_det_bbox_int(self):
        return self.get_int_bbox(self.det_df)

    def get_current_gt_bbox_scale(self):
        return self.get_scale_bbox(self.gt_df)

    def get_current_det_bbox_scale(self):
        return self.get_scale_bbox_det(self.det_df)

    def get_files_in_section(self):
        return list_file_paths_in_dir(self.frames_base_path)

    def open_video_file(self):
        self._cap = cv2.VideoCapture(self.video_path)

    def load_detection_file_as_pd(self):
        """ Loads detection as dataframe """
        df = pd.read_csv(self.det_file_path, names=[
            'frame_id', 'id', 'x', 'y', 'w', 'h', 'c', 'class', 'occlusion', 'what'])
        df.drop(['id', 'what'], inplace=True, axis=1)
        return df

    def load_gt_file_as_pd(self):
        """ Loads ground-truth as dataframe """
        if self.mot_synth:
            df = pd.read_csv(self.gt_file_path, names=[
                'frame_id', 'id', 'x', 'y', 'w', 'h', 'c', 'class', 'visibility', 'x_r', 'y_r', 'z_r'])
            df.drop(['x_r', 'y_r', 'z_r'], inplace=True, axis=1)
            df = df[df['class'] == 1]
            return df
        if not self.zebrafish:
            # Only loads pedestrian annotations.
            df = pd.read_csv(self.gt_file_path, names=[
                'frame_id', 'id', 'x', 'y', 'w', 'h', 'c', 'class', 'visibility', 'what'])
            df.drop(['what'], inplace=True, axis=1)
            df = df[df['class'] == 1]
            return df


        df = pd.read_csv(self.gt_file_path, names=[
            'frame_id', 'id', '3d_x', '3d_y', '3d_z', 'camT_x', 'camT_y', 'camT_left', 'camT_top', 'camT_width', 'camT_height',
            'camT_occlusion', 'camF_x', 'camF_y', 'x', 'y', 'w', 'h', 'occlusion'])
        df.drop(['3d_x', '3d_y', '3d_z', 'camF_x', 'camF_y', 'camT_x', 'camT_y', 'camT_left', 'camT_top', 'camT_width', 'camT_height',
                 'camT_occlusion'], inplace=True, axis=1)
        return df

    def get_int_bbox(self, df):
        """Returns list of bounding boxes(int) from frame_id  and df"""
        return df[df['frame_id'] == self._current_frame_id][[
            'x', 'y', 'w', 'h', 'c']].values.tolist()

    def get_scale_bbox(self, df):
        """Returns list of bounding boxes(float) from frame_id/dataframe and an ID list"""
        w = int(self.config['Sequence']['imWidth'])
        h = int(self.config['Sequence']['imHeight'])
        if self.zebrafish:
            bbox_list = df[df['frame_id'] == self._current_frame_id][[
                'x', 'y', 'w', 'h', 'occlusion']].values.tolist()
        else:
            bbox_list = df[df['frame_id'] == self._current_frame_id][[
                'x', 'y', 'w', 'h', 'c', 'visibility']].values.tolist()
        temp = []
        idx_remove = []
        for index, bbox in enumerate(bbox_list):
            if self.zebrafish:
                if bbox[4]:
                    b_list = bbox_list.copy()
                    b_list.pop(index)
                    visibility = self.get_min_visibility(bbox, b_list)
                    if visibility > self.occlusion_level:
                        temp.append([bbox[0] / w, bbox[1] / h,
                                     bbox[2] / w, bbox[3] / h, bbox[4]])
                else:
                    temp.append([bbox[0] / w, bbox[1] / h,
                                 bbox[2] / w, bbox[3] / h, bbox[4]])
            else:
                if bbox[5] > self.occlusion_level:
                    temp.append([bbox[0] / w, bbox[1] / h,
                                 bbox[2] / w, bbox[3] / h, bbox[4]])
                else:
                    idx_remove.append(index)
        id_list = df[df['frame_id'] == self._current_frame_id][[
            'id']].values.tolist()
        for index in sorted(idx_remove, reverse=True):
            del id_list[index]
        return temp, id_list

    def get_min_visibility(self, bbox, bbox_list):
        max_iou = 0
        for bb in bbox_list:
            iou = self.get_iou(bbox[:4], bb[:4])
            if iou > max_iou:
                max_iou = iou
        return 1-max_iou

    @staticmethod
    def get_iou(boxA, boxB):
        boxA, boxB = np.asarray(boxA), np.asarray(boxB)
        boxA[2:4] += boxA[:2]
        boxB[2:4] += boxB[:2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_scale_bbox_det(self, df):
        """Returns list of bounding boxes(float) from frame_id/dataframe and an ID list"""
        w = int(self.config['Sequence']['imWidth'])
        h = int(self.config['Sequence']['imHeight'])
        bbox_list = df[df['frame_id'] == self._current_frame_id][[
            'x', 'y', 'w', 'h', 'c']].values.tolist()
        temp = []
        for bbox in bbox_list:
            if bbox[4] > self.DET_THRESHOLD:
                temp.append([bbox[0] / w, bbox[1] / h,
                             bbox[2] / w, bbox[3] / h, bbox[4]])
        return temp
