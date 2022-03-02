import os
import numpy as np
import cv2
from utilities.utilities import list_file_paths_in_dir
import pandas as pd
import configparser

# Helper Functions


class MOTDataloader():
    def __init__(self, base_path, DET_THRESHOLD=0.5):
        self.base_path = base_path
        self.frames_base_path = os.path.join(base_path, 'img1')
        self.det_file_path = os.path.join(base_path, 'det', 'det.txt')
        self.gt_file_path = os.path.join(base_path, 'gt', 'gt.txt')

        # Read configs
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(base_path, 'seqinfo.ini'))

        # Read detection and gt files
        self.files = self.get_files_in_section()
        self.det_df = self.load_detection_file_as_pd()
        self.gt_df = self.load_gt_file_as_pd()

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
        return self._current_frame_id

    def set_current_frame_id(self, id):
        if id <= self.get_seuqence_length():
            self._current_frame_id = id

    def get_current_frame(self):
        # corresponds to -1 in files list
        return cv2.imread(self.files[self._current_frame_id - 1])

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

    def load_detection_file_as_pd(self):
        """ Loads detection as dataframe """
        df = pd.read_csv(self.det_file_path, names=[
            'frame_id', 'id', 'x', 'y', 'w', 'h', 'c', 'x1', 'y1', 'z1'])
        df.drop(['id', 'x1', 'y1', 'z1'], inplace=True, axis=1)
        return df

    def load_gt_file_as_pd(self):
        """ Loads ground-truth as dataframe """
        df = pd.read_csv(self.gt_file_path, names=[
            'frame_id', 'id', 'x', 'y', 'w', 'h', 'c', 'x1', 'y1', 'z1'])
        df.drop(['x1', 'y1', 'z1'], inplace=True, axis=1)
        return df

    def get_int_bbox(self, df):
        """Returns list of bounding boxes(int) from frame_id  and df"""
        return df[df['frame_id'] == self._current_frame_id][[
            'x', 'y', 'w', 'h', 'c']].values.tolist()

    def get_scale_bbox(self, df):
        """Returns list of bounding boxes(float) from frame_id/dataframe and an ID list"""
        w = int(self.config['Sequence']['imWidth'])
        h = int(self.config['Sequence']['imHeight'])
        bbox_list = df[df['frame_id'] == self._current_frame_id][[
            'x', 'y', 'w', 'h', 'c']].values.tolist()
        temp = []
        for bbox in bbox_list:
            temp.append([bbox[0] / w, bbox[1] / h,
                         bbox[2] / w, bbox[3] / h, bbox[4]])
        id_list = df[df['frame_id'] == self._current_frame_id][[
            'id']].values.tolist()
        return temp, id_list

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
