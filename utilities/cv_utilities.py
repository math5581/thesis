import cv2
import numpy as np


def load_image_file(img_file: str):
    return cv2.imread(img_file)


def save_image_file(file_name, frame):
    cv2.imwrite(file_name, frame)


def int_or_0(val):
    if int(val) < 0:
        return 0
    else:
        return int(val)


class CVTools:
    def __init__(self, DIM, KERNEL_SIZE=(31, 31)):
        self._DIMENSIONS = DIM
        self._KERNEL_SIZE = KERNEL_SIZE
        self._blur_method = self.gaussian_blur  # Default
        self._frame = None
        self._frame_full = None
        self._blured_frame = None

    def get_current_frame(self):
        return self._frame.copy()

    def get_current_frame_full(self):
        return self._frame_full.copy()

    def set_blur_method(self, blur_method):
        self._blur_method = blur_method

    def set_kernel_size(self, kernel_size):
        self._KERNEL_SIZE = kernel_size

    def set_active_frame(self, frame):
        """ Sets active frame to perform manipulation on"""
        self._frame_full = frame
        self._frame = self.resize_frame(frame)
        self._blured_frame = self._blur_method(
            self._frame.copy(), self._KERNEL_SIZE)

    # Different blur methods
    def blur_image_list_except_bbox(self, bbox_list):
        return [self.blur_image_except_bbox(bbox) for bbox in bbox_list]

    def blur_image_except_bbox(self, bbox):
        """ Returns blurred frame except bbox """
        frame = self._frame.copy()
        bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
        roi = self.crop_bounding_box(bbox, frame)
        blured_frame = self._blured_frame.copy()
        blured_frame[bbox[1]: bbox[1] + bbox[3],
                     bbox[0]: bbox[0] + bbox[2]] = roi
        return blured_frame

    def blur_image_except_bbox_black(self, bbox):
        """ Returns blurred frame except black bbox """
        bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
        blured_frame = self._blured_frame.copy()
        blured_frame[bbox[1]: bbox[1] + bbox[3],
                     bbox[0]: bbox[0] + bbox[2]] = 0
        return blured_frame

    def blur_image_list_except_bbox_newbb(self, bbox_list, new_bbox):
        return [self.blur_image_except_bbox_newbb(bbox, new_bbox[i]) for i, bbox in enumerate(bbox_list)]

    def blur_image_except_bbox_newbb(self, bbox, new_bbox):
        """ Returns blurred frame except bbox and crops new_bbox on top """
        frame = self._frame.copy()
        bbox = self.get_int_bbox(bbox, self._DIMENSIONS)

        roi = self.crop_bounding_box(bbox, frame)
        new_roi = self.get_roi(new_bbox, frame)
        blured_frame = self._blured_frame.copy()
        try:
            # Used to catch errors about the roi
            blured_frame[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] +
                         bbox[2]] = cv2.resize(new_roi, (roi.shape[1], roi.shape[0]))
        except:
            pass
        return blured_frame

    # BBOX methods
    def extract_bbox(self, bbox):
        frame = self.get_current_frame_full()
        bbox = self.get_int_bbox(bbox, (frame.shape[1], frame.shape[0]))
        bbox = self.crop_bounding_box(bbox, frame)
        return cv2.resize(bbox, self._DIMENSIONS, interpolation=cv2.INTER_AREA)

    def extract_bbox_from_list(self, bbox_list):
        return [self.extract_bbox(bbox) for bbox in bbox_list]

    # Blurring Methods
    def gaussian_blur(self, frame, kernel_size):
        return cv2.GaussianBlur(frame, kernel_size, 0)

    def convolution_blurring(self, frame, kernel_size, param=25):  # Not used yet
        kernel = np.ones(kernel_size, np.float32) / param
        return cv2.filter2D(frame, -1, kernel)

    # ROI/BBox
    def get_roi(self, bbox, frame):
        bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
        return self.crop_bounding_box(bbox, frame)

    @staticmethod
    def get_int_bbox(bbox, dim):
        """ Returns int bbox from dimensions """
        return [int_or_0(bbox[0] * dim[0]), int_or_0(bbox[1] * dim[1]),
                int_or_0(bbox[2] * dim[0]), int_or_0(bbox[3] * dim[1]), bbox[4]]

    def crop_bounding_box(self, bbox, frame):
        roi = frame[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        return roi

    def resize_frame(self, frame):
        return cv2.resize(frame, self._DIMENSIONS, interpolation=cv2.INTER_AREA)

    def draw_bbox_from_list(self, bbox_list):
        """ Draws bbox from list """
        frame = self.get_current_frame()
        [self.draw_bbox(bbox[i], frame) for i, bbox in enumerate(bbox_list)]

    def draw_bbox(self, bbox, frame):
        """ Draws single bbox"""
        self.get_int_bbox
