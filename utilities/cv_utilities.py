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
    def __init__(self, DIM, KERNEL_SIZE=(15, 15), std_blur=5):
        self._DIMENSIONS = DIM
        self._KERNEL_SIZE = KERNEL_SIZE
        self.std_blur = std_blur
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

    # enlargens the bbox here. but with what?
    def blur_image_list_except_bbox_large(self, bbox_list):
        return [self.blur_image_except_bbox_large(bbox) for bbox in bbox_list]

    def blur_image_except_bbox_large(self, bbox):
        target_h = 0.5
        """ Returns blurred frame except bbox """
        if bbox[3] < target_h:
            frame_full = self._frame_full.copy()
            h, w, _ = frame_full.shape
            bbox_full = self.get_int_bbox(bbox, (w, h))
            roi_full = self.crop_bounding_box(bbox_full, frame_full)
            # finding the new box:
            w, h = self._DIMENSIONS
            target_h_int = int(h * target_h)
            target_w_int = int(target_h_int * bbox_full[2] / bbox_full[3])

            bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
            # Finding offsets:
            dx = int(bbox[0] + target_w_int / 2 - (bbox[0] + bbox[2] / 2))
            dy = int(bbox[1] + target_h_int / 2 - (bbox[1] + bbox[3] / 2))

            # handling for out of bounds.
            if (bbox[1] - dy) < 0:
                target_h_int += (bbox[1] - dy)
                dy = 0

            if (bbox[0] - dx) < 0:
                # Change width
                target_w_int += (bbox[0] - dx)
                dx = 0

            if bbox[1] + target_h_int - dy > self._DIMENSIONS[1]:
                # Shrink height
                target_h_int = - (bbox[1] - dy - self._DIMENSIONS[1])

            if bbox[0] + target_w_int - dx > self._DIMENSIONS[0]:
                # shrink width
                target_w_int = - (bbox[0] - dx - self._DIMENSIONS[0])

            # print(target_w_int, target_h_int)
            #if target_w_int == 0 or target_h_int == 0:
            #    return self._blured_frame.copy()

            roi = cv2.resize(roi_full, (target_w_int, target_h_int))
            blured_frame = self._blured_frame.copy()
            blured_frame[bbox[1] - dy: bbox[1] + target_h_int - dy,
                         bbox[0] - dx: bbox[0] + target_w_int - dx] = roi
            return blured_frame
        else:
            frame = self._frame.copy()
            bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
            roi = self.crop_bounding_box(bbox, frame)
            blured_frame = self._blured_frame.copy()
            blured_frame[bbox[1]: bbox[1] + bbox[3],
                        bbox[0]: bbox[0] + bbox[2]] = roi
        return blured_frame

    def black_image_list_except_bbox(self, bbox_list):
        return [self.black_image_except_bbox(bbox) for bbox in bbox_list]

    def black_image_except_bbox(self, bbox):
        """ Returns blurred frame except bbox """
        frame = self._frame.copy()
        bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
        roi = self.crop_bounding_box(bbox, frame)
        black_frame = np.zeros(self._DIMENSIONS + (3,), np.uint8)
        black_frame[bbox[1]: bbox[1] + bbox[3],
                    bbox[0]: bbox[0] + bbox[2]] = roi
        return black_frame
    # enlargens the bbox here. but with what?

    def black_image_list_except_bbox_large(self, bbox_list):
        return [self.black_image_except_bbox_large(bbox) for bbox in bbox_list]

    def black_image_except_bbox_large(self, bbox):
        target_h = 0.5
        """ Returns blurred frame except bbox """
        if bbox[3] < target_h:
            frame_full = self._frame_full.copy()
            h, w, _ = frame_full.shape
            bbox_full = self.get_int_bbox(bbox, (w, h))
            roi_full = self.crop_bounding_box(bbox_full, frame_full)
            # finding the new box:
            w, h = self._DIMENSIONS
            target_h_int = int(h * target_h)
            target_w_int = int(target_h_int * bbox_full[2] / bbox_full[3])

            bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
            # Finding offsets:
            dx = int(bbox[0] + target_w_int / 2 - (bbox[0] + bbox[2] / 2))
            dy = int(bbox[1] + target_h_int / 2 - (bbox[1] + bbox[3] / 2))

            # handling for out of bounds.
            if (bbox[1] - dy) < 0:
                target_h_int += (bbox[1] - dy)
                dy = 0

            if (bbox[0] - dx) < 0:
                # Change width
                target_w_int += (bbox[0] - dx)
                dx = 0

            if bbox[1] + target_h_int - dy > self._DIMENSIONS[1]:
                # Shrink height
                target_h_int = - (bbox[1] - dy - self._DIMENSIONS[1])

            if bbox[0] + target_w_int - dx > self._DIMENSIONS[0]:
                # shrink width
                target_w_int = - (bbox[0] - dx - self._DIMENSIONS[0])

            # print(target_w_int, target_h_int)
            #if target_w_int == 0 or target_h_int == 0:
            #    return self._blured_frame.copy()

            roi = cv2.resize(roi_full, (target_w_int, target_h_int))
            black_frame = np.zeros(self._DIMENSIONS + (3,), np.uint8)
            black_frame[bbox[1] - dy: bbox[1] + target_h_int - dy,
                         bbox[0] - dx: bbox[0] + target_w_int - dx] = roi
            return black_frame
        else:
            frame = self._frame.copy()
            bbox = self.get_int_bbox(bbox, self._DIMENSIONS)
            roi = self.crop_bounding_box(bbox, frame)
            black_frame = np.zeros(self._DIMENSIONS + (3,), np.uint8)
            black_frame[bbox[1]: bbox[1] + bbox[3],
                        bbox[0]: bbox[0] + bbox[2]] = roi
        return black_frame

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

    def extract_bbox_full(self, bbox):
        frame = self.get_current_frame_full()
        bbox = self.get_int_bbox(bbox, (frame.shape[1], frame.shape[0]))
        bbox = self.crop_bounding_box(bbox, frame)
        return bbox


    def extract_bbox_full_rotate(self, bbox):
        return cv2.rotate(self.extract_bbox_full(bbox), cv2.ROTATE_90_COUNTERCLOCKWISE)

    def extract_bbox_from_list(self, bbox_list, full = False, rotate = False):
        if full and rotate: 
            return [self.extract_bbox_full_rotate(bbox) for bbox in bbox_list]
        elif full:
            return [self.extract_bbox_full(bbox) for bbox in bbox_list]
        else:
            return [self.extract_bbox(bbox) for bbox in bbox_list]

    # Blurring Methods
    def gaussian_blur(self, frame, kernel_size):
        return cv2.GaussianBlur(frame, kernel_size, self.std_blur)

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
