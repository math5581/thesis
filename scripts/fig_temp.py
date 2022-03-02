import sys
sys.path.insert(0, '/Users/mathiaspoulsen/Desktop/semester10/thesis') 
from utilities.cv_utilities import *
from utilities.MOT import * 
BASE_PATH_MOT = '../MOT15/train/ADL-Rundle-6'

dataloader = MOTDataloader(BASE_PATH_MOT)
dim = dataloader.get_dimensions()
dataloader.set_current_frame_id(80)
cv_tools = CVTools(dim, (51, 51))

frame = dataloader.get_current_frame()
cv_tools.set_active_frame(frame)
bbox_list, id_list  = dataloader.get_current_gt_bbox_scale()

frames = cv_tools.blur_image_list_except_bbox(bbox_list)

bbox_list= dataloader.get_current_gt_bbox_int()
for frame, bbox in zip(frames, bbox_list):
    if id != 22:
        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], 
                            bbox[1]+bbox[3]), (255,255,255), 4)


for idx, frame in enumerate(frames):
    cv2.imwrite(str(idx) + '.jpg', frame)
