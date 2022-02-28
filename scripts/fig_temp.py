from utilities.cv_utilities import *
from utilities.MOT import * 

BASE_PATH_MOT = '../MOT15/train/ADL-Rundle-6'



dataloader = MOTDataloader(BASE_PATH_MOT)
dataloader.get
cv_tools = CVTools(DIMENSIONS)
feature_extraction = FeatureDescription(DIMENSIONS)