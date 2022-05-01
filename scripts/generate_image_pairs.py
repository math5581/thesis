""" Script used to generate image pairs for siamese network """
from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.utilities import *
import random
import time

DIMENSIONS = (480, 480)
# possibly expand with sequences
base_path = '/workspace/data/MOTSynth_3'
annotation_base_path = '/workspace/data/mot_annotations'
l = os.listdir(base_path)
l.sort()
SEQUENCES = [x.split('.')[0] for x in l]




# What do you want??
# image pairs with N=5 distance
# Image pairs with N=10 distance
# Image pairs with N=20 distance
# Image pairs with N=30 distance
# Image pairs with N=60 distance
NS = [5, 15, 30, 60]

SAMPLES_PER_SEQUENCE = 20

output_path = '/workspace/data/siamese/' + 'mot_synth_blur_larger'
anchor = os.path.join(output_path, 'anchor')
positive = os.path.join(output_path, 'positive')
negative = os.path.join(output_path, 'negative')
os.makedirs(anchor, exist_ok=True)
os.makedirs(positive, exist_ok=True)
os.makedirs(negative, exist_ok=True)
cv_tools = CVTools(DIMENSIONS)


def get_data(dataloader, id):
    dataloader.set_current_frame_id(id)
    frame = dataloader.get_current_frame()
    cv_tools.set_active_frame(frame)
    bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
    frame_list = cv_tools.blur_image_list_except_bbox_large(bbox_list)
    return id_list, frame_list


def get_all_pairs(dataloader, start_id, end_id):
    id_list_1, frame_list_1 = get_data(dataloader, start_id)
    id_list_2, frame_list_2 = get_data(dataloader, end_id)

    anchor_temp_list = []
    positive_temp_list = []
    negative_temp_list = []

    match = False
    for index_1, id in enumerate(id_list_1):
        if len(id_list_2) != 1:
            if id in id_list_2:
                match = True
                index_2 = id_list_2.index(id)
                anch = frame_list_1[index_1]
                pos = frame_list_2[index_2]
            if match:
                # Put in a randomly selected negative sample
                match = False
                while True:
                    wrong_id = random.choice(id_list_2)
                    if wrong_id != id:
                        index_2 = id_list_2.index(wrong_id)
                        neg = frame_list_2[index_2]
                        break
                anchor_temp_list.append(anch)
                positive_temp_list.append(pos)
                negative_temp_list.append(neg)

    return anchor_temp_list, positive_temp_list, negative_temp_list

def save_images(start_id, anch_list, pos_list, neg_list):
    for anch, pos, neg in zip(anch_list, pos_list, neg_list):
        cv2.imwrite(os.path.join(anchor, str(start_id) + '.png'), anch)
        cv2.imwrite(os.path.join(positive, str(start_id) + '.png'), pos)
        cv2.imwrite(os.path.join(negative, str(start_id) + '.png'), neg)
        start_id += 1
    return start_id

def extract_images(SEQUENCE, img_id):
    """ Extracts similarity between GT """
    # seq = os.path.join(base_path, SEQUENCE)
    annotation = os.path.join(annotation_base_path, SEQUENCE)

    dataloader = MOTDataloader(base_path, mot_synth=True, gt_path=annotation, seq=SEQUENCE)

    # bboxes = cv_tools.extract_bbox_from_list(bbox_list)
    sequence_length = dataloader.get_seuqence_length()
    samples = 0
    while True:
        for n in NS:
            id_1 = random.choice(range(sequence_length - n))
            id_2 = id_1 + n
            anchor_list, positive_list, negative_list = get_all_pairs(dataloader, id_1, id_2)
            img_id = save_images(img_id, anchor_list, positive_list, negative_list)
            samples += len(anchor_list)
        if samples > SAMPLES_PER_SEQUENCE:
            break
    
    return img_id

if __name__ == '__main__':
    img_id = 0
    for seq in SEQUENCES:
        print(seq)
        if int(seq) != 759:
            img_id = extract_images(seq, img_id)