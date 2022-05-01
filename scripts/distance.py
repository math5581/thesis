""" Script used to extract the various distance metrics:"""
from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.feature_extraction import FeatureDescription, cosine_similarity, eucledian_distance
from utilities.utilities import *
import matplotlib.pyplot as plt
import tensorflow as tf

# Specify_params here:
DIMENSIONS = (224, 224)
# possibly expand with sequences
base_path = '/workspace/data/MOT17/train/'
SEQUENCES = ['MOT17-02-FRCNN', 
             # 'MOT17-04-FRCNN',
             'MOT17-05-FRCNN', 
            'MOT17-13-FRCNN']
metric = 'euclidian'  # cosine

metric_to_extract = cosine_similarity

def get_bbox_center(bbox):
    return np.asarray([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2])

def get_distance(c1, c2):
    return np.sqrt(np.sum(np.square(np.asarray(c1)-np.asarray(c2))))

def get_bbox_size(bb1, bb2):
    return bb1[2] * bb1[3] + bb2[2] * bb2[3]


feature_extraction = FeatureDescription(DIMENSIONS)


def extract_similarity_gt(SEQUENCE):
    """ Extracts similarity between GT """
    seq = os.path.join(base_path, SEQUENCE)
    dataloader = MOTDataloader(seq)
    cv_tools = CVTools(DIMENSIONS)

    similarity_vector = []
    similarity_vector_avg = []
    similarity_vector_max = []
    bbox_size_vector = []
    avg_bbox_size_vector = []
    sequence_length = int(dataloader.get_seuqence_length())
    for i in range(sequence_length):
        # Try skipping every fifth?
        # print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)
        # print(id_list)
        # bboxes = cv_tools.extract_bbox_from_list(bbox_list)
        frame_list = cv_tools.extract_bbox_from_list(bbox_list)
        #for i, frame in enumerate(frame_list):
        #    cv2.imwrite('test'+str(i)+'.png', frame)

        detection_array = feature_extraction.get_detection_features(
            frame_list, bbox_list=bbox_list)
        features = [description() for description in detection_array]
        # print(len(features))
        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                # GTs
                if id in prev_id_list:
                    prev_index = prev_id_list.index(id)
                    similarity_vector.append(metric_to_extract(
                        features[index], prev_feature_list[prev_index]))
                    bbox_size_vector.append(get_bbox_size(bbox_list[index], prev_bbox_list[prev_index]))
                # Not GTs
                temp_similarity = []
                for prev_index, prev_id in enumerate(prev_id_list):
                    if prev_id != id:
                        temp_similarity.append(metric_to_extract(
                            features[index], prev_feature_list[prev_index]))
                        similarity_vector_avg.append(metric_to_extract(
                            features[index], prev_feature_list[prev_index]))
                        avg_bbox_size_vector.append(get_bbox_size(bbox_list[index], prev_bbox_list[prev_index]))

                if len(temp_similarity) != 0:
                    # Max
                    min_value = max(temp_similarity)
                    similarity_vector_max.append(min_value)
        prev_bbox_list = bbox_list
        prev_id_list = id_list
        prev_feature_list = features
        dataloader.next_frame()

    #save_similarity_vector(bbox_size_vector, SEQUENCE + "_bbox.pkl")
    #save_similarity_vector(avg_bbox_size_vector, SEQUENCE + "_avg_bbox.pkl")
    save_similarity_vector(similarity_vector, SEQUENCE + "_gt.pkl")
    save_similarity_vector(similarity_vector_max, SEQUENCE + "_max.pkl")
    save_similarity_vector(similarity_vector_avg, SEQUENCE + "_avg.pkl")



decimals = 5

if __name__ == '__main__':
    for seq in SEQUENCES:
        print(seq)
        # extract_similarity_gt(seq)

        sim_vec_gt = np.asarray(load_similarity_vector(seq + '_gt.pkl'))
        print('\mu ', np.around(np.mean(sim_vec_gt), decimals), ' \pm ', np.around(np.std(sim_vec_gt), decimals))

        sim_vec_max = np.asarray(load_similarity_vector(seq + '_max.pkl'))
        print('\mu ', np.around(np.mean(sim_vec_max), decimals), ' \pm ', np.around(np.std(sim_vec_max), decimals))

        sim_vec_avg = np.asarray(load_similarity_vector(seq + '_avg.pkl'))
        print('\mu ', np.around(np.mean(sim_vec_avg), decimals), ' \pm ', np.around(np.std(sim_vec_avg), decimals))

        #bbox_vec = np.asarray(load_similarity_vector(seq + '_bbox.pkl'))
        #print('\mu ', np.around(np.mean(bbox_vec), decimals), ' \pm ', np.around(np.std(bbox_vec), decimals))

        #bbox_vec_avg = np.asarray(load_similarity_vector(seq + '_avg_bbox.pkl'))
        #print('\mu ', np.around(np.mean(bbox_vec_avg), decimals), ' \pm ', np.around(np.std(bbox_vec_avg), decimals))

        #vec = sim_vec_gt / (1 - bbox_vec/4)
        #print('\mu ', np.mean(vec), ' \pm ', np.std(vec))

        #vec = sim_vec_avg / (1 - bbox_vec_avg/4)
        #print('\mu ', np.mean(vec), ' \pm ', np.std(vec))
