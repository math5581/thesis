""" Script used to extract the various distance metrics:"""


from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.feature_extraction import FeatureDescription, cosine_similarity, eucledian_distance
from utilities.utilities import *
import matplotlib.pyplot as plt

# Specify_params here:
DIMENSIONS = (224, 224)
# possibly expand with sequences
SEQUENCE = '/workspace/data/MOT17/train/MOT17-02-DPM'
metric = 'euclidian'  # cosine
# if metric == 'euclidian':
metric_to_extract = eucledian_distance


bbox_amp = None
pca_model = None


def extract_similarity_gt():
    """ Extracts similarity between GT """
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, pca_model=pca_model, bbox_amplification=bbox_amp)

    similarity_vector = []
    sequence_length = dataloader.get_seuqence_length()
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)

        bboxes = cv_tools.extract_bbox_from_list(bbox_list)

        # Calculates the feature representation of each detection
        features = feature_extraction.get_feature_description(
            bboxes)
        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                if id in prev_id_list:
                    prev_index = prev_id_list.index(id)
                    similarity_vector.append(metric_to_extract(
                        features[index], prev_feature_list[prev_index]))

        prev_id_list = id_list
        prev_feature_list = features
        dataloader.next_frame()

    save_similarity_vector(
        similarity_vector, "sim_gt_" + str(DIMENSIONS[0]) + "_bboxonly.pkl")


def extract_similarity_not_gt():
    """ Extracts the max similarity and avg similarity between not GT"""
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, pca_model=pca_model, bbox_amplification=bbox_amp)

    similarity_vector_avg = []
    similarity_vector_max = []
    sequence_length = dataloader.get_seuqence_length()
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)

        bboxes = cv_tools.extract_bbox_from_list(bbox_list)

        # Calculates the feature representation of each detection
        features = feature_extraction.get_feature_description(
            bboxes)
        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                temp_similarity = []
                for prev_index, prev_id in enumerate(prev_id_list):
                    if prev_id != id:
                        temp_similarity.append(metric_to_extract(
                            features[index], prev_feature_list[prev_index]))
                # Average
                if len(temp_similarity) != 0:
                    avg_sim = sum(temp_similarity) / len(temp_similarity)
                    similarity_vector_avg.append(avg_sim)
                # Max
                similarity_vector_max.append(min(temp_similarity))
                # Average
        prev_id_list = id_list
        prev_feature_list = features
        dataloader.next_frame()

    save_similarity_vector(similarity_vector_max,
                           "sim_max_not_gt_" + str(DIMENSIONS[0]) + "_bboxonly.pkl")
    save_similarity_vector(similarity_vector_avg,
                           "sim_avg_not_gt_" + str(DIMENSIONS[0]) + "_bboxonly.pkl")


if __name__ == '__main__':
    extract_similarity_not_gt()
    extract_similarity_gt()
    sim_vec_gt = np.asarray(load_similarity_vector(
        'sim_gt_' + str(DIMENSIONS[0]) + '_bboxonly.pkl'))
    print('mean ', np.mean(sim_vec_gt), ' std ', np.std(sim_vec_gt))

    sim_vec_max = np.asarray(load_similarity_vector(
        'sim_max_not_gt_' + str(DIMENSIONS[0]) + '_bboxonly.pkl'))
    print('mean ', np.mean(sim_vec_max), ' std ', np.std(sim_vec_max))

    sim_vec_avg = np.asarray(load_similarity_vector(
        'sim_avg_not_gt_' + str(DIMENSIONS[0]) + '_bboxonly.pkl'))
    print('mean ', np.mean(sim_vec_avg), ' std ', np.std(sim_vec_avg))
