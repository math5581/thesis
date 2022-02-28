""" Script used to extract the various distance metrics:"""


from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.feature_extraction import FeatureDescription, cosine_similarity, eucledian_distance
from utilities.utilities import *
import matplotlib.pyplot as plt

# Specify_params here:
DIMENSIONS = (448, 448)
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

        frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)
        features = feature_extraction.get_feature_description(
            frame_list, bbox_list)

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
        similarity_vector, "sim_" + str(DIMENSIONS[0]) + "_gt.pkl")


def extract_similarity_gt_location_different_roi():
    """ Extracts similarity between GT locations with a different bbox/roi on previous """
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, pca_model=pca_model)

    similarity_vector = []
    sequence_length = dataloader.get_seuqence_length()
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)

        # Feature extraction
        frame_list_gt = cv_tools.blur_image_list_except_bbox(bbox_list)
        features_gt = feature_extraction.get_feature_description(frame_list_gt)

        # Feature extraction other bbox
        new_bbox_list = bbox_list[1:] + [bbox_list[0]]
        frame_list_other_bbox = cv_tools.blur_image_list_except_bbox_newbb(
            bbox_list, new_bbox_list)
        features_gt_other = feature_extraction.get_feature_description(
            frame_list_other_bbox)

        # Perform similarity between ground truths
        if "prev_id_list" in locals():
            for index, id in enumerate(id_list):
                if id in prev_id_list:
                    prev_index = prev_id_list.index(id)
                    similarity_vector.append(metric_to_extract(
                        features_gt[index], prev_feature_list[prev_index]))
        prev_id_list = id_list
        prev_feature_list = features_gt_other
        dataloader.next_frame()

    save_similarity_vector(
        similarity_vector, "sim_gt_" + str(DIMENSIONS[0]) + "_new_roi.pkl")


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

        frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)
        features = feature_extraction.get_feature_description(
            frame_list, bbox_list)
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

        prev_id_list = id_list
        prev_feature_list = features
        dataloader.next_frame()

    save_similarity_vector(similarity_vector_max,
                           "sim_max_not_gt_" + str(DIMENSIONS[0]) + ".pkl")
    save_similarity_vector(similarity_vector_avg,
                           "sim_avg_not_gt_" + str(DIMENSIONS[0]) + ".pkl")


# Deprecated functions
def blurring_similarity(kernel):
    dataloader = MOTDataloader(SEQUENCE)
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(DIMENSIONS)

    similarity_vector = []
    sequence_length = dataloader.get_seuqence_length()
    kernel_size = (kernel, kernel)
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        cv_tools.set_active_frame(frame)

        blur_frame = cv_tools.gaussian_blur(
            cv_tools.get_current_frame(), kernel_size)
        save_image_file('test35.png', blur_frame)
        features = feature_extraction.extract_features(
            blur_frame)
        # Perform similarity between ground truths
        if "prev_feature" in locals():
            similarity_vector.append(cosine_similarity(
                features[0], prev_feature[0]))

        prev_feature = features
        dataloader.next_frame()

    save_similarity_vector(
        similarity_vector, "sim_vec_blur_" + str(kernel) + ".pkl")


def visualize_blur(path):
    """ Plots the mean similarity accross sequence."""
    ran = [i for i in range(1, 48, 2)]
    temp = []
    print(ran)
    for i in ran:
        temp.append(np.mean(load_similarity_vector(os.path.join(path,
                                                                'sim_vec_blur_' + str(i) + '.pkl'))))
    plt.plot(ran, temp)
    plt.savefig('blur_effect.png')


if __name__ == '__main__':
    extract_similarity_gt_location_different_roi()
    extract_similarity_not_gt()
    extract_similarity_gt()
    sim_vec_gt = np.asarray(load_similarity_vector(
        'sim_' + str(DIMENSIONS[0]) + '_gt.pkl'))
    print('mean ', np.mean(sim_vec_gt), ' std ', np.std(sim_vec_gt))

    sim_vec_max = np.asarray(load_similarity_vector(
        'sim_max_not_gt_' + str(DIMENSIONS[0]) + '.pkl'))
    print('mean ', np.mean(sim_vec_max), ' std ', np.std(sim_vec_max))

    sim_vec_avg = np.asarray(load_similarity_vector(
        'sim_avg_not_gt_' + str(DIMENSIONS[0]) + '.pkl'))
    print('mean ', np.mean(sim_vec_avg), ' std ', np.std(sim_vec_avg))

    sim_vec_gt_new_roi = np.asarray(load_similarity_vector(
        'sim_gt_' + str(DIMENSIONS[0]) + '_new_roi.pkl'))
    print('mean ', np.mean(sim_vec_gt_new_roi),
        ' std ', np.std(sim_vec_gt_new_roi))
