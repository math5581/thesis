from utilities.MOT import *
from utilities.cv_utilities import *
from utilities.feature_extraction import *
from utilities.utilities import *
import matplotlib.pyplot as plt
from utilities.metric_learning import KISSME

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DIMENSIONS = (224, 224)
base_path = '/workspace/data/MOT17/train/'
SEQUENCE = '/workspace/data/MOT17/train/MOT17-02-DPM'

SEQUENCES = ['MOT17-02-FRCNN']  # , 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
#  'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']


def extract_features():
    """ Extracts features on all GT every 10'th frame """
    for seq in SEQUENCES:
        SEQUENCE = os.path.join(base_path, seq)
        dataloader = MOTDataloader(SEQUENCE)
        cv_tools = CVTools(DIMENSIONS)
        feature_extraction = FeatureDescription(DIMENSIONS)
        df = pd.DataFrame(columns=[*range(1280)])
        sequence_length = dataloader.get_seuqence_length()
        for i in range(sequence_length):
            if i % 10 == 0:
                # only take every fifth frame.
                print('i ', i, ' of ', sequence_length)
                frame = dataloader.get_current_frame()
                bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
                cv_tools.set_active_frame(frame)
                frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)
                # for frame in frame_list:
                #    features.append(feature_extraction.extract_features(frame))
                print(frame_list[0].shape)
                features = feature_extraction.get_feature_description(
                    frame_list)
                for feat in features:
                    df_length = len(df)
                    df.loc[df_length] = feat
            dataloader.next_frame()

        df.to_pickle("features" + seq + ".pkl")
        del df

# Kiss me stuff here:
# NOT really working...
def feature_enhancement():
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS)  # , bbox_amplification = 10)
    dataloader = MOTDataloader(SEQUENCE)
    df = pd.DataFrame(columns=[*range(1280)])
    sequence_length = dataloader.get_seuqence_length()
    for i in range(sequence_length):
        if i % 10 == 0:
            # only take every fifth frame.
            print('i ', i, ' of ', sequence_length)
            frame = dataloader.get_current_frame()
            bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
            cv_tools.set_active_frame(frame)
            frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)

            features = feature_extraction.get_feature_description(
                frame_list)  # , bbox_list)
            if i == 0:
                feats = features
                full_id_list = np.asarray(id_list)
            else:
                feats = np.append(feats, features, axis=0)
                full_id_list = np.append(
                    full_id_list, np.asarray(id_list), axis=0)
        dataloader.next_frame()
    # The feature enhancement outlies, that it should instead compute mahanobilis distance:
    kissme = KISSME()
    x = feats
    y = np.reshape(full_id_list, (full_id_list.shape[0],))
    print(x.shape, y.shape)
    kissme.fit(x, y)
    kissme.save_m('M_no')


def feature_comparison():
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS)  # , bbox_amplification=10)
    dataloader = MOTDataloader(SEQUENCE)
    sequence_length = dataloader.get_seuqence_length()
    kissme = KISSME(m_name_load='M_no')
    similarity_vector = []
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)
        frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)

        features = feature_extraction.get_feature_description(
            frame_list)  # , bbox_list)
        if "prev_id_list" in locals():
            print(id_list)
            print(prev_id_list)
            distances = kissme.get_distance(features, prev_feature_list)
            # print(distances)
            for index, id in enumerate(id_list):
                if id in prev_id_list:
                    prev_index = prev_id_list.index(id)
                    print(index, prev_index)
                    print(distances[index, prev_index])
                    similarity_vector.append(distances[index, prev_index])
                    # features[index], prev_feature_list[prev_index]))

        prev_id_list = id_list
        prev_feature_list = features
        dataloader.next_frame()
    save_similarity_vector(
        similarity_vector, "sim_gt_" + str(DIMENSIONS[0]) + "_M.pkl")
    # The feature enhancement outlies, that it should instead compute mahanobilis distance:


def feature_comparison_not_gt():
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS)  # , bbox_amplification=10)
    dataloader = MOTDataloader(SEQUENCE)
    sequence_length = dataloader.get_seuqence_length()
    kissme = KISSME(m_name_load='M_no')
    similarity_vector_avg = []
    similarity_vector_max = []
    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        bbox_list, id_list = dataloader.get_current_gt_bbox_scale()
        cv_tools.set_active_frame(frame)
        frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)

        features = feature_extraction.get_feature_description(
            frame_list)  # , bbox_list)
        if "prev_id_list" in locals():
            distances = kissme.get_distance(features, prev_feature_list)
            for index, id in enumerate(id_list):
                temp_similarity = []
                for prev_index, prev_id in enumerate(prev_id_list):
                    if prev_id != id:
                        if id in prev_id_list:
                            prev_index = prev_id_list.index(id)
                            temp_similarity.append(
                                distances[index, prev_index])
                        # features[index], prev_feature_list[prev_index]))
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
                           "sim_max_not_gt_" + str(DIMENSIONS[0]) + "_M.pkl")
    save_similarity_vector(similarity_vector_avg,
                           "sim_avg_not_gt_" + str(DIMENSIONS[0]) + "_M.pkl")


def pca():
    major_df = pd.DataFrame(columns=[*range(1280)])
    for seq in SEQUENCES:
        fil = os.path.join('data/pca/sigmoid/224/features',
                           ('features' + seq + '.pkl'))
        df = pd.read_pickle(fil)
        major_df = major_df.append(df, ignore_index=True)

    x = major_df.values
    f_test = x[:2]
    x = StandardScaler().fit_transform(x)  # Standardize the data.
    pca = PCA(0.75)  # We want 95 % of the data.
    pca.fit(x)
    print(pca.components_.shape)
    pkl.dump(pca, open("data/pca/sigmoid/224/pca_075.pkl", "wb"))
    # plot_bar(pca)


def plot_bar(pca):
    #
    # Determine explained variance using explained_variance_ration_ attribute
    #
    exp_var_pca = pca.explained_variance_ratio_
    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    #
    # Create the visualization plot
    #
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5,
            align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues,
             where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('pca.png')


if __name__ == '__main__':
    # feature_comparison_not_gt()
    # feature_enhancement()
    # feature_comparison()
    sim_vec_avg = np.asarray(load_similarity_vector(
        'sim_gt_224_M.pkl'))
    print('mean ', np.mean(sim_vec_avg), ' std ', np.std(sim_vec_avg))

    sim_vec_avg = np.asarray(load_similarity_vector(
        'sim_max_not_gt_224_M.pkl'))
    print('mean ', np.mean(sim_vec_avg), ' std ', np.std(sim_vec_avg))

    sim_vec_avg = np.asarray(load_similarity_vector(
        'sim_avg_not_gt_224_M.pkl'))
    print('mean ', np.mean(sim_vec_avg), ' std ', np.std(sim_vec_avg))
