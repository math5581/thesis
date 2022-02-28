import tensorflow as tf
import numpy as np
import pickle as pkl


def cosine_similarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


def eucledian_distance(a, b):
    return np.linalg.norm(a - b)


class DeepFeatureExtraction:
    def __init__(self, DIMENSIONS):
        """ Specify model here if change is needed """
        model = tf.keras.applications.MobileNetV3Large(input_shape=DIMENSIONS + (3,), alpha=1.0, include_top=False,
                                                       weights='imagenet', dropout_rate=0.2)

        out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        out = tf.keras.layers.Flatten()(out)
        out = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(out)
        self.model = tf.keras.Model(model.input, out)
        # print(self.model.summary())

    def extract_features(self, frame):
        image_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        features = self.model.predict(image_tensor, use_multiprocessing=True)
        return features

    def extract_features_batch(self, frame_list):
        array = np.asarray(frame_list)
        features = self.model.predict_on_batch(
            array)
        del array
        return features


# Create somesort of a feature wrapper descriptor thingy.

class FeatureDescription(DeepFeatureExtraction):
    def __init__(self, DIMENSIONS, pca_model=None, bbox_amplification=None, combine=False):
        super().__init__(DIMENSIONS)
        """ Interface for the deep feature extraction """
        """ Specify pca_model to use pca reduction """
        """ Specify amount of bbox_amplification to append bbox"""
        self.bbox_amplification = bbox_amplification
        if pca_model is not None:
            self.pca = pkl.load(open(pca_model, 'rb'))
        else:
            self.pca = None
        self.combine = combine

    def get_feature_description(self, frames=None, bbox_list=None):
        if np.asarray(frames).shape[0] > 1:  # Extract as batch
            features = self.extract_features_batch(frames)
        else:
            features = self.extract_features(frames)

        # Append bbox_amp is specified and there is a bbox_list
        if self.pca is not None:
            features = self.perform_pca_transform(features)
        if (self.bbox_amplification and bbox_list) is not None:
            temp_feat = []
            for feature, bbox in zip(features, bbox_list):
                bb = np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2],
                                 bbox[1] + bbox[3]]) * self.bbox_amplification  # bbox[:4]
                temp_feat.append(np.append(feature, bb, axis=0))
            features = np.asarray(temp_feat)
        return features

    def get_feature_description_combined(self, frames=None, bboxes=None, bbox_list=None, ):
        if np.asarray(frames).shape[0] > 1:  # Extract as batch
            frame_features = self.extract_features_batch(frames)
            if bboxes is not None:  # Extract bounding boxes on top?
                bbox_features = self.extract_features_batch(bboxes)
        else:
            frame_features = self.extract_features(frames)
            if bboxes is not None:
                bbox_features = self.extract_features(bboxes)
        features = []
        if bboxes is not None:
            for frame_feature, bbox_feature in zip(frame_features, bbox_features):
                features.append(np.append(frame_feature, bbox_feature, axis=0))
        else:
            features = frame_features

        # Append bbox_amp is specified and there is a bbox_list
        if self.pca is not None:
            features = self.perform_pca_transform(features)
        if (self.bbox_amplification and bbox_list) is not None:
            temp_feat = []
            for feature, bbox in zip(features, bbox_list):
                bb = np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2],
                                 bbox[1] + bbox[3]]) * self.bbox_amplification  # bbox[:4]
                temp_feat.append(np.append(feature, bb, axis=0))
            features = np.asarray(temp_feat)
        return features

    def perform_pca_transform(self, features):
        feature = np.asarray(features)
        if feature.shape[0] == 0 or feature.shape[0] is None:
            feature = np.reshape(feature, (1, feature.shape[1]))
        return self.pca.transform(feature)
