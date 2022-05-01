import tensorflow as tf
import numpy as np
import pickle as pkl
from utilities.utilities import *
from utilities.description import Description
import tensorflow_hub as hub
import os
from tensorflow.keras import layers, Model

def cosine_similarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


def eucledian_distance(a, b):
    return np.linalg.norm(a - b)


class DeepFeatureExtraction:
    def __init__(self, DIMENSIONS):
        """ Specify model here if change is needed """
        # For basleine
        #model = tf.keras.applications.MobileNetV3Large(input_shape=DIMENSIONS + (3,), alpha=1.0, include_top=False,
        #                                               weights='imagenet', dropout_rate=0.2)

        #out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        #out = tf.keras.layers.Flatten()(out)
        # out = tf.keras.layers.Activation(tf.keras.activations.tanh)(out)
        #self.model = tf.keras.Model(model.input, out)
        #print(self.model.summary())

        self.model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
                    trainable=False)
            #tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.build([None, DIMENSIONS[0], DIMENSIONS[1], 3])  # Batch input shape.

        #model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(input_shape=DIMENSIONS + (3,),  include_top=False,
        #                                               weights='imagenet', include_preprocessing=True)

        #out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        #out = tf.keras.layers.Flatten()(out)
        # out = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(out)
        #self.model = tf.keras.Model(model.input, out)
        # print(self.model.summary())
        
        # FOR THE NORMAL STUFF...
        #self.model = tf.keras.Sequential([
        #    hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
        #            trainable=False)
            #tf.keras.layers.Dense(num_classes, activation='softmax')
        #])
        #self.model.build([None, DIMENSIONS[0], DIMENSIONS[1], 3])  # Batch input shape.


    def extract_features(self, frame):
        # frame = frame/255
        image_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        features = self.model.predict(image_tensor, use_multiprocessing=True)
        return features[0]

    def extract_features_batch(self, frame_list):
        array = np.asarray(frame_list)
        array = array/255
        features = self.model.predict_on_batch(
                array)
        del array
        return np.asarray(features)



# Create somesort of a feature wrapper descriptor thingy.

class FeatureDescription(DeepFeatureExtraction):
    def __init__(self, DIMENSIONS, pca_model=None, bbox_amplification=None, combine=False):
        super().__init__(DIMENSIONS)
        """ Interface for the deep feature extraction """
        """ Specify pca_model to use pca reduction DEPRECATED"""
        """ Specify amount of bbox_amplification to append bbox"""

        self.bbox_amplification = bbox_amplification
        if pca_model is not None:
            self.pca = pkl.load(open(pca_model, 'rb'))
        else:
            self.pca = None
        self.combine = combine

    def feature_description_bbox_normalized(self, frames=None, bbox_list=None):
        if np.asarray(frames).shape[0] > 1:  # Extract as batch
            features = self.extract_features_batch(frames)
        else:
            features = self.extract_features(frames)
        features = self.normalize_features(features, bbox_list)
        # Append bbox_amp is specified and there is a bbox_list
        if self.pca is not None:  # Deprecated
            features = self.perform_pca_transform(features)
        if (self.bbox_amplification and bbox_list) is not None:
            temp_feat = []
            for feature, bbox in zip(features, bbox_list):
                bb = np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2],
                                 bbox[1] + bbox[3]]) * self.bbox_amplification  # bbox[:4]
                temp_feat.append(np.append(feature, bb, axis=0))
            features = np.asarray(temp_feat)
        return features

    def normalize_features(self, feature_list, bboxes):
        """ Normalized with size of bbox """
        # DEPRECATED
        new_feat_list = []
        for feat, bbox in zip(feature_list, bboxes):
            area = bbox[2] * bbox[3]
            new_feat_list.append(np.asarray(feat) / area)
        return new_feat_list

    def get_feature_description(self, frames=None, bbox_list=None):
        if np.asarray(frames).shape[0] > 1:  # Extract as batch
            features = self.extract_features_batch(frames)
        else:
            features = self.extract_features(frames)
        # features = self.normalize_features(features, bbox_list)
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

    def get_detection_features(self, frames=None, bboxes=None, bbox_list=None):
        """ Returns the feature description for the current detection """
        if len(frames) == 0:
            return []
        #if np.asarray(frames).shape[0] > 1:  # Extract as batch
            # change back here
            #features = [] 
            #for frame in frames:
            #    features.append(self.extract_features(frame))
            #features = np.squeeze(features)
            #print(features.shape)
        features = self.extract_features_batch(frames)
        #features = []
        #for frame in frames:
        #    features.append(self.extract_features(frame))
        #if bboxes is not None:  # Extract bounding boxes on top?
        #    bbox_features = self.extract_features_batch(bboxes)

        #else:
            # features = self.extract_features(frames)
        #    if bboxes is not None:
        #        bbox_features = self.extract_features(bboxes)

        description_arr = []
        for feature, bbox in zip(features, bbox_list):
            description_arr.append(Description(bbox, feature))
        return description_arr

    def save_detection_features(self, det_features, folder, sequence, frame_number):
        # Change to save list of detection objects as pkl.
        feat_vector = np.asarray([feat() for feat in det_features])
        path_out = os.path.join(folder, sequence)
        os.makedirs(path_out, exist_ok=True)
        out_file = os.path.join(path_out,  str(frame_number) + '.npy')
        np.save(out_file, feat_vector)
    
    @staticmethod
    def load_detection_features(folder, sequence, frame_number, bbox_list):
        # Change to load list of detction objects as pkl.
        file = os.path.join(folder, sequence,  str(frame_number) + '.npy')
        vectors = np.load(file)
        return [Description(bbox, vec) for vec, bbox in zip(vectors, bbox_list)]

    def append_features(self, feats1, feats2):
        temp_feats = []
        for feat1, feat2 in zip(feats1, feats2):
            temp_feats.append(np.append(feat1, feat2, axis=0))
        return temp_feats

    def perform_pca_transform(self, features):
        feature = np.asarray(features)
        if feature.shape[0] == 0 or feature.shape[0] is None:
            feature = np.reshape(feature, (1, feature.shape[1]))
        return self.pca.transform(feature)

