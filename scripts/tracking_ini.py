import configparser
from TrackEval.trackeval import utils
from TrackEval import trackeval
from utilities.tracker.tracker import *
import matplotlib.pyplot as plt
from utilities.utilities import *
from utilities.feature_extraction import *
from utilities.cv_utilities import *
from utilities.MOT import *
from feature_stuff import pca


# This overwrites everything

DIMENSIONS = (224, 224)
base_path = '/workspace/data/MOT17/train/'
tracker_base_path = '/workspace/evaluation/data/trackers/mot_challenge/MOT17-train/'
SEQUENCE = 'MOT17-02-FRCNN'

pca_model = None
bbox_amp = 0.25


def tracking():
    dataloader = MOTDataloader(os.path.join(base_path, SEQUENCE))
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, bbox_amplification=bbox_amp)
    sequence_length = dataloader.get_seuqence_length()
    frame = dataloader.get_current_frame()

    tracker = Tracker(frame_width=frame.shape[1], frame_height=frame.shape[0])
    out = cv.VideoWriter(
        'project.mp4', cv.VideoWriter_fourcc(*'MP4V'), 25, (frame.shape[1], frame.shape[0]))

    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        # Get the set of current detections
        # dataloader.get_current_gt_bbox_scale()
        bbox_list = dataloader.get_current_det_bbox_scale()
        cv_tools.set_active_frame(frame)
        frame_list = cv_tools.blur_image_list_except_bbox(bbox_list)
        # Calculates the feature representation of each detection
        feature_list = feature_extraction.get_feature_description(
            frame_list, bbox_list)

        tracker.update_tracks(feature_list, bbox_list)
        frame = tracker.draw_active_tracks_on_frame(frame)
        # Get integration?
        tracker.add_existing_tracks_to_df(dataloader.get_current_frame_id())
        out.write(frame)
        dataloader.next_frame()
    # save in
    path = os.path.join(tracker_base_path,
                        str(DIMENSIONS[0]), 'data')
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    out_file = os.path.join(path, SEQUENCE + '.txt')
    tracker.save_track_file(out_file)
    out.release()


def tracking_w_bbox():
    dataloader = MOTDataloader(os.path.join(base_path, SEQUENCE))
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, bbox_amplification=bbox_amp)
    sequence_length = dataloader.get_seuqence_length()
    frame = dataloader.get_current_frame()

    tracker = Tracker(frame_width=frame.shape[1], frame_height=frame.shape[0])
    out = cv.VideoWriter(
        'project.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, (frame.shape[1], frame.shape[0]))

    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        # Get the set of current detections
        # dataloader.get_current_gt_bbox_scale()
        bbox_list = dataloader.get_current_det_bbox_scale()
        cv_tools.set_active_frame(frame)
        frame_list = cv_tools.extract_bbox_from_list(bbox_list)

        # Calculates the feature representation of each detection
        feature_list = feature_extraction.get_feature_description(
            frame_list, bbox_list)

        # print(feature_list)
        tracker.update_tracks(feature_list, bbox_list)
        frame = tracker.draw_active_tracks_on_frame(frame)
        # Get integration?
        tracker.add_existing_tracks_to_df(dataloader.get_current_frame_id())
        out.write(frame)
        dataloader.next_frame()
    # save in
    path = os.path.join(tracker_base_path,
                        str(DIMENSIONS[0]), 'data')
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    out_file = os.path.join(path, SEQUENCE + '.txt')
    tracker.save_track_file(out_file)
    out.release()


def tracking_combined():
    dataloader = MOTDataloader(os.path.join(base_path, SEQUENCE))
    cv_tools = CVTools(DIMENSIONS)
    feature_extraction = FeatureDescription(
        DIMENSIONS, bbox_amplification=bbox_amp)
    sequence_length = dataloader.get_seuqence_length()
    frame = dataloader.get_current_frame()

    tracker = Tracker(frame_width=frame.shape[1], frame_height=frame.shape[0])
    out = cv.VideoWriter(
        'project.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, (frame.shape[1], frame.shape[0]))

    for i in range(sequence_length):
        print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        # Get the set of current detections
        # dataloader.get_current_gt_bbox_scale()
        bbox_list = dataloader.get_current_det_bbox_scale()
        cv_tools.set_active_frame(frame)
        bboxes = cv_tools.extract_bbox_from_list(bbox_list)
        frames = cv_tools.blur_image_list_except_bbox(bbox_list)

        # Calculates the feature representation of each detection
        frame_feature_list, bbox_feature_list = feature_extraction.get_feature_description_combined(
            frames=frames, bbox_list=bbox_list, bboxes=bboxes)
        tracker.update_tracks_combined(
            frame_feature_list, bbox_feature_list, bbox_list)
        frame = tracker.draw_active_tracks_on_frame(frame)
        # Get integration?
        tracker.add_existing_tracks_to_df(dataloader.get_current_frame_id())
        out.write(frame)
        dataloader.next_frame()
    # save in
    path = os.path.join(tracker_base_path,
                        str(DIMENSIONS[0]), 'data')
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    out_file = os.path.join(path, SEQUENCE + '.txt')
    tracker.save_track_file(out_file)
    out.release()


def evaluation(tracker=str(DIMENSIONS[0]), seq=SEQUENCE):
    dataset = dataset_setup(tracker, seq)
    tracker = tracker_base_path + tracker
    metrics_list, metric_names = metric_setup()

    def eval_sequence(seq, dataset, tracker, metrics_list, metric_names):
        """Function for evaluating a single sequence"""
        raw_data = dataset.get_raw_seq_data(tracker, seq)
        seq_res = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls='pedestrian')
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[met_name] = metric.eval_sequence(data)
        return seq_res
    result = eval_sequence(seq, dataset, tracker, metrics_list, metric_names)
    print('HOTA', result['HOTA']['HOTA(0)'])
    print('MOTA', result['CLEAR']['MOTA'])
    print('IDF1', result['Identity']['IDF1'])


def dataset_setup(tracker, seq):
    # Not really useful, but a necessity
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_dataset_config['SEQ_INFO'] = {seq: None}
    default_dataset_config['GT_FOLDER'] = '/workspace/evaluation/data/gt/mot_challenge'
    default_dataset_config['TRACKERS_FOLDER'] = '/workspace/evaluation/data/trackers/mot_challenge/'
    default_dataset_config['TRACKERS_TO_EVAL'] = [tracker]
    dataset = trackeval.datasets.MotChallenge2DBox(default_dataset_config)
    return dataset


def metric_setup(metrics=['HOTA', 'CLEAR', 'Identity']):
    # Metric_setup
    default_metrics_config = {'METRICS': metrics, 'THRESHOLD': 0.5}
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in default_metrics_config['METRICS']:
            metrics_list.append(metric(default_metrics_config))
    metric_names = utils.validate_metrics_list(metrics_list)
    return metrics_list, metric_names


if __name__ == '__main__':
    tracking()
    evaluation()
